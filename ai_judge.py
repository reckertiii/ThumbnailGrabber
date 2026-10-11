"""Second-opinion review of candidate captures by a vision model.

The heuristic analyzer scores individual signals — jaw open, eyes shut, hands over the
torso — and each one can pass while the photo as a whole is still wrong: arms folded,
a finger pointed at the floor, a face caught between two expressions. Nothing in the
pipeline looked at the frame the way a person picking a slide image does. This does.

A candidate only reaches the judge after it has cleared the heuristics *and* would
improve the kept set, so calls are sparse — a few dozen per service, not one per frame.
The verdict is blended into the capture's score and can veto a save outright.

Two backends. The Claude API gives the sharpest judgement; a local Ollama vision model
(qwen3-vl by default) runs on the same box with no per-call cost and nothing leaving
the network, at the price of slower and somewhat blunter reviews. Because review runs
on its own thread and a candidate is skipped rather than queued while one is in flight,
a slow local model simply reviews fewer frames — it never stalls capture.

Entirely optional. With neither credentials nor a reachable Ollama, the judge stays
off and says so once.
"""
import os
import json
import time
import base64
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional, List, Any, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from config import AppConfig


class JudgeVerdict(BaseModel):
    """What the judge returns. Kept flat so it maps straight onto the gallery UI.

    Field order matters: the JSON is generated in schema order, so listing the flaws
    comes before committing to a number. Scored first, a small model reaches for the
    same middling value every time; made to enumerate first, the score follows from
    what it actually found.
    """
    issues: List[str] = Field(
        description="Every flaw you see, worst first, at most 5, each under 10 words and ending "
                    "with (small), (moderate) or (serious). Empty list only if the frame is flawless.")
    strengths: List[str] = Field(
        description="At most 3 things that work, each under 8 words. Never list a flaw here.")
    score: float = Field(
        ge=0.0, le=1.0,
        description="1.0 minus 0.1 per (small), 0.2 per (moderate), 0.4 per (serious) item in issues; floor 0.")
    usable: bool = Field(description="False only if a fatal flaw is present.")
    summary: str = Field(description="One sentence, under 25 words, as a person would say it.")


@dataclass
class JudgeResult:
    verdict: Optional[JudgeVerdict]
    blended_score: float
    latency_sec: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    skipped: bool = False   # True when the judge was off or rate-limited, not when it failed


# The rubric is the whole point, so it is spelled out rather than left to the model to
# infer. It describes the end use, because "is this a good photo" depends entirely on
# what the photo is for.
SYSTEM_PROMPT = """You review still frames pulled from a video of a speaker on stage. Each frame you \
see has already been cropped to the speaker. The frame will have its background removed and \
the speaker placed onto a presentation slide as a cutout, next to the sermon title. You are \
choosing the best available frame from a live recording, not judging a studio portrait - \
most frames have some flaw, and your job is to rank them, not to reject them.

What makes a frame better:
- Eyes open and engaged, toward the camera or audience. Glancing at notes is a small flaw; \
eyes closed or mid-blink is a serious one.
- Mouth closed, or a natural pleasant expression. A slight smile is best. Caught mid-word is \
a moderate flaw; a grimace or half-formed expression is worse.
- Open posture. Hands gesturing outward or resting naturally are good. Arms folded or \
hands clasped tight against the chest read as closed-off. Hands simply being somewhere in \
front of the body is normal for someone speaking and is measured separately - only raise \
it when they genuinely merge into the torso as a single shape.
- Expression and gaze are what you are most useful for, because they are the hardest to \
measure and the easiest for a person to notice. Weigh them above everything else: is he \
engaged and present, or caught between two expressions, glancing away, or blank?
- Nothing crossing the body. A microphone stand, iPad, or lectern edge in front of the torso \
is a serious flaw because it breaks the cutout. A hand in front of the torso is a moderate \
flaw. Anything covering the face is fatal.
- Sharp. Look specifically for motion blur - a smeared hand or arm caught mid-gesture, or a smeared face. Blur on a raised or gesturing hand is a real flaw, because that hand is part of the cutout and will look wrong frozen on a slide. Call it out when you see it: moderate when a hand or arm is smeared, serious when the face is.
- Whole head in frame with a little room above it.

First list every flaw you actually see, tagging each (small), (moderate) or (serious), \
then the strengths. Then compute the score: start at 1.0 and subtract 0.1 for each small \
flaw, 0.2 for each moderate flaw, 0.4 for each serious flaw, not going below 0. A frame \
with no flaws is 1.0; one mid-word and one hand across the torso is 0.6. Do not round \
toward the middle - the score must follow from the list.

Set usable to false only for a fatal flaw: eyes closed, face blurred or covered, head cut \
off, or an object across the face. Everything else is usable with its flaws listed."""


def _normalise_verdict_json(text: str) -> str:
    """Small models sometimes answer on a 0-100 scale despite the schema saying 0-1.
    The JSON-schema `maximum` is not enforced by Ollama's grammar, so fix it here."""
    try:
        obj = json.loads(text)
    except Exception:  # noqa: BLE001 - let the caller's parser report it
        return text
    score = obj.get("score")
    if isinstance(score, (int, float)) and score > 1.0:
        obj["score"] = max(0.0, min(1.0, float(score) / 100.0))
        return json.dumps(obj)
    return text


def _keep_alive_value(raw: str):
    """Ollama takes keep_alive as a duration string ("30m") or a number of seconds,
    where -1 means never unload. A bare number arriving as a string from an env var
    is rejected with 'missing unit in duration', so numerics are sent as numbers.
    """
    text = str(raw).strip()
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _encode_image_for_judge(frame_bgr: np.ndarray, max_px: int) -> str:
    """Resizes and JPEG-encodes a BGR frame to a base64 string for the API."""
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_px / float(max(h, w)))
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("could not encode frame for the judge")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


class AIJudge:
    """Optional vision-model review of a candidate capture.

    Thread-safe for one call at a time. Callers should check `available` and expect
    `judge()` to return quickly with `skipped=True` when it declines to run.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.enabled = bool(getattr(config, "ai_judge_enabled", False))
        self.backend = str(getattr(config, "ai_judge_backend", "auto")).strip().lower()
        self.model = getattr(config, "ai_judge_model", "claude-opus-5")
        self.effort = getattr(config, "ai_judge_effort", "medium")
        self.ollama_url = str(getattr(config, "ai_judge_ollama_url", "http://localhost:11434")).rstrip("/")
        self.ollama_model = getattr(config, "ai_judge_ollama_model", "qwen3-vl:4b")
        self.ollama_keep_alive = str(getattr(config, "ai_judge_ollama_keep_alive", "30m"))
        self.ollama_num_thread = int(getattr(config, "ai_judge_ollama_num_thread", 0) or 0)
        self.timeout = float(getattr(config, "ai_judge_timeout_sec", 120.0))
        self.weight = float(getattr(config, "ai_judge_weight", 0.5))
        self.max_px = int(getattr(config, "ai_judge_image_max_px", 768))
        self.min_interval = float(getattr(config, "ai_judge_min_interval_sec", 3.0))
        self.max_calls = int(getattr(config, "ai_judge_max_calls_per_session", 300))
        self.active_seconds = max(0.0, float(getattr(config, "ai_judge_active_hours", 2.0)) * 3600.0)
        self.started_at: float = time.time()
        self._expiry_handled: bool = False
        self._preloading: bool = False      # a load is in flight
        self._preload_error: str = ""

        self._client: Any = None            # Anthropic client, when that backend is active
        self._client_error: Optional[str] = None
        self._lock = threading.Lock()
        self._last_call_at: float = 0.0
        self.calls_made: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self._announced = False

        if self.enabled:
            self._select_backend()

    # ------------------------------------------------------------------
    def _select_backend(self) -> None:
        """Picks a backend without raising. Order for "auto": Claude if credentials
        resolve, else a local Ollama if one answers and has the model, else off."""
        if self.backend == "off":
            self.enabled = False
            self._client_error = "disabled (ai_judge_backend=off)"
            return

        errors: List[str] = []
        if self.backend in ("anthropic", "auto"):
            err = self._init_anthropic()
            if err is None:
                self.backend = "anthropic"
                return
            errors.append(err)
            if self.backend == "anthropic":
                self.enabled = False
                self._client_error = err
                return

        if self.backend in ("ollama", "auto"):
            err = self._init_ollama()
            if err is None:
                self.backend = "ollama"
                return
            errors.append(err)

        self.enabled = False
        self._client_error = "; ".join(errors) if errors else f"unknown backend '{self.backend}'"

    def _init_anthropic(self) -> Optional[str]:
        """Returns None on success, else a short reason. Never raises."""
        try:
            import anthropic  # deferred so the rest of the app has no hard dependency
        except ImportError:
            return "the 'anthropic' package is not installed (pip install anthropic)"

        # The SDK resolves credentials itself (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
        # or an `ant auth login` profile), so the only cheap check here is whether any
        # of those exist. A wrong key surfaces on the first call and disables the judge.
        has_env = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        has_profile = os.path.isdir(os.path.expanduser("~/.config/anthropic"))
        if not has_env and not has_profile:
            return "no Anthropic credentials (set ANTHROPIC_API_KEY)"
        try:
            self._client = anthropic.Anthropic(max_retries=1, timeout=self.timeout)
        except Exception as e:  # noqa: BLE001 - anything here just means "off"
            return f"could not create Anthropic client: {e}"
        return None

    def _init_ollama(self) -> Optional[str]:
        """Checks that an Ollama server answers and already has the model pulled."""
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - unreachable server is a normal state
            return f"no Ollama at {self.ollama_url} ({type(e).__name__})"

        names = {m.get("name", "") for m in tags.get("models", [])}
        # "qwen3-vl:4b" is stored as exactly that; a bare "qwen3-vl" means ":latest".
        want = self.ollama_model if ":" in self.ollama_model else f"{self.ollama_model}:latest"
        if want not in names and self.ollama_model not in names:
            return (f"Ollama at {self.ollama_url} is up but '{self.ollama_model}' is not pulled "
                    f"(run: ollama pull {self.ollama_model})")
        return None

    # ------------------------------------------------------------------
    # Active window
    #
    # The judge is only wanted while a service is actually being recorded. Left to its
    # own devices with keep_alive=-1 it would hold 20 GB of weights on the Ollama host
    # from Sunday to Sunday. So reviewing stops after a fixed window from startup and
    # the model is unloaded, releasing that memory until someone re-arms it.
    # ------------------------------------------------------------------

    @property
    def seconds_remaining(self) -> float:
        """Seconds left in the active window; inf when the window is disabled."""
        if self.active_seconds <= 0:
            return float("inf")
        return max(0.0, self.active_seconds - (time.time() - self.started_at))

    @property
    def expired(self) -> bool:
        return self.seconds_remaining <= 0

    def reset_window(self) -> None:
        """Re-arms the judge for another full window without restarting the app."""
        self.started_at = time.time()
        self._expiry_handled = False
        if self.enabled:
            print(f"[AIJudge] Woken for another {self.active_seconds / 3600:.1f}h.")

    def wake(self) -> None:
        """Re-arms the window and pulls the model back into memory.

        Resetting the timer alone left Ollama idle at a few MB until the first
        review, so the first capture of a service paid for loading 20GB off disk.
        Waking before the service is exactly when you want that cost paid, so the
        load is started here and runs on its own thread -- it can take minutes,
        which is far too long to hold an HTTP request open.
        """
        self.reset_window()
        self.preload_async()

    def preload_async(self) -> None:
        """Starts a model load in the background, if one is not already running."""
        if self.backend != "ollama" or not self.enabled:
            return
        if self._preloading:
            return
        self._preloading = True
        self._preload_error = ""
        threading.Thread(target=self._load_ollama_model, daemon=True).start()

    def _load_ollama_model(self) -> None:
        """Asks Ollama to hold the model in memory.

        An empty prompt is Ollama's documented way to load without generating. The
        keep_alive sent here is the configured one, so the model stays resident on
        the same terms a real review would give it.
        """
        started = time.time()
        try:
            body = json.dumps({
                "model": self.ollama_model,
                "prompt": "",
                "keep_alive": _keep_alive_value(self.ollama_keep_alive),
            }).encode()
            req = urllib.request.Request(f"{self.ollama_url}/api/generate", data=body,
                                         headers={"Content-Type": "application/json"},
                                         method="POST")
            # Generous: a 20GB model off spinning disk is minutes, not seconds.
            with urllib.request.urlopen(req, timeout=600) as resp:
                resp.read()
            print(f"[AIJudge] Loaded {self.ollama_model} on {self.ollama_url} "
                  f"in {time.time() - started:.0f}s.")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:200]
            self._preload_error = f"HTTP {e.code}: {detail}"
            print(f"[AIJudge] Could not load {self.ollama_model}: {self._preload_error}")
        except Exception as e:  # noqa: BLE001
            self._preload_error = f"{type(e).__name__}: {e}"
            print(f"[AIJudge] Could not load {self.ollama_model}: {self._preload_error}")
        finally:
            self._preloading = False

    def sleep_now(self) -> None:
        """Ends the window early and releases the model."""
        self.started_at = time.time() - max(self.active_seconds, 1.0)
        self._handle_expiry()

    def _handle_expiry(self) -> None:
        if self._expiry_handled:
            return
        self._expiry_handled = True
        hours = self.active_seconds / 3600.0
        print(f"[AIJudge] {hours:.1f}h active window elapsed - reviews are off.")
        self._unload_ollama_model()

    def _unload_ollama_model(self) -> None:
        """Asks Ollama to drop the model from memory.

        A zero keep_alive is Ollama's documented way to evict immediately; the server
        answers with done_reason "unload". Best effort - a host that has already gone
        away needs no eviction.
        """
        if self.backend != "ollama":
            return
        try:
            body = json.dumps({"model": self.ollama_model, "keep_alive": 0}).encode()
            req = urllib.request.Request(f"{self.ollama_url}/api/generate", data=body,
                                         headers={"Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=20) as resp:
                resp.read()
            print(f"[AIJudge] Released {self.ollama_model} on {self.ollama_url}.")
        except Exception as e:  # noqa: BLE001 - eviction is advisory
            print(f"[AIJudge] Could not unload {self.ollama_model}: {type(e).__name__}")

    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        if self.expired:
            # Checked on every capture and every telemetry tick, so expiry is noticed
            # and the model released promptly even when nothing is being captured.
            self._handle_expiry()
            return False
        if self.backend == "anthropic":
            return self._client is not None
        return self.backend == "ollama"

    @property
    def active_model(self) -> str:
        return self.ollama_model if self.backend == "ollama" else self.model

    def status_line(self) -> str:
        if self.available:
            where = "local" if self.backend == "ollama" else "Claude API"
            left = self.seconds_remaining
            window = "" if left == float("inf") else f", {int(left // 60)} min left"
            if self._preloading:
                return (f"AI judge waking ({self.active_model} loading on {where}"
                        f"{window})")
            if self._preload_error:
                return (f"AI judge on ({self.active_model} via {where}, "
                        f"{self.calls_made} calls{window}) - load failed: "
                        f"{self._preload_error}")
            return f"AI judge on ({self.active_model} via {where}, {self.calls_made} calls{window})"
        if self.enabled and self.expired:
            return (f"AI judge asleep - {self.active_seconds / 3600:.1f}h window elapsed "
                    f"({self.calls_made} calls). Model released.")
        return f"AI judge off: {self._client_error or 'disabled in config'}"

    def announce_once(self) -> None:
        if not self._announced:
            self._announced = True
            print(f"[AIJudge] {self.status_line()}")

    # ------------------------------------------------------------------
    def judge(self, subject_crop_bgr: np.ndarray, heuristic_score: float,
              heuristic_notes: Optional[List[str]] = None) -> JudgeResult:
        """Reviews one subject crop. Never raises; failures come back in `error`.

        `heuristic_score` is blended with the verdict. On any failure the heuristic
        score is returned unchanged so a network blip can't block a capture.
        """
        self.announce_once()
        if not self.available:
            return JudgeResult(verdict=None, blended_score=heuristic_score, skipped=True)

        with self._lock:
            now = time.time()
            if self.calls_made >= self.max_calls:
                if self.calls_made == self.max_calls:
                    print(f"[AIJudge] Reached ai_judge_max_calls_per_session ({self.max_calls}); "
                          f"judge is off for the rest of this session.")
                    self.calls_made += 1
                return JudgeResult(verdict=None, blended_score=heuristic_score, skipped=True)
            if now - self._last_call_at < self.min_interval:
                return JudgeResult(verdict=None, blended_score=heuristic_score, skipped=True)
            self._last_call_at = now
            self.calls_made += 1

        notes = ""
        if heuristic_notes:
            # Only observations. Anything mentioning a score is our verdict, not a fact,
            # and a number in the prompt is the one thing the model reliably copies.
            observations = [n for n in heuristic_notes if "score" not in n.lower() and "%" not in n]
            if observations:
                notes = "\nThe automated checks flagged: " + "; ".join(observations) + "."

        # The heuristic score is deliberately NOT mentioned: given "65%", the model
        # tends to answer 65. The judge is meant to be an independent second opinion,
        # so it only sees the flags — those are observations, not a verdict.
        user_text = f"Review this frame for use as a slide cutout.{notes}"

        started = time.time()
        try:
            image_b64 = _encode_image_for_judge(subject_crop_bgr, self.max_px)
            if self.backend == "ollama":
                verdict, in_tok, out_tok, err = self._call_ollama(image_b64, user_text)
            else:
                verdict, in_tok, out_tok, err = self._call_anthropic(image_b64, user_text)
        except Exception as e:  # noqa: BLE001 - see below
            # Every failure is treated the same way: log it, disable on auth problems,
            # and let the heuristic score stand. The capture pipeline must never stall
            # on the network.
            msg = f"{type(e).__name__}: {e}"
            self._handle_failure(e, msg)
            return JudgeResult(verdict=None, blended_score=heuristic_score,
                               latency_sec=time.time() - started, error=msg)

        latency = time.time() - started
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok

        if verdict is None:
            return JudgeResult(verdict=None, blended_score=heuristic_score, latency_sec=latency,
                               input_tokens=in_tok, output_tokens=out_tok,
                               error=err or "no structured verdict in response")

        blended = (1.0 - self.weight) * heuristic_score + self.weight * float(verdict.score)
        return JudgeResult(verdict=verdict, blended_score=round(max(0.0, min(1.0, blended)), 3),
                           latency_sec=latency, input_tokens=in_tok, output_tokens=out_tok)

    def _call_anthropic(self, image_b64: str, user_text: str
                        ) -> Tuple[Optional[JudgeVerdict], int, int, Optional[str]]:
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            output_config={"effort": self.effort},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                    {"type": "text", "text": user_text},
                ],
            }],
            output_format=JudgeVerdict,
        )
        usage = getattr(response, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        if getattr(response, "stop_reason", None) == "refusal":
            return None, in_tok, out_tok, "refusal"
        return getattr(response, "parsed_output", None), in_tok, out_tok, None

    def _call_ollama(self, image_b64: str, user_text: str
                     ) -> Tuple[Optional[JudgeVerdict], int, int, Optional[str]]:
        """One /api/chat round trip with the verdict schema as the output format.

        `format` constrains the model to the JudgeVerdict JSON schema; `keep_alive`
        holds the weights in memory between reviews, which matters on CPU where a cold
        load can cost as much as the inference itself.
        """
        body = {
            "model": self.ollama_model,
            "stream": False,
            # Qwen3-family models reason out loud by default. That slows every review
            # and there is nothing for it to reason about here — the rubric is fixed
            # and the answer is a short JSON object.
            "think": False,
            "keep_alive": _keep_alive_value(self.ollama_keep_alive),
            # num_predict bounds a runaway answer: a verdict is ~150 tokens, and one
            # that rambled to 800+ took four minutes and came back malformed.
            "options": ({"temperature": 0, "num_predict": 400, "num_thread": self.ollama_num_thread}
                        if self.ollama_num_thread > 0 else {"temperature": 0, "num_predict": 400}),
            "format": JudgeVerdict.model_json_schema(),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text, "images": [image_b64]},
            ],
        }
        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Ollama puts the reason in the body; a bare status code is useless here.
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                detail = ""
            raise RuntimeError(f"Ollama HTTP {e.code}: {detail or e.reason}") from None

        in_tok = int(data.get("prompt_eval_count", 0) or 0)
        out_tok = int(data.get("eval_count", 0) or 0)
        msg = data.get("message") or {}
        text = msg.get("content", "") or ""
        if not text.strip():
            # With think=false on a thinking-capable model, Ollama has been observed to
            # deliver the answer in `thinking` and leave `content` empty.
            text = msg.get("thinking", "") or ""
        if not text.strip():
            return None, in_tok, out_tok, "empty response from Ollama"

        # Small models occasionally wrap the JSON in prose or fences despite `format`.
        # Take the outermost object rather than failing the whole review on that.
        try:
            verdict = JudgeVerdict.model_validate_json(_normalise_verdict_json(text))
        except Exception:
            lo, hi = text.find("{"), text.rfind("}")
            if lo < 0 or hi <= lo:
                return None, in_tok, out_tok, f"unparseable verdict: {text[:120]!r}"
            try:
                verdict = JudgeVerdict.model_validate_json(_normalise_verdict_json(text[lo:hi + 1]))
            except Exception as e:  # noqa: BLE001
                return None, in_tok, out_tok, f"invalid verdict JSON: {e}"
        return verdict, in_tok, out_tok, None

    def _handle_failure(self, exc: Exception, msg: str) -> None:
        if self.backend == "anthropic":
            try:
                import anthropic
                if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
                    self.enabled = False
                    self._client_error = f"credentials rejected ({msg}); judge disabled"
                    print(f"[AIJudge] {self._client_error}")
                    return
            except ImportError:
                pass
        # A local server that went away mid-session may well come back; keep trying.
        print(f"[AIJudge] Review failed, keeping heuristic score: {msg}")


def _main() -> int:
    """`python ai_judge.py [image]` — report which backend is active and, given an
    image, run one review against it. The quickest way to confirm a NAS-hosted Ollama
    is reachable and has the model, from wherever the app will actually run.
    """
    import sys
    import argparse

    ap = argparse.ArgumentParser(description="Check the AI judge backend and optionally review one image.")
    ap.add_argument("image", nargs="?", help="Path to a JPEG/PNG to review (ideally a subject crop)")
    ap.add_argument("--backend", choices=["auto", "anthropic", "ollama", "off"], help="Override ai_judge_backend")
    ap.add_argument("--ollama", metavar="URL", help="Override OLLAMA_HOST, e.g. http://192.168.1.60:11434")
    ap.add_argument("--model", help="Override the Ollama model tag, e.g. qwen3-vl:8b")
    args = ap.parse_args()

    cfg = AppConfig()
    if args.backend:
        cfg.ai_judge_backend = args.backend
    if args.ollama:
        cfg.ai_judge_ollama_url = args.ollama
    if args.model:
        cfg.ai_judge_ollama_model = args.model
    cfg.ai_judge_enabled = True

    judge = AIJudge(cfg)
    print(judge.status_line())
    if not judge.available:
        return 1
    if not args.image:
        return 0

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"could not read image: {args.image}")
        return 1
    judge.min_interval = 0.0
    print(f"reviewing {args.image} ({frame.shape[1]}x{frame.shape[0]}) ...")
    r = judge.judge(frame, 0.70)
    if r.verdict is None:
        print(f"no verdict: {r.error}  ({r.latency_sec:.1f}s)")
        return 1
    v = r.verdict
    print(f"  usable   : {v.usable}")
    print(f"  score    : {v.score:.2f}   (blended with a 70% heuristic -> {r.blended_score:.2f})")
    print(f"  summary  : {v.summary}")
    if v.issues:
        print(f"  issues   : {'; '.join(v.issues)}")
    if v.strengths:
        print(f"  strengths: {'; '.join(v.strengths)}")
    print(f"  latency  : {r.latency_sec:.1f}s   tokens in/out: {r.input_tokens}/{r.output_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
