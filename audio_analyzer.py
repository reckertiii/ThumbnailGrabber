"""Audio-Aware Pause/Laughter Gating & Local Whisper Sermon Title Generator (#8).

Two complementary features:
1. `LiveAudioMonitor`: Attaches a lightweight `ffmpeg` 16kHz mono PCM pipe to RTSP or
   file streams, tracking real-time speech RMS envelopes so `StreamProcessor` gives a
   subtle capture priority boost during natural 200-650ms inter-sentence pauses (when
   the speaker's mouth is relaxed/smiling rather than mid-syllable) and audience laughs.
   Also holds a rolling PCM ring buffer in RAM (~115 MB per hour at 16kHz 16-bit) so
   any recent window can be transcribed on demand.
2. `SermonNotesGenerator`: Transcribes either a recorded video/audio file or the live
   RAM audio buffer via `faster-whisper` (cached in `TG_DATA_DIR/whisper`), then asks
   the configured Ollama / Claude model for 5 punchy YouTube/slide thumbnail titles +
   3 verbatim pull-quotes + a short summary, saving `sermon_notes.md` alongside captures.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
import wave
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field


def _whisper_cache_dir() -> str:
    base = (
        os.environ.get("TG_DATA_DIR", "").strip()
        or os.environ.get("TG_MODELS_DIR", "").strip()
        or os.path.join(os.getcwd(), "models")
    )
    path = os.path.join(base, "whisper")
    os.makedirs(path, exist_ok=True)
    return path


class SermonNotesPayload(BaseModel):
    titles: List[str] = Field(description="5 punchy 2-6 word YouTube/slide sermon thumbnail titles.")
    quotes: List[str] = Field(description="Up to 3 memorable verbatim pull-quotes from the transcript.")
    summary: str = Field(description="Concise 2-sentence summary of the message theme.")


@dataclass
class AudioTelemetry:
    active: bool = False
    rms_db: float = -60.0
    is_speaking: bool = False
    in_speech_pause: bool = False
    is_laughter_burst: bool = False
    capture_bonus: float = 0.0
    buffered_minutes: float = 0.0


class LiveAudioMonitor:
    """Reads 16kHz mono PCM from RTSP/video via ffmpeg, detecting pauses & caching audio."""

    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 800  # 50ms per chunk (1600 bytes)

    def __init__(self, max_buffer_minutes: float = 90.0) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._source: str = ""

        # 50ms chunks: 20 chunks/sec -> 1200 chunks/min
        max_chunks = max(120, int(max_buffer_minutes * 60 * 20))
        self._pcm_chunks: deque = deque(maxlen=max_chunks)

        # Envelope state
        self._noise_floor_rms: float = 180.0
        self._last_speech_at: float = 0.0
        self._silence_started_at: float = 0.0
        self._rms_history: deque = deque(maxlen=30)
        self._latest = AudioTelemetry()

    def start(self, stream_source: str) -> None:
        self.stop()
        src = (stream_source or "").strip()
        # Skip bare webcam integer indices or empty/NDI sources
        if not src or src.isdigit() or src.lower().startswith("ndi://"):
            return
        if shutil.which("ffmpeg") is None:
            return
        self._source = src
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            self._proc = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            self._latest = AudioTelemetry()

    def _reader_loop(self) -> None:
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error"]
        if self._source.lower().startswith("rtsp://"):
            cmd += ["-rtsp_transport", "tcp"]
        cmd += [
            "-i", self._source,
            "-vn",
            "-ac", "1",
            "-ar", str(self.SAMPLE_RATE),
            "-f", "s16le",
            "pipe:1",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=self.CHUNK_SAMPLES * 4
            )
        except Exception:
            return

        byte_count = self.CHUNK_SAMPLES * 2
        try:
            while not self._stop_event.is_set() and self._proc and self._proc.stdout:
                raw = self._proc.stdout.read(byte_count)
                if not raw or len(raw) < byte_count:
                    break
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                self._process_chunk(raw, samples, time.time())
        except Exception:
            pass
        finally:
            if self._proc is not None:
                try:
                    self._proc.kill()
                except Exception:
                    pass
                self._proc = None
            with self._lock:
                self._latest.active = False

    def _process_chunk(self, raw_bytes: bytes, samples: np.ndarray, now: float) -> None:
        rms = float(np.sqrt(np.mean(samples * samples)) + 1e-3)
        rms_db = round(20.0 * float(np.log10(max(1.0, rms) / 32768.0)), 1)

        with self._lock:
            self._pcm_chunks.append(raw_bytes)
            self._rms_history.append(rms)
            # Adaptive noise floor tracks quiet 15th percentile slowly
            if len(self._rms_history) >= 10:
                p15 = float(np.percentile(list(self._rms_history), 15))
                self._noise_floor_rms = 0.95 * self._noise_floor_rms + 0.05 * max(60.0, p15)

            speech_thresh = max(320.0, self._noise_floor_rms * 2.4)
            is_speaking = rms >= speech_thresh
            if is_speaking:
                self._last_speech_at = now
                self._silence_started_at = 0.0
            else:
                if self._silence_started_at == 0.0:
                    self._silence_started_at = now

            pause_dur = (now - self._silence_started_at) if (not is_speaking and self._silence_started_at > 0) else 0.0
            since_speech = now - self._last_speech_at if self._last_speech_at > 0 else 999.0

            # Sweet spot: speaker was talking within last 2.2s and is now in a 0.15s..0.85s breath pause
            in_pause = (not is_speaking) and (0.15 <= pause_dur <= 0.85) and (since_speech <= 2.2)

            # Laughter / applause proxy: high modulation variance across last 0.8s with elevated energy
            is_laugh = False
            if len(self._rms_history) >= 16:
                recent = np.array(list(self._rms_history)[-16:], dtype=np.float32)
                mean_r = float(np.mean(recent))
                cv = float(np.std(recent)) / max(1.0, mean_r)
                zcr = float(np.mean(np.abs(np.diff(np.signbit(samples)))))
                if mean_r > speech_thresh * 1.35 and 0.35 <= cv <= 0.95 and zcr > 0.14:
                    is_laugh = True

            bonus = 0.035 if in_pause else (0.025 if is_laugh else 0.0)
            mins = (len(self._pcm_chunks) * 0.05) / 60.0
            self._latest = AudioTelemetry(
                active=True,
                rms_db=rms_db,
                is_speaking=is_speaking,
                in_speech_pause=in_pause,
                is_laughter_burst=is_laugh,
                capture_bonus=bonus,
                buffered_minutes=round(mins, 2),
            )

    def telemetry(self) -> AudioTelemetry:
        with self._lock:
            return AudioTelemetry(**self._latest.__dict__)

    def export_wav(self, dest_wav: str, last_minutes: float = 60.0) -> bool:
        """Writes the buffered mono 16kHz PCM audio to a WAV file for Whisper."""
        with self._lock:
            if not self._pcm_chunks:
                return False
            max_c = max(1, int(last_minutes * 60 * 20))
            chunks = list(self._pcm_chunks)[-max_c:]
        with wave.open(dest_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(b"".join(chunks))
        return True


class SermonNotesGenerator:
    """Lazy-loaded `faster-whisper` transcriber + Ollama/Claude title & pull-quote generator."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._whisper_model: Any = None
        self._loaded_size: str = ""
        self.last_notes: Optional[Dict[str, Any]] = None

    @staticmethod
    def whisper_available() -> bool:
        try:
            import faster_whisper  # type: ignore  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_whisper(self, model_size: str = "base.en") -> Any:
        with self._lock:
            if self._whisper_model is not None and self._loaded_size == model_size:
                return self._whisper_model
            from faster_whisper import WhisperModel  # type: ignore
            self._whisper_model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=max(2, min(12, (os.cpu_count() or 4) // 2)),
                download_root=_whisper_cache_dir(),
            )
            self._loaded_size = model_size
            return self._whisper_model

    def transcribe_media(self, media_path: str, model_size: str = "base.en", max_minutes: float = 60.0) -> str:
        """Transcribes `media_path` (video or WAV) into plain text using faster-whisper."""
        if not os.path.isfile(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")
        model = self._get_whisper(model_size=model_size)
        segments, _info = model.transcribe(media_path, beam_size=2, vad_filter=True)
        max_sec = max(60.0, float(max_minutes) * 60.0)
        pieces: List[str] = []
        for seg in segments:
            if getattr(seg, "start", 0.0) > max_sec:
                break
            txt = (getattr(seg, "text", "") or "").strip()
            if txt:
                pieces.append(txt)
        return " ".join(pieces).strip()

    @staticmethod
    def summarize_transcript_into_titles(
        transcript: str,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen3-vl:4b",
        num_thread: int = 0,
        timeout_sec: float = 180.0,
    ) -> Dict[str, Any]:
        """Generates 5 punchy sermon titles + 3 pull-quotes via Ollama (with heuristic fallback)."""
        clean = (transcript or "").strip()
        if not clean:
            return {
                "titles": [],
                "quotes": [],
                "summary": "No speech transcript available.",
                "transcript_chars": 0,
                "source": "empty",
            }

        excerpt = clean[:14000]
        prompt = (
            "Below is the transcript of a live message/sermon. Generate:\n"
            "1. `titles`: 5 short, compelling, high-impact YouTube / presentation slide titles (2 to 6 words each, Title Case, no clickbait gimmicks).\n"
            "2. `quotes`: Up to 3 memorable, punchy verbatim 1-sentence quotes from the speaker.\n"
            "3. `summary`: A concise 2-sentence overview of the message.\n\n"
            f"TRANSCRIPT:\n{excerpt}"
        )

        try:
            opts: Dict[str, Any] = {"temperature": 0.3, "num_predict": 500}
            if num_thread > 0:
                opts["num_thread"] = int(num_thread)
            body = {
                "model": ollama_model,
                "stream": False,
                "think": False,
                "format": SermonNotesPayload.model_json_schema(),
                "options": opts,
                "messages": [
                    {"role": "system", "content": "You are an expert church media producer crafting concise sermon slide titles and pull-quotes."},
                    {"role": "user", "content": prompt},
                ],
            }
            req = urllib.request.Request(
                f"{ollama_url.rstrip('/')}/api/chat",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            msg = data.get("message") or {}
            txt = (msg.get("content") or msg.get("thinking") or "").strip()
            lo, hi = txt.find("{"), txt.rfind("}")
            if lo >= 0 and hi > lo:
                txt = txt[lo:hi + 1]
            parsed = SermonNotesPayload.model_validate_json(txt)
            return {
                "titles": parsed.titles[:5],
                "quotes": parsed.quotes[:3],
                "summary": parsed.summary,
                "transcript_chars": len(clean),
                "source": f"ollama:{ollama_model}",
            }
        except Exception as e:
            # Extractive offline fallback so title generation still returns useful output without Ollama
            sentences = [s.strip() for s in clean.replace("?", ".").replace("!", ".").split(".") if 25 <= len(s.strip()) <= 140]
            quotes = sentences[:3]
            words = [w.strip(",.;:\"'()") for w in clean.split() if len(w.strip(",.;:\"'()")) >= 5]
            freq: Dict[str, int] = {}
            stop = {"about", "there", "their", "would", "could", "should", "going", "think", "because", "really", "people", "every", "thing", "where", "which", "these", "those"}
            for w in words:
                lw = w.lower()
                if lw not in stop and lw.isalpha():
                    freq[lw] = freq.get(lw, 0) + 1
            top_terms = [k.title() for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:5]]
            titles = [f"Walking In {t}" for t in top_terms[:3]] or ["Message Highlights"]
            return {
                "titles": titles,
                "quotes": quotes,
                "summary": (sentences[0] + ".") if sentences else clean[:200],
                "transcript_chars": len(clean),
                "source": f"extractive_fallback ({type(e).__name__})",
            }

    def generate_and_save(
        self,
        output_dir: str,
        media_path: Optional[str] = None,
        audio_monitor: Optional[LiveAudioMonitor] = None,
        raw_transcript: str = "",
        whisper_model: str = "base.en",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen3-vl:4b",
        num_thread: int = 0,
    ) -> Dict[str, Any]:
        """Transcribes audio (if needed), generates titles/quotes, and writes `sermon_notes.md`."""
        transcript = (raw_transcript or "").strip()
        if not transcript:
            if media_path and os.path.isfile(media_path):
                transcript = self.transcribe_media(media_path, model_size=whisper_model)
            elif audio_monitor is not None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tmp_wav = tf.name
                try:
                    if audio_monitor.export_wav(tmp_wav):
                        transcript = self.transcribe_media(tmp_wav, model_size=whisper_model)
                finally:
                    try:
                        os.unlink(tmp_wav)
                    except OSError:
                        pass

        result = self.summarize_transcript_into_titles(
            transcript=transcript,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            num_thread=num_thread,
        )
        result["transcript_preview"] = transcript[:1200]
        result["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        if output_dir and result.get("titles"):
            os.makedirs(output_dir, exist_ok=True)
            notes_path = os.path.join(output_dir, "sermon_notes.md")
            md_lines = [
                "# Sermon Thumbnail Titles & Notes",
                f"*Generated {result['generated_at']} ({result.get('source', '')})*",
                "",
                "## Suggested Thumbnail / Slide Titles",
            ]
            for i, t in enumerate(result.get("titles") or [], 1):
                md_lines.append(f"{i}. **{t}**")
            if result.get("quotes"):
                md_lines += ["", "## Key Pull-Quotes"]
                for q in result["quotes"]:
                    md_lines.append(f"> \"{q}\"")
            if result.get("summary"):
                md_lines += ["", "## Summary", result["summary"]]
            if transcript:
                md_lines += ["", "## Transcript Excerpt", transcript[:4000]]
            try:
                with open(notes_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(md_lines) + "\n")
                result["notes_file"] = notes_path
            except OSError:
                pass

        self.last_notes = result
        return result
