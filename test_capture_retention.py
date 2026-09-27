"""Retention and review logic for live capture.

These run without loading any ML models: StreamProcessor is built with __new__ and
only the pieces under test are populated.
"""
import os
import random
import tempfile
import types
import unittest

from config import AppConfig
from analyzer import AnalysisResult
from stream_processor import StreamProcessor, CapturedScreenshot


def _bare_processor(cfg: AppConfig) -> StreamProcessor:
    proc = StreamProcessor.__new__(StreamProcessor)
    proc.config = cfg
    proc.saved_captures = []
    proc.captures_count = 0
    proc.on_capture_callback = None
    return proc


def _capture(cfg: AppConfig, score: float, t: float) -> CapturedScreenshot:
    path = os.path.join(cfg.output_dir, f"s{int(t * 1000)}.png")
    open(path, "w").close()
    return CapturedScreenshot(filepath=path, timestamp_str=str(t), overall_score=score,
                              analysis=AnalysisResult(), captured_at=t)


class TestRetention(unittest.TestCase):
    def setUp(self):
        self.cfg = AppConfig()
        self.cfg.output_dir = tempfile.mkdtemp()
        self.cfg.single_winner_mode = True
        self.cfg.max_retained_captures = 5
        self.cfg.min_capture_spacing_sec = 20.0
        self.proc = _bare_processor(self.cfg)

    def test_early_strong_capture_does_not_stop_later_captures(self):
        """Regression: a single strong early shot used to pin the bar for the session.

        The old check demanded each candidate beat the *last saved* score by 2% unless
        the frame looked visually different, so after one 0.90 nothing else saved.
        """
        random.seed(3)
        saves_after_spike = 0
        for t in range(0, 2400, 3):
            score = round(random.uniform(0.60, 0.82), 3)
            if t == 60:
                score = 0.90
            if self.proc._would_improve_retained(score, t):
                self.proc.saved_captures.append(_capture(self.cfg, score, t))
                self.proc.prune_to_best_n(self.cfg.max_retained_captures)
                if t > 60:
                    saves_after_spike += 1
        self.assertGreater(saves_after_spike, 10,
                           "capturing should continue for the whole session after a strong early shot")
        kept = sorted(self.proc.saved_captures, key=lambda c: c.overall_score, reverse=True)
        self.assertEqual(len(kept), 5)
        self.assertEqual(kept[0].overall_score, 0.90, "the strong shot itself must still be kept")

    def test_candidate_must_beat_weakest_kept_when_full(self):
        for i, s in enumerate([0.80, 0.78, 0.76, 0.74, 0.70]):
            self.proc.saved_captures.append(_capture(self.cfg, s, i * 100.0))
        self.assertFalse(self.proc._would_improve_retained(0.69, 1000.0))
        self.assertTrue(self.proc._would_improve_retained(0.71, 1000.0))

    def test_nearby_capture_only_displaced_by_better_one(self):
        for i, s in enumerate([0.80, 0.78, 0.76, 0.74, 0.70]):
            self.proc.saved_captures.append(_capture(self.cfg, s, i * 100.0))
        # 0.72 would beat the weakest kept (0.70), but it is 5s from the 0.80 shot,
        # so it is only worth saving if it beats *that* one.
        self.assertFalse(self.proc._would_improve_retained(0.72, 5.0))
        self.assertTrue(self.proc._would_improve_retained(0.81, 5.0))

    def test_prune_spreads_keepers_and_removes_files(self):
        # Six shots within 10s of each other plus two well separated. With a 20s
        # spacing rule the keepers should not all come from the cluster.
        times_scores = [(0, 0.90), (3, 0.89), (6, 0.88), (9, 0.87), (12, 0.86), (15, 0.85),
                        (300, 0.75), (600, 0.70)]
        for t, s in times_scores:
            self.proc.saved_captures.append(_capture(self.cfg, s, float(t)))
        self.proc.prune_to_best_n(3)
        kept_times = sorted(c.captured_at for c in self.proc.saved_captures)
        self.assertEqual(len(kept_times), 3)
        self.assertEqual(kept_times, [0.0, 300.0, 600.0])
        self.assertEqual(len(os.listdir(self.cfg.output_dir)), 3, "pruned shots must be deleted")

    def test_no_single_winner_keeps_everything(self):
        self.cfg.single_winner_mode = False
        for i in range(8):
            self.proc.saved_captures.append(_capture(self.cfg, 0.9 - i * 0.01, i * 1.0))
        self.assertTrue(self.proc._would_improve_retained(0.10, 100.0))


class TestJudgeBlending(unittest.TestCase):
    """The judge must refine, never block: any failure leaves the heuristic score."""

    def _judge(self, client):
        from ai_judge import AIJudge
        cfg = AppConfig()
        cfg.ai_judge_enabled = False   # construct quiet, then arm with the fake client
        j = AIJudge(cfg)
        j.enabled = True
        j.backend = "anthropic"
        j._client = client
        j.min_interval = 0.0
        return j

    def _fake(self, verdict=None, exc=None):
        class Usage:
            input_tokens = 1000
            output_tokens = 50

        class Resp:
            stop_reason = "end_turn"
            parsed_output = verdict
            usage = Usage()

        class Messages:
            def parse(self, **kw):
                if exc:
                    raise exc
                return Resp()

        return types.SimpleNamespace(messages=Messages())

    def test_blends_verdict_into_score(self):
        import numpy as np
        from ai_judge import JudgeVerdict
        j = self._judge(self._fake(JudgeVerdict(issues=["mouth slightly open (small)"],
                                                strengths=["eyes engaged"],
                                                score=0.9, usable=True, summary="fine")))
        j.weight = 0.5
        r = j.judge(np.zeros((64, 64, 3), dtype=np.uint8), 0.6)
        self.assertIsNotNone(r.verdict)
        self.assertAlmostEqual(r.blended_score, 0.75, places=3)

    def test_failure_keeps_heuristic(self):
        import numpy as np
        j = self._judge(self._fake(exc=RuntimeError("network down")))
        r = j.judge(np.zeros((64, 64, 3), dtype=np.uint8), 0.6)
        self.assertIsNone(r.verdict)
        self.assertEqual(r.blended_score, 0.6)
        self.assertTrue(j.available, "a transient error must not disable the judge")

    def test_off_without_credentials(self):
        import numpy as np
        from ai_judge import AIJudge
        saved = {k: os.environ.pop(k, None) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        try:
            cfg = AppConfig()
            cfg.ai_judge_enabled = True
            # Hermetic: a developer's .env may point OLLAMA_HOST at a real server.
            cfg.ai_judge_ollama_url = "http://127.0.0.1:1"
            j = AIJudge(cfg)
            if os.path.isdir(os.path.expanduser("~/.config/anthropic")):
                self.skipTest("an `ant auth login` profile is present on this machine")
            self.assertFalse(j.available)
            r = j.judge(np.zeros((64, 64, 3), dtype=np.uint8), 0.6)
            self.assertTrue(r.skipped)
            self.assertEqual(r.blended_score, 0.6)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
