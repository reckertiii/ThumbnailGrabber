"""Waking the judge must actually load the model.

Resetting the window alone left Ollama idle until the first review, so the first
capture of a service paid for pulling 20GB off disk. These tests run against a
stub Ollama and assert the request Ollama actually needs to preload: an empty
prompt with the configured keep_alive.
"""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from config import AppConfig
from ai_judge import AIJudge, _keep_alive_value

MODEL = "qwen3-vl:30b"


class StubOllama(BaseHTTPRequestHandler):
    """Records what it was asked. `requests` is set per-server by the test."""

    def log_message(self, *a):          # keep the test output quiet
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._json(200, {"models": [{"name": MODEL}]})
        else:
            self._json(404, {"error": "no"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        self.server.requests.append((self.path, body))
        if self.server.fail_load:
            self._json(500, {"error": "out of memory"})
        else:
            self._json(200, {"model": MODEL, "done": True, "done_reason": "load"})


def _serve():
    srv = HTTPServer(("127.0.0.1", 0), StubOllama)
    srv.requests = []
    srv.fail_load = False
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class TestWakeLoadsTheModel(unittest.TestCase):
    def setUp(self):
        self.srv = _serve()
        host, port = self.srv.server_address
        cfg = AppConfig()
        cfg.ai_judge_backend = "ollama"
        cfg.ai_judge_ollama_url = f"http://{host}:{port}"
        cfg.ai_judge_ollama_model = MODEL
        self.cfg = cfg
        self.judge = AIJudge(cfg)

    def tearDown(self):
        self.srv.shutdown()

    def _wait(self, timeout=10):
        end = time.time() + timeout
        while self.judge._preloading and time.time() < end:
            time.sleep(0.05)
        self.assertFalse(self.judge._preloading, "preload never finished")

    def _loads(self):
        return [b for p, b in self.srv.requests if p == "/api/generate"]

    def test_wake_sends_a_preload(self):
        self.judge.wake()
        self._wait()
        loads = self._loads()
        self.assertEqual(len(loads), 1, "wake should load the model exactly once")
        body = loads[0]
        self.assertEqual(body["model"], MODEL)
        # An empty prompt is what makes Ollama load without generating.
        self.assertEqual(body["prompt"], "")
        self.assertEqual(body["keep_alive"],
                         _keep_alive_value(self.cfg.ai_judge_ollama_keep_alive))

    def test_wake_also_rearms_the_window(self):
        self.judge.started_at = time.time() - (self.judge.active_seconds + 60)
        self.judge._expiry_handled = True
        self.assertTrue(self.judge.expired)

        self.judge.wake()
        self._wait()

        self.assertFalse(self.judge.expired)
        self.assertFalse(self.judge._expiry_handled)

    def test_status_says_it_is_loading(self):
        self.judge._preloading = True
        self.assertIn("waking", self.judge.status_line().lower())
        self.judge._preloading = False
        self.assertNotIn("waking", self.judge.status_line().lower())

    def test_a_failed_load_is_reported_not_swallowed(self):
        self.srv.fail_load = True
        self.judge.wake()
        self._wait()
        self.assertTrue(self.judge._preload_error, "a failed load must be recorded")
        self.assertIn("500", self.judge._preload_error)
        self.assertIn("load failed", self.judge.status_line())

    def test_a_second_wake_does_not_stack_loads(self):
        self.judge._preloading = True        # pretend one is already in flight
        self.judge.preload_async()
        self.assertEqual(len(self._loads()), 0,
                         "a load already running must not be duplicated")
        self.judge._preloading = False

    def test_preload_is_a_no_op_for_the_anthropic_backend(self):
        self.judge.backend = "anthropic"
        self.judge.preload_async()
        self._wait()
        self.assertEqual(len(self._loads()), 0)


if __name__ == "__main__":
    unittest.main()


class TestSuspendResume(unittest.TestCase):
    """A source change must not leave the judge asleep.

    Stopping the processor releases the model, which is correct for a shutdown.
    A camera reconnect goes through the same stop, and the judge used to come
    back permanently expired from it -- and the log claimed a two-hour window had
    elapsed ninety seconds after startup.
    """

    def setUp(self):
        self.srv = _serve()
        host, port = self.srv.server_address
        cfg = AppConfig()
        cfg.ai_judge_backend = "ollama"
        cfg.ai_judge_ollama_url = f"http://{host}:{port}"
        cfg.ai_judge_ollama_model = MODEL
        cfg.ai_judge_active_hours = 2.0
        self.judge = AIJudge(cfg)

    def tearDown(self):
        self.srv.shutdown()

    def _settle(self, timeout=10):
        end = time.time() + timeout
        while self.judge._preloading and time.time() < end:
            time.sleep(0.05)

    def test_suspend_then_resume_keeps_the_remaining_window(self):
        # Burn half an hour of a two-hour window.
        self.judge.started_at = time.time() - 1800
        before = self.judge.seconds_remaining

        left = self.judge.suspend()
        self.assertTrue(self.judge.expired, "suspend should release the model")
        self.assertAlmostEqual(left, before, delta=5)

        self.judge.resume(left)
        self._settle()

        self.assertFalse(self.judge.expired, "resume must re-arm the judge")
        self.assertAlmostEqual(self.judge.seconds_remaining, before, delta=5)

    def test_resume_reloads_the_model(self):
        left = self.judge.suspend()
        self.srv.requests.clear()
        self.judge.resume(left)
        self._settle()
        loads = [b for p, b in self.srv.requests
                 if p == "/api/generate" and b.get("prompt") == ""]
        self.assertTrue(loads, "resume should pull the model back")

    def test_suspend_does_not_claim_the_window_elapsed(self):
        import io as _io, contextlib
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.judge.suspend()
        out = buf.getvalue()
        self.assertNotIn("elapsed", out,
                         "a restart must not be reported as the window elapsing")

    def test_a_genuinely_elapsed_window_still_says_so(self):
        import io as _io, contextlib
        self.judge.started_at = time.time() - (self.judge.active_seconds + 60)
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.judge._handle_expiry()
        self.assertIn("elapsed", buf.getvalue())
