"""Camera credentials must not escape the process in readable form.

The web UI has no authentication, so anything an endpoint returns is readable by
anyone who can reach the port, and anything printed is readable in the host's app
logs. These tests pin that down, because the leak they cover was real: /api/state
served the RTSP password in cleartext.
"""

import unittest

from config import AppConfig, redact_credentials, is_redacted

SECRET = "hunter2-do-not-leak"
URL = f"rtsp://camop:{SECRET}@192.0.2.10:554/media/video1"


class TestRedaction(unittest.TestCase):
    def test_password_and_username_both_removed(self):
        out = redact_credentials(URL)
        self.assertNotIn(SECRET, out)
        self.assertNotIn("camop", out)
        self.assertIn("192.0.2.10:554/media/video1", out)

    def test_leaves_credential_free_sources_alone(self):
        for src in ("0", "1", "/mnt/video/service.mp4",
                    "rtsp://192.0.2.10:554/media/video1",
                    r"C:\Users\x\Videos\a.mp4", ""):
            self.assertEqual(redact_credentials(src), src)

    def test_a_redacted_url_is_recognisable(self):
        # The UI relies on this to keep "***" out of the source box, which would
        # otherwise be submitted as the real password.
        self.assertTrue(is_redacted(redact_credentials(URL)))
        self.assertFalse(is_redacted("rtsp://192.0.2.10/media/video1"))
        self.assertFalse(is_redacted("0"))

    def test_config_to_dict_carries_no_password(self):
        cfg = AppConfig()
        cfg.stream_source = URL
        self.assertNotIn(SECRET, repr(cfg.to_dict()))


class TestStatePayload(unittest.TestCase):
    def test_state_endpoint_does_not_expose_the_password(self):
        import web.server as ws

        ws.state.config.stream_source = URL
        body = repr(ws.get_state())   # the real /api/state handler
        self.assertNotIn(SECRET, body)
        self.assertNotIn("camop", body)

    def test_set_source_echo_does_not_reflect_the_password(self):
        import web.server as ws

        # Avoid starting a real capture thread for a string-handling check.
        original = ws.state.start_processor
        ws.state.start_processor = lambda: None
        try:
            echo = ws.set_source({"source": URL})
        finally:
            ws.state.start_processor = original
        self.assertNotIn(SECRET, repr(echo))
        self.assertTrue(is_redacted(echo["source"]))


if __name__ == "__main__":
    unittest.main()
