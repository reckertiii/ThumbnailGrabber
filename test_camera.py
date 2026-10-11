"""Camera connection: diagnosis, and remembering the answer.

The probe exists because OpenCV reports every RTSP failure identically, so these
tests check it tells the four real causes apart -- bad credentials, bad path,
closed port, unresolvable host. The fake camera validates the Digest response
properly, because "wrong password" is the failure that actually happens and its
message is the one that has to be right.
"""

import hashlib
import os
import socket
import tempfile
import threading
import time
import unittest

import camera_probe

REALM = "cam"
NONCE = "abc123"
PASSWORD = "rightpass"
USER = "admin"
GOOD_PATH = "/media/video1"


def _expected_response(method, uri, user=USER, password=PASSWORD):
    h = lambda s: hashlib.md5(s.encode()).hexdigest()
    return h(f"{h(f'{user}:{REALM}:{password}')}:{NONCE}:{h(f'{method}:{uri}')}")


class FakeCamera(threading.Thread):
    """A minimal RTSP responder: 401 with a Digest challenge, then 200 only for
    a correct digest on the correct path."""

    daemon = True

    def __init__(self):
        super().__init__()
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self._stop = False

    def stop(self):
        self._stop = True
        try:
            self.sock.close()
        except OSError:
            pass

    def run(self):
        while not self._stop:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        with conn:
            conn.settimeout(3)
            try:
                for _ in range(3):
                    data = conn.recv(8192).decode("utf-8", "replace")
                    if not data:
                        return
                    request_line = data.splitlines()[0]
                    parts = request_line.split()
                    method, uri = (parts[0], parts[1]) if len(parts) >= 2 else ("", "")

                    if "Authorization:" not in data:
                        conn.sendall(
                            b"RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\n"
                            b'WWW-Authenticate: Digest realm="' + REALM.encode() +
                            b'", nonce="' + NONCE.encode() + b'"\r\n'
                            b"Server: FakeCam/1.0\r\n\r\n")
                        continue

                    got = ""
                    for token in data.split():
                        if token.startswith("response="):
                            got = token.split("=", 1)[1].strip('",')
                    if got != _expected_response(method, uri):
                        conn.sendall(b"RTSP/1.0 401 Unauthorized\r\nCSeq: 2\r\n"
                                     b"Server: FakeCam/1.0\r\n\r\n")
                        return

                    if GOOD_PATH not in uri:
                        conn.sendall(b"RTSP/1.0 404 Not Found\r\nCSeq: 2\r\n\r\n")
                        continue

                    sdp = (b"v=0\r\nm=video 0 RTP/AVP 96\r\n"
                           b"a=rtpmap:96 H264/90000\r\n"
                           b"a=framesize:96 1920-1080\r\n")
                    conn.sendall(b"RTSP/1.0 200 OK\r\nCSeq: 2\r\nServer: FakeCam/1.0\r\n"
                                 b"Content-Type: application/sdp\r\nContent-Length: " +
                                 str(len(sdp)).encode() + b"\r\n\r\n" + sdp)
                    return
            except (socket.timeout, OSError):
                return


class TestProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cam = FakeCamera()
        cls.cam.start()
        time.sleep(0.2)
        cls.host = "127.0.0.1"
        cls.port = cls.cam.port

    @classmethod
    def tearDownClass(cls):
        cls.cam.stop()

    def test_correct_credentials_connect(self):
        r = camera_probe.probe(self.host, self.port, GOOD_PATH, USER, PASSWORD, timeout=3)
        self.assertTrue(r.ok, r.message)
        self.assertEqual(r.rtsp_status, 200)
        self.assertEqual(r.working_path, GOOD_PATH)
        self.assertIn("H264", r.video)

    def test_wrong_password_is_named_as_such(self):
        r = camera_probe.probe(self.host, self.port, GOOD_PATH, USER, "wrongpass", timeout=3)
        self.assertFalse(r.ok)
        self.assertEqual(r.code, "unauthorized")
        self.assertEqual(r.rtsp_status, 401)
        self.assertIn("rejected", r.message.lower())
        # Must not be mistaken for a path problem, or the operator edits the
        # wrong field -- which is the whole reason this probe exists.
        self.assertNotIn("path", r.message.lower())

    def test_missing_credentials_asks_for_them(self):
        r = camera_probe.probe(self.host, self.port, GOOD_PATH, "", "", timeout=3)
        self.assertFalse(r.ok)
        self.assertEqual(r.code, "unauthorized")
        self.assertEqual(r.auth_scheme, "digest")
        self.assertIn("needs a username", r.message)

    def test_wrong_path_is_discovered(self):
        r = camera_probe.probe(self.host, self.port, "/nope", USER, PASSWORD, timeout=3)
        self.assertTrue(r.ok, r.message)
        self.assertEqual(r.working_path, GOOD_PATH)
        self.assertGreater(len(r.tried_paths), 1)

    def test_wrong_path_without_discovery_stays_failed(self):
        r = camera_probe.probe(self.host, self.port, "/nope", USER, PASSWORD,
                               timeout=3, discover=False)
        self.assertFalse(r.ok)
        self.assertEqual(r.code, "no_path")
        self.assertEqual(r.tried_paths, ["/nope"])

    def test_closed_port_blames_the_network(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
        s.close()
        r = camera_probe.probe("127.0.0.1", dead, GOOD_PATH, USER, PASSWORD, timeout=3)
        self.assertFalse(r.ok)
        self.assertEqual(r.stage, "tcp")
        self.assertIn(r.code, ("refused", "timeout", "unreachable"))

    def test_unresolvable_host(self):
        r = camera_probe.probe("no-such-camera.invalid", 554, GOOD_PATH, timeout=3)
        self.assertFalse(r.ok)
        self.assertEqual(r.stage, "dns")
        self.assertEqual(r.code, "unresolved")

    def test_empty_host_is_rejected_without_a_lookup(self):
        r = camera_probe.probe("", 554)
        self.assertFalse(r.ok)
        self.assertEqual(r.code, "no_host")


class TestUrlBuilding(unittest.TestCase):
    def test_password_special_characters_are_encoded(self):
        # An unencoded @ or / in a password silently produces a URL pointing
        # somewhere else entirely.
        url = camera_probe.build_url("10.0.0.5", 554, "/media/video1",
                                     "admin", "p@ss/word?x")
        self.assertIn("p%40ss%2Fword%3Fx", url)
        self.assertTrue(url.startswith("rtsp://admin:"))
        self.assertIn("@10.0.0.5/media/video1", url)

    def test_default_port_is_omitted(self):
        self.assertNotIn(":554", camera_probe.build_url("10.0.0.5", 554, "/x"))
        self.assertIn(":8554", camera_probe.build_url("10.0.0.5", 8554, "/x"))

    def test_no_credentials_means_no_at_sign(self):
        self.assertEqual(camera_probe.build_url("10.0.0.5", 554, "/x"),
                         "rtsp://10.0.0.5/x")


class TestStore(unittest.TestCase):
    def setUp(self):
        self._prev = os.environ.get("TG_DATA_DIR")
        self.tmp = tempfile.mkdtemp()
        os.environ["TG_DATA_DIR"] = self.tmp

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("TG_DATA_DIR", None)
        else:
            os.environ["TG_DATA_DIR"] = self._prev

    def _settings(self):
        import camera_store
        return camera_store.CameraSettings(
            kind="rtsp", host="10.0.0.5", port=554, path=GOOD_PATH,
            username=USER, password=PASSWORD)

    def test_round_trip(self):
        import camera_store
        camera_store.save(self._settings())
        back = camera_store.load()
        self.assertIsNotNone(back)
        self.assertEqual(back.host, "10.0.0.5")
        self.assertEqual(back.password, PASSWORD)
        self.assertEqual(back.url(), camera_probe.build_url(
            "10.0.0.5", 554, GOOD_PATH, USER, PASSWORD))

    def test_public_view_withholds_the_password(self):
        import camera_store
        pub = self._settings().public()
        self.assertNotIn("password", pub)
        self.assertTrue(pub["has_password"])
        self.assertNotIn(PASSWORD, repr(pub))
        # The displayed URL is masked too.
        self.assertIn("***", pub["url"])
        self.assertNotIn(PASSWORD, pub["url"])

    def test_file_is_not_world_readable(self):
        import camera_store
        p = camera_store.save(self._settings())
        if os.name != "nt":   # POSIX modes are meaningless on Windows
            mode = os.stat(p).st_mode & 0o777
            self.assertEqual(mode, 0o600, f"camera.json is {oct(mode)}")

    def test_missing_file_is_not_an_error(self):
        import camera_store
        self.assertIsNone(camera_store.load())

    def test_corrupt_file_is_ignored_rather_than_fatal(self):
        import camera_store
        with open(camera_store.settings_path(), "w", encoding="utf-8") as f:
            f.write("{not json at all")
        self.assertIsNone(camera_store.load())

    def test_unknown_keys_are_dropped(self):
        # A settings file written by a newer version must not crash an older one.
        import camera_store, json
        with open(camera_store.settings_path(), "w", encoding="utf-8") as f:
            json.dump({"kind": "rtsp", "host": "10.0.0.9", "path": "/x",
                       "something_new": True}, f)
        back = camera_store.load()
        self.assertIsNotNone(back)
        self.assertEqual(back.host, "10.0.0.9")

    def test_webcam_and_file_sources(self):
        import camera_store
        self.assertEqual(
            camera_store.CameraSettings(kind="webcam", webcam_index=1).url(), "1")
        self.assertEqual(
            camera_store.CameraSettings(kind="file", file_path="/v/a.mp4").url(),
            "/v/a.mp4")
        self.assertFalse(camera_store.CameraSettings(kind="file").is_configured())


if __name__ == "__main__":
    unittest.main()


class TestSourceSwitching(unittest.TestCase):
    """Changing the source must leave the stream running.

    The failure this pins down looked like a working connection: the camera was
    saved, the UI said connected, and the feed stayed empty forever, because a
    source that failed to open had already cleared `running`, and the restart was
    conditional on it.

    The dead state is set directly rather than produced by a real bad source --
    FFmpeg takes tens of seconds to give up on an unroutable address, and the
    invariant under test is "restarts when not running", not how it stopped.
    """

    def setUp(self):
        self._prev = os.environ.get("TG_DATA_DIR")
        os.environ["TG_DATA_DIR"] = tempfile.mkdtemp()

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("TG_DATA_DIR", None)
        else:
            os.environ["TG_DATA_DIR"] = self._prev

    def _processor(self, source):
        from config import AppConfig
        from stream_processor import StreamProcessor
        cfg = AppConfig()
        cfg.stream_source = source
        # Keep the test off the network: constructing a judge would otherwise
        # reach the configured Ollama host.
        cfg.ai_judge_backend = "off"
        cfg.dvr_buffer_seconds = 0.0
        return StreamProcessor(config=cfg)

    def test_update_source_restarts_a_processor_that_died(self):
        proc = self._processor("0")
        try:
            # Exactly the state _grabber_loop leaves behind when a source will
            # not open.
            proc.running = False
            proc._stop_event.set()
            proc.connection_status = proc.STATUS_FAILED

            proc.update_source("0")

            self.assertTrue(proc.running,
                            "a source change must restart a processor that had died")
            self.assertFalse(proc._stop_event.is_set(),
                             "the stop event must be cleared or the threads exit at once")
            self.assertNotEqual(proc.connection_status, proc.STATUS_DISCONNECTED)
        finally:
            proc.stop()

    def test_update_source_keeps_a_live_processor_running(self):
        proc = self._processor("0")
        try:
            proc.start()
            proc.update_source("0")
            self.assertTrue(proc.running)
        finally:
            proc.stop()

    def test_update_source_applies_the_new_source(self):
        proc = self._processor("0")
        try:
            proc.running = False
            proc.update_source("rtsp://10.0.0.9/media/video1")
            self.assertEqual(proc.config.stream_source, "rtsp://10.0.0.9/media/video1")
            self.assertEqual(proc.reconnect_attempt, 0)
        finally:
            proc.stop()
