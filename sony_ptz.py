"""Sony SRG-A40 / SRG-A12 PTZ & AI Auto-Framing Controller (#10).

Talks directly to Sony PTZ cameras over HTTP CGI (using the same host + username +
password already stored in `camera_store.py`), supporting both HTTP Digest Auth
(RFC 2617 MD5 / SHA-256, standard on Sony SRG firmware) and Basic Auth:
  - Pan / Tilt directional move, timed nudge, & stop (`/command/ptzf.cgi`)
  - Optical Zoom tele / wide / stop (`/command/ptzf.cgi`)
  - Home position (`/command/ptzf.cgi?Relative=home` / `AbsolutePanTilt`)
  - Preset 1..16 Recall & Save (`/command/presetposition.cgi`)
  - Sony Built-in AI PTZ Auto-Framing On / Off (`/command/ptzautoframing.cgi`
    and `/command/analytics.cgi` fallback across SRG-A40 firmware revisions)
  - Live camera status inquiry (`/command/inquiry.cgi?inq=ptzautoframing`)
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

DIRECTION_VECTORS = {
    "up": ("00", "01", "up"),
    "down": ("00", "FF", "down"),
    "left": ("FF", "00", "left"),
    "right": ("01", "00", "right"),
    "upleft": ("FF", "01", "leftup"),
    "upright": ("01", "01", "rightup"),
    "downleft": ("FF", "FF", "leftdown"),
    "downright": ("01", "FF", "rightdown"),
}


class SonyPTZController:
    """HTTP CGI client for Sony SRG-series PTZ cameras."""

    def __init__(
        self,
        host: str = "",
        username: str = "",
        password: str = "",
        http_port: int = 80,
        timeout_sec: float = 4.0,
    ) -> None:
        self.host = (host or "").strip()
        self.username = username or ""
        self.password = password or ""
        self.http_port = int(http_port or 80)
        self.timeout_sec = float(timeout_sec)
        self._nudge_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    @classmethod
    def from_camera_settings(cls, cam: Any) -> "SonyPTZController":
        if cam is None or getattr(cam, "kind", "") != "rtsp" or not getattr(cam, "host", ""):
            return cls()
        return cls(
            host=cam.host,
            username=getattr(cam, "username", "") or "",
            password=getattr(cam, "password", "") or "",
            http_port=int(getattr(cam, "http_port", 80) or 80),
        )

    @property
    def configured(self) -> bool:
        return bool(self.host)

    @property
    def base_url(self) -> str:
        if ":" in self.host and not self.host.startswith("["):
            host_part = self.host
        else:
            host_part = f"{self.host}:{self.http_port}" if self.http_port != 80 else self.host
        return f"http://{host_part}"

    def _build_opener(self) -> urllib.request.OpenerDirector:
        if self.username or self.password:
            mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            mgr.add_password(None, self.base_url, self.username, self.password)
            return urllib.request.build_opener(
                urllib.request.HTTPDigestAuthHandler(mgr),
                urllib.request.HTTPBasicAuthHandler(mgr),
            )
        return urllib.request.build_opener()

    def _request_cgi(self, path_and_query: str) -> Dict[str, Any]:
        if not self.configured:
            return {"ok": False, "error": "No RTSP camera IP is configured."}
        url = f"{self.base_url}{path_and_query}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ThumbnailGrabber/2.0",
                "Referer": f"{self.base_url}/",
            },
            method="GET",
        )
        try:
            opener = self._build_opener()
            with opener.open(req, timeout=self.timeout_sec) as resp:
                body = resp.read().decode("utf-8", "replace").strip()
                return {"ok": True, "status": resp.status, "body": body}
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace").strip()[:160]
            except Exception:
                pass
            return {"ok": False, "status": e.code, "error": f"HTTP {e.code}: {detail or e.reason}"}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def _cancel_pending_stop(self) -> None:
        with self._lock:
            if self._nudge_timer is not None:
                self._nudge_timer.cancel()
                self._nudge_timer = None

    def _schedule_stop(self, kind: str, duration_ms: int) -> None:
        delay = max(0.08, min(2.0, duration_ms / 1000.0))

        def _do_stop() -> None:
            if kind == "zoom":
                self.zoom_stop()
            else:
                self.stop()

        with self._lock:
            if self._nudge_timer is not None:
                self._nudge_timer.cancel()
            t = threading.Timer(delay, _do_stop)
            t.daemon = True
            t.start()
            self._nudge_timer = t

    def move(self, direction: str, speed: int = 8, duration_ms: Optional[int] = 260) -> Dict[str, Any]:
        """Starts directional pan/tilt (`up`, `down`, `left`, `right`, diagonals, or `stop`)."""
        d = (direction or "").strip().lower()
        if d == "stop":
            return self.stop()
        if d not in DIRECTION_VECTORS:
            return {"ok": False, "error": f"Unsupported PTZ direction '{direction}'"}
        self._cancel_pending_stop()
        spd = max(1, min(24, int(speed)))
        _, _, sony_dir = DIRECTION_VECTORS[d]
        res = self._request_cgi(f"/command/ptzf.cgi?Move={sony_dir},{spd}")
        if res.get("ok") and duration_ms and duration_ms > 0:
            self._schedule_stop("pt", int(duration_ms))
        return res

    def stop(self) -> Dict[str, Any]:
        self._cancel_pending_stop()
        return self._request_cgi("/command/ptzf.cgi?Move=stop,motor")

    def zoom(self, direction: str, speed: int = 4, duration_ms: Optional[int] = 240) -> Dict[str, Any]:
        """Zooms `tele` (in), `wide` (out), or `stop`."""
        d = (direction or "").strip().lower()
        if d in ("in", "tele", "zoom_in"):
            sony_z = "tele"
        elif d in ("out", "wide", "zoom_out"):
            sony_z = "wide"
        elif d == "stop":
            return self.zoom_stop()
        else:
            return {"ok": False, "error": f"Unsupported zoom direction '{direction}'"}
        self._cancel_pending_stop()
        spd = max(1, min(8, int(speed)))
        res = self._request_cgi(f"/command/ptzf.cgi?Move={sony_z},{spd}")
        if res.get("ok") and duration_ms and duration_ms > 0:
            self._schedule_stop("zoom", int(duration_ms))
        return res

    def zoom_stop(self) -> Dict[str, Any]:
        self._cancel_pending_stop()
        return self._request_cgi("/command/ptzf.cgi?Move=stop,zoom")

    def home(self) -> Dict[str, Any]:
        self._cancel_pending_stop()
        res = self._request_cgi("/command/presetposition.cgi?HomePos=recall")
        if not res.get("ok"):
            res = self._request_cgi("/command/ptzf.cgi?AbsolutePanTilt=0000,0000,12")
        return res

    def recall_preset(self, preset_id: int) -> Dict[str, Any]:
        pid = max(1, min(64, int(preset_id)))
        return self._request_cgi(f"/command/presetposition.cgi?PresetCall={pid},18")

    def save_preset(self, preset_id: int, name: str = "") -> Dict[str, Any]:
        pid = max(1, min(64, int(preset_id)))
        label = urllib.parse.quote((name or f"Preset{pid}")[:16])
        return self._request_cgi(f"/command/presetposition.cgi?PresetSet={pid},{label},on")

    def set_auto_framing(self, enabled: bool) -> Dict[str, Any]:
        """Toggles Sony SRG-A40 / SRG-A12 built-in AI PTZ Auto-Framing."""
        val = "on" if enabled else "off"
        # SRG-A40 primary endpoint
        res = self._request_cgi(f"/command/ptzautoframing.cgi?PtzAutoFraming={val}")
        if not res.get("ok"):
            # Alternate firmware CGI key
            res = self._request_cgi(f"/command/analytics.cgi?PtzAutoFraming={val}")
        res["auto_framing"] = bool(enabled)
        return res

    def status(self) -> Dict[str, Any]:
        if not self.configured:
            return {"configured": False, "reachable": False, "auto_framing": None}
        inq = self._request_cgi("/command/inquiry.cgi?inq=ptzautoframing")
        auto_framing: Optional[bool] = None
        if inq.get("ok"):
            body = (inq.get("body") or "").lower()
            if "ptzautoframing=on" in body or "autoframing=on" in body:
                auto_framing = True
            elif "ptzautoframing=off" in body or "autoframing=off" in body:
                auto_framing = False
        else:
            inq_ptz = self._request_cgi("/command/inquiry.cgi?inq=ptzf")
            if inq_ptz.get("ok"):
                inq = inq_ptz
        return {
            "configured": True,
            "host": self.host,
            "reachable": bool(inq.get("ok")),
            "auto_framing": auto_framing,
            "error": inq.get("error", "") if not inq.get("ok") else "",
            "saved_at": round(time.time(), 1),
        }
