"""Remembering which camera to connect to, across restarts.

The stream source used to live only in the environment and in memory. Setting it
from the UI worked until the process restarted, and then the box showed
TG_STREAM_SOURCE again -- which on the NAS reads as "it forgot my camera". So the
settings are written to the data directory, the same mounted dataset the ratings
live on, and loaded at startup in preference to the environment.

The password is part of that file, so the file is the only place it is kept and it
is created 0600. Callers get the password only by asking for it explicitly; the
shape meant for the UI leaves it out entirely.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from camera_probe import DEFAULT_PORT, build_url

FILENAME = "camera.json"


def _data_dir() -> str:
    return (os.environ.get("TG_DATA_DIR", "").strip()
            or os.path.join(os.getcwd(), "models"))


def settings_path() -> str:
    return os.path.join(_data_dir(), FILENAME)


@dataclass
class CameraSettings:
    kind: str = "rtsp"          # rtsp | webcam | file
    host: str = ""
    port: int = DEFAULT_PORT
    path: str = ""
    username: str = ""
    password: str = ""
    webcam_index: int = 0
    file_path: str = ""

    # --- what the UI is allowed to see ------------------------------------
    def public(self) -> Dict[str, Any]:
        """Everything except the password, plus a flag for whether one is set.

        The UI needs to show that a password exists without being given it, so a
        saved camera can be reconnected without retyping, and so the value cannot
        be read back out of an endpoint that has no authentication in front of it.
        """
        d = asdict(self)
        d.pop("password", None)
        d["has_password"] = bool(self.password)
        d["url"] = self.url(redacted=True)
        return d

    def url(self, redacted: bool = False) -> str:
        """The source string the stream processor takes."""
        if self.kind == "webcam":
            return str(int(self.webcam_index))
        if self.kind == "file":
            return self.file_path
        if not self.host:
            return ""
        if redacted:
            from config import redact_credentials
            return redact_credentials(
                build_url(self.host, self.port, self.path or "/", self.username, self.password))
        return build_url(self.host, self.port, self.path or "/", self.username, self.password)

    def is_configured(self) -> bool:
        if self.kind == "webcam":
            return True
        if self.kind == "file":
            return bool(self.file_path)
        return bool(self.host and self.path)


def load() -> Optional[CameraSettings]:
    """Saved settings, or None if nothing has been saved yet."""
    p = settings_path()
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[CameraStore] Ignoring unreadable {FILENAME}: {e}")
        return None

    known = {f for f in CameraSettings.__dataclass_fields__}
    data = {k: v for k, v in raw.items() if k in known}
    try:
        s = CameraSettings(**data)
        s.port = int(s.port or DEFAULT_PORT)
        s.webcam_index = int(s.webcam_index or 0)
        return s
    except (TypeError, ValueError) as e:
        print(f"[CameraStore] Ignoring malformed {FILENAME}: {e}")
        return None


def save(settings: CameraSettings) -> str:
    """Writes the settings 0600, replacing any previous file atomically."""
    p = settings_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"

    # Create with the restrictive mode from the outset rather than widening then
    # narrowing, which would leave the password briefly world-readable.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(asdict(settings), f, indent=2)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, p)

    # os.open's mode is subject to umask, and os.replace does not carry one, so
    # set it explicitly. Harmless no-op on Windows, which ignores POSIX modes.
    try:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return p


def clear() -> bool:
    try:
        os.unlink(settings_path())
        return True
    except OSError:
        return False
