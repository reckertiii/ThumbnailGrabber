"""Finding out why a camera will not connect.

OpenCV gives one answer for every RTSP failure: `isOpened()` is false. Wrong
password, wrong stream path, closed port and wrong VLAN all look identical, which
leaves the operator changing settings at random.

RTSP is a text protocol close enough to HTTP to speak by hand, and it does
distinguish these cases -- 401 for credentials, 404 for a bad path, a TCP error
for anything below that. This module asks the camera directly and reports which
one it is, then offers to find the stream path by trying the usual ones.

Standard library only, on purpose: this has to run inside the container image
without adding to the dependency surface.
"""

from __future__ import annotations

import base64
import hashlib
import re
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DEFAULT_PORT = 554
DEFAULT_TIMEOUT = 6.0

# Tried in order when discovering a path. Sony SRG first -- that is the camera
# this was written for -- then the other makes that turn up in church racks.
COMMON_PATHS: List[str] = [
    "/media/video1",                        # Sony SRG series, main stream
    "/media/video2",                        # Sony, second stream
    "/video1",
    "/Streaming/Channels/101",              # Hikvision
    "/cam/realmonitor?channel=1&subtype=0",  # Dahua
    "/axis-media/media.amp",                # Axis
    "/live",
    "/live/ch0",
    "/stream1",
    "/h264",
    "/onvif1",
]

# What the camera is asked. DESCRIBE is the one that validates both the
# credentials and the path, which is exactly what needs testing.
_METHOD = "DESCRIBE"


@dataclass
class ProbeResult:
    """A verdict plus enough detail to act on it."""
    ok: bool = False
    stage: str = ""              # dns | tcp | rtsp | stream
    code: str = ""               # machine-readable cause
    message: str = ""            # what to tell the operator
    hint: str = ""               # what to do about it
    rtsp_status: int = 0
    auth_scheme: str = ""
    server: str = ""
    video: str = ""              # codec/resolution from the SDP, when offered
    working_path: str = ""
    tried_paths: List[str] = field(default_factory=list)
    elapsed_ms: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "rtsp_status": self.rtsp_status,
            "auth_scheme": self.auth_scheme,
            "server": self.server,
            "video": self.video,
            "working_path": self.working_path,
            "tried_paths": list(self.tried_paths),
            "elapsed_ms": self.elapsed_ms,
        }


# ---------------------------------------------------------------------------
# Digest auth. Cameras overwhelmingly use Digest rather than Basic, so without
# this every probe stops at "401" and cannot tell a wrong password from no
# password having been offered yet.
# ---------------------------------------------------------------------------

def _parse_challenge(header: str) -> Tuple[str, Dict[str, str]]:
    """Splits `Digest realm="x", nonce="y"` into its scheme and parameters."""
    header = header.strip()
    scheme, _, rest = header.partition(" ")
    params: Dict[str, str] = {}
    for m in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|([^,\s]+))', rest):
        params[m.group(1).lower()] = m.group(2) if m.group(2) is not None else m.group(3)
    return scheme.lower(), params


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _digest_header(user: str, password: str, uri: str, params: Dict[str, str]) -> str:
    realm = params.get("realm", "")
    nonce = params.get("nonce", "")
    qop = params.get("qop", "")
    ha1 = _md5(f"{user}:{realm}:{password}")
    ha2 = _md5(f"{_METHOD}:{uri}")

    bits = [f'username="{user}"', f'realm="{realm}"', f'nonce="{nonce}"', f'uri="{uri}"']
    if qop:
        # Cameras that advertise qop need the client nonce echoed back.
        cnonce = uuid.uuid4().hex[:16]
        nc = "00000001"
        response = _md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}")
        bits += ["qop=auth", f"nc={nc}", f'cnonce="{cnonce}"']
    else:
        response = _md5(f"{ha1}:{nonce}:{ha2}")
    bits.append(f'response="{response}"')

    opaque = params.get("opaque")
    if opaque:
        bits.append(f'opaque="{opaque}"')
    return "Digest " + ", ".join(bits)


# ---------------------------------------------------------------------------
# Talking to the camera
# ---------------------------------------------------------------------------

def _recv_response(sock: socket.socket, timeout: float) -> str:
    """Reads one RTSP response. Headers end at a blank line; any SDP body after
    it is useful but optional, so a read timeout mid-body is not an error."""
    sock.settimeout(timeout)
    buf = b""
    deadline = time.monotonic() + timeout
    while b"\r\n\r\n" not in buf and time.monotonic() < deadline:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            break
        if not chunk:
            break
        buf += chunk
    # One more read for the body, best effort.
    if b"\r\n\r\n" in buf:
        try:
            sock.settimeout(0.8)
            buf += sock.recv(16384)
        except (socket.timeout, OSError):
            pass
    return buf.decode("utf-8", errors="replace")


def _status_of(response: str) -> int:
    m = re.match(r"RTSP/\d\.\d\s+(\d{3})", response)
    return int(m.group(1)) if m else 0


def _header(response: str, name: str) -> str:
    m = re.search(rf"^{re.escape(name)}\s*:\s*(.+)$", response, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""


def _best_challenge(response: str) -> Tuple[str, Dict[str, str]]:
    """Extracts the preferred WWW-Authenticate challenge (preferring Digest over Basic)."""
    matches = re.findall(r"^WWW-Authenticate\s*:\s*(.+)$", response, re.IGNORECASE | re.MULTILINE)
    if not matches:
        return "", {}
    parsed = [_parse_challenge(m.strip()) for m in matches]
    for scheme, params in parsed:
        if scheme == "digest":
            return scheme, params
    return parsed[0]


def _describe(host: str, port: int, path: str, user: str, password: str,
              timeout: float) -> Tuple[int, str]:
    """One DESCRIBE, retried once with credentials if the camera asks.

    Returns the final status and the raw response.
    """
    if not path.startswith("/"):
        path = "/" + path
    uri = f"rtsp://{host}:{port}{path}"

    def _send_on(sock: socket.socket, extra_auth: str = "", cseq: int = 1) -> str:
        lines = [
            f"{_METHOD} {uri} RTSP/1.0",
            f"CSeq: {cseq}",
            "User-Agent: ThumbnailGrabber/1.0",
            "Accept: application/sdp",
        ]
        if extra_auth:
            lines.append(f"Authorization: {extra_auth}")
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("utf-8"))
        return _recv_response(sock, timeout)

    with socket.create_connection((host, port), timeout=timeout) as sock:
        resp = _send_on(sock)
        status = _status_of(resp)

        if status == 401 and (user or password):
            scheme, params = _best_challenge(resp)
            if scheme == "digest":
                auth = _digest_header(user, password, uri, params)
            else:
                token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode()
                auth = f"Basic {token}"
            try:
                resp2 = _send_on(sock, auth, cseq=2)
                status2 = _status_of(resp2)
            except (OSError, socket.timeout):
                resp2, status2 = "", 0

            if status2 == 0:
                # Some cameras close the TCP socket immediately after sending 401;
                # retry the authenticated request on a fresh connection.
                with socket.create_connection((host, port), timeout=timeout) as sock2:
                    resp2 = _send_on(sock2, auth, cseq=2)
                    status2 = _status_of(resp2)

            if status2 != 0:
                resp, status = resp2, status2

        return status, resp


def _video_summary(sdp: str) -> str:
    """Codec and resolution out of the SDP, when the camera volunteers them."""
    codec = ""
    m = re.search(r"^a=rtpmap:\d+\s+([A-Za-z0-9\-]+)", sdp, re.MULTILINE)
    if m:
        codec = m.group(1)
    size = ""
    m = re.search(r"a=framesize:\d+\s+(\d+)-(\d+)", sdp)
    if m:
        size = f"{m.group(1)}x{m.group(2)}"
    elif re.search(r"^m=video", sdp, re.MULTILINE):
        size = "video track present"
    return " ".join(p for p in (codec, size) if p)


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------

def probe(host: str, port: int = DEFAULT_PORT, path: str = "",
          user: str = "", password: str = "", timeout: float = DEFAULT_TIMEOUT,
          discover: bool = True) -> ProbeResult:
    """Works out whether this camera can be streamed, and if not, why.

    With `discover` and no `path`, or when the given path is rejected, the usual
    paths are tried and the one that answers is reported.
    """
    started = time.monotonic()
    r = ProbeResult()
    host = (host or "").strip()
    if not host:
        r.stage, r.code = "dns", "no_host"
        r.message = "No camera address was given."
        r.hint = "Enter the camera's IP address."
        return r

    # --- can we even find it ------------------------------------------------
    try:
        socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        r.stage, r.code = "dns", "unresolved"
        r.message = f"'{host}' could not be looked up."
        r.hint = "Check the spelling, or use the camera's IP address instead of a name."
        r.elapsed_ms = int((time.monotonic() - started) * 1000)
        return r

    # --- is the RTSP port open ----------------------------------------------
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except socket.timeout:
        r.stage, r.code = "tcp", "timeout"
        r.message = f"{host}:{port} did not answer within {timeout:.0f}s."
        r.hint = ("Usually a firewall or a different VLAN. Test from the machine "
                  "running this app, not your desktop.")
        r.elapsed_ms = int((time.monotonic() - started) * 1000)
        return r
    except ConnectionRefusedError:
        r.stage, r.code = "tcp", "refused"
        r.message = f"{host} refused the connection on port {port}."
        r.hint = ("The host is reachable but nothing is serving RTSP there. Check the "
                  "port — some cameras use 8554 — and that RTSP is enabled.")
        r.elapsed_ms = int((time.monotonic() - started) * 1000)
        return r
    except OSError as e:
        r.stage, r.code = "tcp", "unreachable"
        r.message = f"Could not reach {host}:{port} ({e.strerror or e})."
        r.hint = "Check the address and that the camera is powered on."
        r.elapsed_ms = int((time.monotonic() - started) * 1000)
        return r

    # --- does it like our credentials and path ------------------------------
    candidates = [path] if path else []
    if discover:
        candidates += [p for p in COMMON_PATHS if p != path]
    if not candidates:
        candidates = [COMMON_PATHS[0]]

    last_status = 0
    last_resp = ""
    for cand in candidates:
        r.tried_paths.append(cand)
        try:
            status, resp = _describe(host, port, cand, user, password, timeout)
        except (OSError, socket.timeout):
            continue
        last_status, last_resp = status, resp

        if status == 200:
            r.ok = True
            r.stage, r.code = "stream", "ok"
            r.rtsp_status = 200
            r.working_path = cand if cand.startswith("/") else "/" + cand
            r.server = _header(resp, "Server")
            r.video = _video_summary(resp)
            r.message = f"Connected. The stream is at {r.working_path}."
            if len(r.tried_paths) > 1:
                r.hint = f"The path you entered did not work; {r.working_path} did."
            r.elapsed_ms = int((time.monotonic() - started) * 1000)
            return r

        if status == 401:
            # Credentials are wrong, and no other path will change that, so stop
            # rather than walking the whole list collecting the same answer.
            scheme, _ = _best_challenge(resp)
            r.stage, r.code = "rtsp", "unauthorized"
            r.rtsp_status = 401
            r.auth_scheme = scheme
            r.server = _header(resp, "Server")
            if user or password:
                r.message = "The camera rejected that username and password."
                r.hint = ("Check them in the camera's own web page. Some cameras need a "
                          "separate streaming user from the admin login.")
            else:
                r.message = "The camera needs a username and password."
                r.hint = "Enter the camera's streaming credentials."
            r.elapsed_ms = int((time.monotonic() - started) * 1000)
            return r

    # --- reachable and authorised, but nothing streamed ---------------------
    r.stage, r.code = "rtsp", "no_path"
    r.rtsp_status = last_status
    r.server = _header(last_resp, "Server") if last_resp else ""
    tried = len(r.tried_paths)
    r.message = (f"{host}:{port} answered, but none of the {tried} stream paths tried "
                 f"returned a video stream.")
    r.hint = ("Find the RTSP path in the camera's web page, usually under Network or "
              "Streaming, and enter it in the Stream path field. For Sony SRG cameras "
              "it is normally /media/video1.")
    r.elapsed_ms = int((time.monotonic() - started) * 1000)
    return r


def build_url(host: str, port: int, path: str, user: str = "", password: str = "") -> str:
    """Assembles an RTSP URL. Credentials are percent-encoded, since camera
    passwords routinely contain characters that would otherwise break the URL."""
    from urllib.parse import quote
    if not path.startswith("/"):
        path = "/" + path
    auth = ""
    if user:
        auth = quote(user, safe="")
        if password:
            auth += ":" + quote(password, safe="")
        auth += "@"
    port_part = "" if port == DEFAULT_PORT else f":{port}"
    return f"rtsp://{auth}{host}{port_part}{path}"


# Redaction lives in config so there is exactly one implementation of it; a
# second copy is a second thing to get wrong.
from config import redact_credentials as redact  # noqa: E402


# ---------------------------------------------------------------------------

def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Diagnose an RTSP camera connection.")
    ap.add_argument("host")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--path", default="")
    ap.add_argument("--user", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--no-discover", action="store_true",
                    help="Only try the given path")
    a = ap.parse_args()

    r = probe(a.host, a.port, a.path, a.user, a.password, a.timeout,
              discover=not a.no_discover)
    mark = "OK " if r.ok else "FAIL"
    print(f"[{mark}] {r.message}")
    if r.hint:
        print(f"       {r.hint}")
    for label, value in (("stage", r.stage), ("code", r.code),
                         ("rtsp status", r.rtsp_status or ""), ("auth", r.auth_scheme),
                         ("server", r.server), ("video", r.video),
                         ("tried", ", ".join(r.tried_paths)),
                         ("took", f"{r.elapsed_ms} ms")):
        if value:
            print(f"       {label}: {value}")
    if r.ok:
        print(f"       url: {redact(build_url(a.host, a.port, r.working_path, a.user, a.password))}")
    return 0 if r.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
