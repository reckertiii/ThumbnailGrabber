"""NDI Network Stream Discovery & Input Adapter (#12).

Keeps the public Docker image 100% license-clean while giving TrueNAS deployments
drop-in NDI v5/v6 support:
  1. Searches for `libndi.so*` (Linux) or `Processing.NDI.Lib.x64.dll` (Windows) in
     `$TG_NDI_LIB`, `$TG_DATA_DIR/ndi/` (`/mnt/Main/apps/tg-models/ndi/`),
     `$NDI_RUNTIME_DIR_V6`, `$NDI_RUNTIME_DIR_V5`, and `/usr/local/lib` / `/usr/lib`.
  2. When `libndi` is present, binds the C ABI via `ctypes` to discover sources on
     the VLAN (`NDIlib_find_create_v2`) and pull uncompressed `BGRX` frames directly
     into OpenCV via `NDIVideoCapture` (`ndi://<Source Name>`).
  3. Even before `libndi.so` is placed in `TG_DATA_DIR/ndi/`, runs a zero-dependency
     UDP mDNS query (`_ndi._tcp.local.`) so the UI can detect NDI senders on the
     network and show a one-step setup hint.
"""

from __future__ import annotations

import ctypes
import glob
import os
import socket
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _candidate_ndi_paths() -> List[str]:
    paths: List[str] = []
    explicit = os.environ.get("TG_NDI_LIB", "").strip()
    if explicit:
        paths.append(explicit)

    data_dir = os.environ.get("TG_DATA_DIR", "").strip() or os.path.join(os.getcwd(), "models")
    ndi_dir = os.path.join(data_dir, "ndi")
    for env_key in ("NDI_RUNTIME_DIR_V6", "NDI_RUNTIME_DIR_V5", "NDI_RUNTIME_DIR_V4"):
        ev = os.environ.get(env_key, "").strip()
        if ev:
            paths.extend(glob.glob(os.path.join(ev, "*ndi*")))

    if os.name == "nt":
        paths.extend(glob.glob(os.path.join(ndi_dir, "Processing.NDI.Lib*.dll")))
        prog = os.environ.get("ProgramFiles", r"C:\Program Files")
        paths.extend(glob.glob(os.path.join(prog, "NDI", "**", "Processing.NDI.Lib.x64.dll"), recursive=True))
    else:
        for folder in (ndi_dir, "/usr/local/lib", "/usr/lib", "/usr/lib/x86_64-linux-gnu", "/opt/ndi/lib"):
            paths.extend(sorted(glob.glob(os.path.join(folder, "libndi.so*")), reverse=True))
    return paths


# --- Minimal ctypes NDI v5 ABI structs --------------------------------------

class _NDIlib_source_t(ctypes.Structure):
    _fields_ = [
        ("p_ndi_name", ctypes.c_char_p),
        ("p_url_address", ctypes.c_char_p),
    ]


class _NDIlib_find_create_t(ctypes.Structure):
    _fields_ = [
        ("show_local_sources", ctypes.c_bool),
        ("p_groups", ctypes.c_char_p),
        ("p_extra_ips", ctypes.c_char_p),
    ]


class _NDIlib_recv_create_v3_t(ctypes.Structure):
    _fields_ = [
        ("source_to_connect_to", _NDIlib_source_t),
        ("color_format", ctypes.c_int),       # 0 = BGRX_BGRA
        ("bandwidth", ctypes.c_int),          # 100 = highest
        ("allow_video_fields", ctypes.c_bool),
        ("p_ndi_recv_name", ctypes.c_char_p),
    ]


class _NDIlib_video_frame_v2_t(ctypes.Structure):
    _fields_ = [
        ("xres", ctypes.c_int),
        ("yres", ctypes.c_int),
        ("FourCC", ctypes.c_int),
        ("frame_rate_N", ctypes.c_int),
        ("frame_rate_D", ctypes.c_int),
        ("picture_aspect_ratio", ctypes.c_float),
        ("frame_format_type", ctypes.c_int),
        ("timecode", ctypes.c_int64),
        ("p_data", ctypes.POINTER(ctypes.c_uint8)),
        ("line_stride_in_bytes", ctypes.c_int),
        ("p_metadata", ctypes.c_char_p),
        ("timestamp", ctypes.c_int64),
    ]


_NDI_LIB: Any = None
_NDI_LIB_PATH: str = ""
_NDI_LOAD_ERR: str = ""


def _load_ndi_library() -> Any:
    global _NDI_LIB, _NDI_LIB_PATH, _NDI_LOAD_ERR
    if _NDI_LIB is not None:
        return _NDI_LIB
    for cand in _candidate_ndi_paths():
        if not os.path.isfile(cand):
            continue
        try:
            lib = ctypes.CDLL(cand)
            lib.NDIlib_initialize.restype = ctypes.c_bool
            if not lib.NDIlib_initialize():
                _NDI_LOAD_ERR = f"NDIlib_initialize() returned false for {cand}"
                continue
            lib.NDIlib_find_create_v2.argtypes = [ctypes.POINTER(_NDIlib_find_create_t)]
            lib.NDIlib_find_create_v2.restype = ctypes.c_void_p
            lib.NDIlib_find_destroy.argtypes = [ctypes.c_void_p]
            lib.NDIlib_find_wait_for_sources.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            lib.NDIlib_find_wait_for_sources.restype = ctypes.c_bool
            lib.NDIlib_find_get_current_sources.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)
            ]
            lib.NDIlib_find_get_current_sources.restype = ctypes.POINTER(_NDIlib_source_t)

            lib.NDIlib_recv_create_v3.argtypes = [ctypes.POINTER(_NDIlib_recv_create_v3_t)]
            lib.NDIlib_recv_create_v3.restype = ctypes.c_void_p
            lib.NDIlib_recv_destroy.argtypes = [ctypes.c_void_p]
            lib.NDIlib_recv_capture_v2.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_NDIlib_video_frame_v2_t),
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
            ]
            lib.NDIlib_recv_capture_v2.restype = ctypes.c_int
            lib.NDIlib_recv_free_video_v2.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(_NDIlib_video_frame_v2_t)
            ]
            _NDI_LIB = lib
            _NDI_LIB_PATH = cand
            _NDI_LOAD_ERR = ""
            return lib
        except Exception as e:
            _NDI_LOAD_ERR = f"{cand}: {e}"
    return None


def _mdns_probe_ndi(timeout_sec: float = 0.8) -> List[str]:
    """Zero-dependency multicast DNS probe for `_ndi._tcp.local.` service names."""
    found: List[str] = []
    # Standard DNS PTR query packet for _ndi._tcp.local.
    qname = b"\x04_ndi\x04_tcp\x05local\x00"
    packet = struct.pack(">HHHHHH", 0, 0, 1, 0, 0, 0) + qname + struct.pack(">HH", 12, 1)
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(timeout_sec)
        sock.sendto(packet, ("224.0.0.251", 5353))
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                break
            if b"_ndi" in data:
                # Extract printable ASCII labels preceding ._ndi._tcp
                text = "".join(chr(b) if 32 <= b <= 126 else "|" for b in data)
                for chunk in text.split("|"):
                    c = chunk.strip()
                    if len(c) >= 3 and c not in ("_ndi", "_tcp", "local") and not c.startswith("_"):
                        label = f"{c} ({addr[0]})"
                        if label not in found:
                            found.append(label)
                        break
    except Exception:
        pass
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
    return found


def ndi_status() -> Dict[str, Any]:
    lib = _load_ndi_library()
    data_dir = os.environ.get("TG_DATA_DIR", "").strip() or os.path.join(os.getcwd(), "models")
    drop_dir = os.path.join(data_dir, "ndi")
    return {
        "available": lib is not None,
        "library_path": _NDI_LIB_PATH,
        "drop_in_dir": drop_dir,
        "error": _NDI_LOAD_ERR,
        "setup_hint": (
            "" if lib is not None else
            f"Place libndi.so.6 (Linux) or Processing.NDI.Lib.x64.dll into {drop_dir} to enable live NDI video decode."
        ),
    }


def discover_ndi_sources(timeout_ms: int = 1200) -> Dict[str, Any]:
    """Discovers active NDI sources on the network via `libndi` (or mDNS fallback)."""
    st = ndi_status()
    lib = _load_ndi_library()
    sources: List[Dict[str, str]] = []
    if lib is not None:
        cfg = _NDIlib_find_create_t(show_local_sources=True, p_groups=None, p_extra_ips=None)
        finder = lib.NDIlib_find_create_v2(ctypes.byref(cfg))
        if finder:
            try:
                lib.NDIlib_find_wait_for_sources(finder, int(max(200, timeout_ms)))
                count = ctypes.c_uint32(0)
                ptr = lib.NDIlib_find_get_current_sources(finder, ctypes.byref(count))
                for i in range(int(count.value)):
                    raw_name = ptr[i].p_ndi_name
                    raw_addr = ptr[i].p_url_address
                    name = raw_name.decode("utf-8", "replace") if raw_name else ""
                    addr = raw_addr.decode("utf-8", "replace") if raw_addr else ""
                    if name:
                        sources.append({
                            "name": name,
                            "url": f"ndi://{name}",
                            "address": addr,
                        })
            finally:
                lib.NDIlib_find_destroy(finder)
        return {**st, "sources": sources, "discovery_mode": "libndi"}

    mdns_hits = _mdns_probe_ndi(timeout_sec=min(1.5, timeout_ms / 1000.0))
    for h in mdns_hits:
        sources.append({"name": h, "url": f"ndi://{h}", "address": ""})
    return {**st, "sources": sources, "discovery_mode": "mdns_only"}


class NDIVideoCapture:
    """`cv2.VideoCapture`-compatible wrapper around `NDIlib_recv`."""

    FRAME_TYPE_VIDEO = 1

    def __init__(self, source_url_or_name: str, extra_ips: str = "") -> None:
        self._lib = _load_ndi_library()
        self._recv: Any = None
        self._opened: bool = False
        self._last_frame: Optional[np.ndarray] = None
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 30.0

        name = (source_url_or_name or "").strip()
        if name.lower().startswith("ndi://"):
            name = name[6:].strip()
        self.source_name = name

        if self._lib is not None and self.source_name:
            self._connect(extra_ips=extra_ips)

    def _connect(self, extra_ips: str = "") -> None:
        lib = self._lib
        extra_b = extra_ips.encode("utf-8") if extra_ips else None
        f_cfg = _NDIlib_find_create_t(show_local_sources=True, p_groups=None, p_extra_ips=extra_b)
        finder = lib.NDIlib_find_create_v2(ctypes.byref(f_cfg))
        matched_src = _NDIlib_source_t(
            p_ndi_name=self.source_name.encode("utf-8"),
            p_url_address=None,
        )
        if finder:
            try:
                lib.NDIlib_find_wait_for_sources(finder, 1500)
                count = ctypes.c_uint32(0)
                ptr = lib.NDIlib_find_get_current_sources(finder, ctypes.byref(count))
                want_low = self.source_name.lower()
                for i in range(int(count.value)):
                    nm = ptr[i].p_ndi_name.decode("utf-8", "replace") if ptr[i].p_ndi_name else ""
                    if nm.lower() == want_low or want_low in nm.lower():
                        matched_src = _NDIlib_source_t(
                            p_ndi_name=ptr[i].p_ndi_name,
                            p_url_address=ptr[i].p_url_address,
                        )
                        break
            finally:
                lib.NDIlib_find_destroy(finder)

        r_cfg = _NDIlib_recv_create_v3_t(
            source_to_connect_to=matched_src,
            color_format=0,   # NDIlib_recv_color_format_BGRX_BGRA
            bandwidth=100,    # NDIlib_recv_bandwidth_highest
            allow_video_fields=False,
            p_ndi_recv_name=b"Thumbnail Grabber",
        )
        self._recv = lib.NDIlib_recv_create_v3(ctypes.byref(r_cfg))
        self._opened = bool(self._recv)

    def isOpened(self) -> bool:
        return self._opened

    def grab(self) -> bool:
        ok, _ = self.read()
        return ok

    def retrieve(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._last_frame is None:
            return False, None
        return True, self._last_frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened or not self._recv or self._lib is None:
            return False, None
        vf = _NDIlib_video_frame_v2_t()
        # Wait up to 1000ms across up to 3 attempts (metadata/audio packets may arrive between video frames)
        for _ in range(4):
            ftype = self._lib.NDIlib_recv_capture_v2(self._recv, ctypes.byref(vf), None, None, 500)
            if ftype == self.FRAME_TYPE_VIDEO:
                try:
                    w, h = int(vf.xres), int(vf.yres)
                    stride = int(vf.line_stride_in_bytes) or (w * 4)
                    if w > 0 and h > 0 and vf.p_data:
                        total_bytes = stride * h
                        buf = ctypes.string_at(vf.p_data, total_bytes)
                        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, stride))
                        bgrx = arr[:, : w * 4].reshape((h, w, 4))
                        bgr = np.ascontiguousarray(bgrx[:, :, :3])
                        self._width, self._height = w, h
                        if vf.frame_rate_D > 0:
                            self._fps = float(vf.frame_rate_N) / float(vf.frame_rate_D)
                        self._last_frame = bgr
                        return True, bgr
                finally:
                    self._lib.NDIlib_recv_free_video_v2(self._recv, ctypes.byref(vf))
        return False, None

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self._fps)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        return False

    def release(self) -> None:
        if self._recv and self._lib is not None:
            try:
                self._lib.NDIlib_recv_destroy(self._recv)
            except Exception:
                pass
        self._recv = None
        self._opened = False
