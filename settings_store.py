"""Persistent UI settings and per-folder capture/pin manifests in TG_DATA_DIR.

Container restarts (such as Watchtower pulling a new image) wipe in-memory state.
Storing UI preferences (`ui_settings.json`) and folder-scoped gallery manifests
(`manifests/<hash>.json`) inside TG_DATA_DIR keeps SMB capture directories free
of hidden JSON clutter while preserving:
  - Chosen output folder & confirmation state
  - Threshold sliders, buffer depth, max capture timer, and view toggles
  - Captured gallery cards, scores, AI Judge summaries, and 📌 Pinned flags
  - Clean per-folder isolation when switching between campus/service directories
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

UI_SETTINGS_FILE = "ui_settings.json"
_SCORE_RE = re.compile(r"_score_?(\d{1,3})", re.IGNORECASE)
_TS_RE = re.compile(r"best_shot_(\d{8}_\d{6})", re.IGNORECASE)


def _data_dir() -> str:
    return (os.environ.get("TG_DATA_DIR", "").strip()
            or os.path.join(os.getcwd(), "models"))


def ui_settings_path() -> str:
    return os.path.join(_data_dir(), UI_SETTINGS_FILE)


def _atomic_write_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def load_ui_settings() -> Dict[str, Any]:
    """Returns persisted UI settings dict, or {} if none exist."""
    p = ui_settings_path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError) as e:
        print(f"[SettingsStore] Ignoring unreadable {UI_SETTINGS_FILE}: {e}")
        return {}


def save_ui_settings(patch: Dict[str, Any]) -> Optional[str]:
    """Merges `patch` into persisted UI settings atomically."""
    if not patch:
        return None
    current = load_ui_settings()
    current.update(patch)
    try:
        return _atomic_write_json(ui_settings_path(), current)
    except OSError as e:
        print(f"[SettingsStore] Could not save {UI_SETTINGS_FILE}: {e}")
        return None


def _normalize_dir_key(output_dir: str) -> str:
    try:
        norm = str(Path(output_dir).resolve())
    except Exception:
        norm = os.path.abspath(output_dir)
    if os.name == "nt":
        norm = norm.lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:20]


def manifest_path_for_dir(output_dir: str) -> str:
    key = _normalize_dir_key(output_dir)
    return os.path.join(_data_dir(), "manifests", f"captures_{key}.json")


def _analysis_to_dict(a: Any) -> Dict[str, Any]:
    if a is None:
        return {}
    fields = (
        "face_detected", "people_detected_count", "overall_score", "smoothed_score",
        "ml_score", "ml_trained", "is_qualified", "is_stable",
        "head_pose_score", "eye_openness_score", "expression_score",
        "shoulder_pose_score", "sharpness_score", "hand_pose_score",
        "exposure_score", "centering_score", "yaw_deg", "pitch_deg", "roll_deg",
        "shoulder_tilt_deg", "is_shoulders_level", "is_shoulder_tilted",
        "shoulder_detected", "avg_ear", "is_blinking", "mar", "smile_ratio",
        "is_smiling", "is_awkward_expression", "laplacian_variance",
        "face_brightness", "cutout_score", "subject_height_ratio",
        "is_subject_clipped", "hands_on_torso", "background_score",
        "torso_clear_score", "is_prop_moment", "mood_bucket", "color_cast_score",
        "speaker_name",
    )
    out: Dict[str, Any] = {}
    for k in fields:
        if hasattr(a, k):
            val = getattr(a, k)
            if isinstance(val, (bool, int, float, str)) or val is None:
                out[k] = val
    bbox = getattr(a, "face_bbox", None)
    if bbox and len(bbox) == 4:
        out["face_bbox"] = [int(x) for x in bbox]
    return out


def _dict_to_analysis(raw: Dict[str, Any]):
    from analyzer import AnalysisResult
    a = AnalysisResult()
    if not isinstance(raw, dict):
        return a
    for k, v in raw.items():
        if k == "face_bbox" and isinstance(v, (list, tuple)) and len(v) == 4:
            a.face_bbox = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
        elif hasattr(a, k):
            setattr(a, k, v)
    return a


def save_captures_manifest(output_dir: str, captures: List[Any]) -> Optional[str]:
    """Persists gallery captures (including pinned state & AI judge notes) for `output_dir`."""
    if not output_dir:
        return None
    base = Path(output_dir)
    items = []
    for c in captures:
        fp = getattr(c, "filepath", "") or ""
        if not fp:
            continue
        try:
            rel_fp = Path(fp).resolve().relative_to(base.resolve()).as_posix()
        except Exception:
            rel_fp = os.path.basename(fp)

        crop_fp = getattr(c, "crop_filepath", None)
        rel_crop = None
        if crop_fp:
            try:
                rel_crop = Path(crop_fp).resolve().relative_to(base.resolve()).as_posix()
            except Exception:
                rel_crop = os.path.basename(crop_fp)

        cutout_fp = getattr(c, "cutout_filepath", None)
        rel_cutout = None
        if cutout_fp:
            try:
                rel_cutout = Path(cutout_fp).resolve().relative_to(base.resolve()).as_posix()
            except Exception:
                rel_cutout = os.path.basename(cutout_fp)

        items.append({
            "rel_path": rel_fp,
            "crop_rel_path": rel_crop,
            "cutout_rel_path": rel_cutout,
            "timestamp_str": getattr(c, "timestamp_str", "") or "",
            "overall_score": float(getattr(c, "overall_score", 0.0) or 0.0),
            "captured_at": float(getattr(c, "captured_at", 0.0) or 0.0),
            "heuristic_score": float(getattr(c, "heuristic_score", 0.0) or 0.0),
            "judge_score": getattr(c, "judge_score", None),
            "judge_summary": getattr(c, "judge_summary", "") or "",
            "judge_issues": list(getattr(c, "judge_issues", []) or []),
            "is_prop_moment": bool(getattr(c, "is_prop_moment", False)),
            "from_video_scan": bool(getattr(c, "from_video_scan", False)),
            "is_pinned": bool(getattr(c, "is_pinned", False)),
            "finals_badge": str(getattr(c, "finals_badge", "") or ""),
            "mood_bucket": str(getattr(c, "mood_bucket", "") or ""),
            "analysis": _analysis_to_dict(getattr(c, "analysis", None)),
        })

    payload = {
        "version": 1,
        "output_dir": str(output_dir),
        "captures": items,
    }
    try:
        return _atomic_write_json(manifest_path_for_dir(output_dir), payload)
    except OSError as e:
        print(f"[SettingsStore] Could not save capture manifest: {e}")
        return None


def load_captures_manifest(output_dir: str, adopt_orphans: bool = False) -> List[Any]:
    """Loads persisted captures for `output_dir`, dropping entries whose image was deleted."""
    from stream_processor import CapturedScreenshot
    from analyzer import AnalysisResult

    if not output_dir or not os.path.isdir(output_dir):
        return []

    base = Path(output_dir).resolve()
    mpath = manifest_path_for_dir(output_dir)
    loaded: List[CapturedScreenshot] = []
    seen_paths = set()

    if os.path.exists(mpath):
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for entry in (raw.get("captures") or []):
                rel = entry.get("rel_path") or ""
                if not rel:
                    continue
                full = (base / rel).resolve()
                if base not in full.parents or not full.is_file():
                    continue
                crop_rel = entry.get("crop_rel_path")
                crop_full = None
                if crop_rel:
                    cand_crop = (base / crop_rel).resolve()
                    if base in cand_crop.parents and cand_crop.is_file():
                        crop_full = str(cand_crop)

                cutout_rel = entry.get("cutout_rel_path")
                cutout_full = None
                if cutout_rel:
                    cand_cutout = (base / cutout_rel).resolve()
                    if base in cand_cutout.parents and cand_cutout.is_file():
                        cutout_full = str(cand_cutout)

                analysis = _dict_to_analysis(entry.get("analysis") or {})
                mood_b = str(entry.get("mood_bucket") or getattr(analysis, "mood_bucket", "") or "")
                cap = CapturedScreenshot(
                    filepath=str(full),
                    timestamp_str=str(entry.get("timestamp_str") or ""),
                    overall_score=float(entry.get("overall_score", 0.0) or 0.0),
                    analysis=analysis,
                    crop_filepath=crop_full,
                    captured_at=float(entry.get("captured_at") or os.path.getmtime(full)),
                    heuristic_score=float(entry.get("heuristic_score") or entry.get("overall_score", 0.0) or 0.0),
                    judge_score=(float(entry["judge_score"]) if entry.get("judge_score") is not None else None),
                    judge_summary=str(entry.get("judge_summary") or ""),
                    judge_issues=list(entry.get("judge_issues") or []),
                    is_prop_moment=bool(entry.get("is_prop_moment", False)),
                    from_video_scan=bool(entry.get("from_video_scan", False)),
                    is_pinned=bool(entry.get("is_pinned", False)),
                    cutout_filepath=cutout_full,
                    finals_badge=str(entry.get("finals_badge") or ""),
                    mood_bucket=mood_b,
                )
                loaded.append(cap)
                seen_paths.add(os.path.normcase(str(full)))
        except (OSError, ValueError) as e:
            print(f"[SettingsStore] Ignoring unreadable capture manifest: {e}")

    if adopt_orphans:
        try:
            for fname in sorted(os.listdir(output_dir)):
                low = fname.lower()
                if "_subject." in low or "_cutout." in low or "_polished." in low or "_4k." in low:
                    continue
                if not low.endswith((".png", ".jpg", ".jpeg")):
                    continue
                is_live_shot = low.startswith("best_shot_")
                is_scan_shot = "_rank" in low and "_score" in low
                if not (is_live_shot or is_scan_shot):
                    continue
                fpath = str((base / fname).resolve())
                if os.path.normcase(fpath) in seen_paths or not os.path.isfile(fpath):
                    continue
                m_score = _SCORE_RE.search(fname)
                score = min(1.0, max(0.0, int(m_score.group(1)) / 100.0)) if m_score else 0.70
                m_ts = _TS_RE.search(fname)
                mtime = os.path.getmtime(fpath)
                ts_str = m_ts.group(1) if m_ts else ""
                is_prop = "_prop_" in low or low.endswith("_prop.png") or low.endswith("_prop.jpg")
                analysis = AnalysisResult(face_detected=True, overall_score=score, is_qualified=True,
                                          is_prop_moment=is_prop)
                loaded.append(CapturedScreenshot(
                    filepath=fpath,
                    timestamp_str=ts_str,
                    overall_score=score,
                    analysis=analysis,
                    captured_at=mtime,
                    heuristic_score=score,
                    is_prop_moment=is_prop,
                    from_video_scan=is_scan_shot and not is_live_shot,
                    is_pinned=False,
                ))
                seen_paths.add(os.path.normcase(fpath))
        except OSError:
            pass

    return loaded
