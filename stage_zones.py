"""Interactive Stage Safe-Zone & Obstruction Polygon Masks (#9).

Coordinates are stored in normalised frame fractions (`0.0 .. 1.0`), making them
resolution-independent across 1080p, 4K, and sub-sampled preview frames:
  - `safe_zone`: Optional polygon `[[x, y], ...]` (>= 3 vertices). Faces whose center
    falls outside the safe zone (e.g. front-row congregation, worship team in the
    corner) are automatically unselected and ignored.
  - `obstruction_zones`: List of named polygons `[{"id", "name", "points": [[x, y], ...]}]`
    marking fixed stage traps (lectern, iPad stand, confidence monitor, drum shield).
    When a speaker's torso/body box overlaps an obstruction zone, `torso_clear_score`
    and `overall_score` are penalized proportionally, and heavy overlap (>= 22%) flags
    `"behind stage obstruction"`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _clamp_pts(points: Any) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    if not isinstance(points, (list, tuple)):
        return out
    for pt in points:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                x = max(0.0, min(1.0, float(pt[0])))
                y = max(0.0, min(1.0, float(pt[1])))
                out.append((round(x, 4), round(y, 4)))
            except (TypeError, ValueError):
                continue
    return out


@dataclass
class ObstructionZone:
    id: str
    name: str
    points: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "points": [[p[0], p[1]] for p in self.points],
        }


@dataclass
class StageZoneConfig:
    safe_zone: List[Tuple[float, float]] = field(default_factory=list)
    obstruction_zones: List[ObstructionZone] = field(default_factory=list)

    @property
    def has_safe_zone(self) -> bool:
        return len(self.safe_zone) >= 3

    @property
    def has_obstructions(self) -> bool:
        return any(len(z.points) >= 3 for z in self.obstruction_zones)

    def is_face_inside_safe_zone(self, bbox: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> bool:
        if not self.has_safe_zone or frame_w <= 0 or frame_h <= 0:
            return True
        fx, fy, fw, fh = bbox
        cx = (fx + fw * 0.5) / float(frame_w)
        cy = (fy + fh * 0.5) / float(frame_h)
        poly = np.array(self.safe_zone, dtype=np.float32) * 1000.0
        dist = cv2.pointPolygonTest(poly, (float(cx * 1000.0), float(cy * 1000.0)), False)
        return dist >= 0.0

    def compute_torso_obstruction(
        self,
        torso_rect: Tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, Optional[str]]:
        """Returns `(overlap_fraction 0..1, worst_zone_name)` for a speaker's body rect."""
        if not self.has_obstructions or frame_w <= 0 or frame_h <= 0:
            return 0.0, None
        tx, ty, tw, th = torso_rect
        if tw <= 2 or th <= 2:
            return 0.0, None

        # Fast 160x90 raster intersection (<0.1ms)
        gw, gh = 160, 90
        rx0 = max(0, min(gw - 1, int((tx / float(frame_w)) * gw)))
        ry0 = max(0, min(gh - 1, int((ty / float(frame_h)) * gh)))
        rx1 = max(rx0 + 1, min(gw, int(((tx + tw) / float(frame_w)) * gw)))
        ry1 = max(ry0 + 1, min(gh, int(((ty + th) / float(frame_h)) * gh)))
        torso_area = float((rx1 - rx0) * (ry1 - ry0))
        if torso_area <= 0:
            return 0.0, None

        worst_ratio = 0.0
        worst_name: Optional[str] = None
        mask = np.zeros((gh, gw), dtype=np.uint8)
        for z in self.obstruction_zones:
            if len(z.points) < 3:
                continue
            mask.fill(0)
            pts_px = np.array(
                [[int(round(px * gw)), int(round(py * gh))] for px, py in z.points],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [pts_px], 1)
            overlap = float(np.sum(mask[ry0:ry1, rx0:rx1])) / torso_area
            if overlap > worst_ratio:
                worst_ratio = overlap
                worst_name = z.name or "obstruction"
        return round(min(1.0, worst_ratio), 3), worst_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safe_zone": [[p[0], p[1]] for p in self.safe_zone],
            "obstruction_zones": [z.to_dict() for z in self.obstruction_zones],
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "StageZoneConfig":
        if not isinstance(raw, dict):
            return cls()
        safe = _clamp_pts(raw.get("safe_zone"))
        obs_list: List[ObstructionZone] = []
        for item in (raw.get("obstruction_zones") or []):
            if not isinstance(item, dict):
                continue
            pts = _clamp_pts(item.get("points"))
            if len(pts) >= 3:
                obs_list.append(ObstructionZone(
                    id=str(item.get("id") or uuid.uuid4().hex[:8]),
                    name=str(item.get("name") or "Stage Obstruction")[:40],
                    points=pts,
                ))
        return cls(safe_zone=safe if len(safe) >= 3 else [], obstruction_zones=obs_list)
