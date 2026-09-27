"""Shared HUD overlay renderer.

Draws the analysis overlay (hand skeletons, shoulder pose, per-person face
boxes, composite score meter) onto a BGR frame. Used by both the PyQt desktop
GUI and the web server so both surfaces render identically.
"""
import cv2
import numpy as np
from typing import Optional

from analyzer import AnalysisResult, SingleFaceAnalysis, HAND_CONNECTIONS

# Color palette for up to 6 tracked people (BGR)
PERSON_COLORS = [
    (0, 255, 0),      # P1: Green
    (255, 255, 0),    # P2: Cyan
    (255, 191, 0),    # P3: Sky Blue
    (255, 0, 255),    # P4: Magenta
    (0, 165, 255),    # P5: Orange
    (0, 255, 255),    # P6: Yellow
]


def draw_overlay(frame_bgr: np.ndarray, analysis: Optional[AnalysisResult],
                 show_mesh: bool = True, show_hud: bool = True) -> np.ndarray:
    """Returns a copy of frame_bgr with the analysis HUD drawn on it."""
    display_frame = frame_bgr.copy()
    if analysis is None:
        return display_frame

    # 1. Hand skeletal overlay
    if getattr(analysis, 'hand_skeletons', None):
        for hand_pts in analysis.hand_skeletons:
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(hand_pts) and end_idx < len(hand_pts):
                    cv2.line(display_frame, hand_pts[start_idx], hand_pts[end_idx], (255, 255, 0), 2)
            for px, py in hand_pts:
                cv2.circle(display_frame, (px, py), 3, (0, 0, 255), -1)

    # 2. Pose & shoulder level overlay
    if show_hud and getattr(analysis, 'pose_skeletons', None):
        for pose in analysis.pose_skeletons:
            if not (pose.get('visible') and pose.get('left_shoulder') and pose.get('right_shoulder')):
                continue
            ls, rs = pose['left_shoulder'], pose['right_shoulder']
            is_lvl = pose.get('is_level', True)
            tilt = pose.get('tilt_deg', 0.0)

            sh_color = (74, 222, 128) if is_lvl else (0, 140, 255)
            cv2.line(display_frame, ls, rs, sh_color, 3)
            cv2.circle(display_frame, ls, 5, (56, 189, 248), -1)
            cv2.circle(display_frame, rs, 5, (56, 189, 248), -1)

            if pose.get('left_elbow'):
                cv2.line(display_frame, ls, pose['left_elbow'], (56, 189, 248), 2)
                cv2.circle(display_frame, pose['left_elbow'], 4, (0, 200, 255), -1)
            if pose.get('right_elbow'):
                cv2.line(display_frame, rs, pose['right_elbow'], (56, 189, 248), 2)
                cv2.circle(display_frame, pose['right_elbow'], 4, (0, 200, 255), -1)

            mid_x = int((ls[0] + rs[0]) / 2)
            mid_y = int((ls[1] + rs[1]) / 2) - 8
            badge_txt = f"{tilt:.1f}deg {'[LEVEL]' if is_lvl else '[TILTED]'}"
            (bw, bh), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(display_frame, (mid_x - bw // 2 - 4, mid_y - bh - 4),
                          (mid_x + bw // 2 + 4, mid_y + 4), (15, 23, 42), -1)
            cv2.rectangle(display_frame, (mid_x - bw // 2 - 4, mid_y - bh - 4),
                          (mid_x + bw // 2 + 4, mid_y + 4), sh_color, 1)
            cv2.putText(display_frame, badge_txt, (mid_x - bw // 2, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, sh_color, 1)

    # 3. Multi-person face HUD
    if analysis.face_detected and show_hud:
        faces_to_render = analysis.faces if getattr(analysis, 'faces', None) else []

        # Fallback for legacy single-face results
        if not faces_to_render and analysis.face_bbox:
            faces_to_render = [SingleFaceAnalysis(
                person_id=1, bbox=analysis.face_bbox, score=analysis.overall_score,
                is_qualified=analysis.is_qualified, is_blinking=analysis.is_blinking,
                is_smiling=analysis.is_smiling, is_awkward_expression=analysis.is_awkward_expression,
                mesh_points=analysis.mesh_points
            )]

        for idx, f_item in enumerate(faces_to_render):
            bx, by, bw_, bh_ = f_item.bbox
            is_sel = getattr(f_item, 'is_selected', True)

            if is_sel:
                color = PERSON_COLORS[idx % len(PERSON_COLORS)] if f_item.is_qualified else (0, 165, 255)
                label = f"P{f_item.person_id}: {int(f_item.score * 100)}%"
                if getattr(f_item, 'is_licking_lips', False):
                    label += " [LICKING]"
                elif getattr(f_item, 'is_talking_or_open', False):
                    label += " [OPEN MOUTH]"
                elif getattr(f_item, 'is_weird_mouth', False):
                    label += " [WEIRD MOUTH]"
                elif f_item.is_blinking:
                    label += " [BLINK]"
                elif f_item.is_smiling:
                    label += " [SMILE]"
                elif f_item.is_looking_down:
                    label += " [LOOK DOWN]"
                elif f_item.is_awkward_expression:
                    label += " [SPEECH]"
                elif f_item.is_shoulder_tilted:
                    label += " [SHOULDER TILT]"
                elif f_item.is_qualified:
                    label += " [CLEAN]"

                cv2.rectangle(display_frame, (bx, by), (bx + bw_, by + bh_), color, 2)
                cv2.rectangle(display_frame, (bx, max(0, by - 26)),
                              (bx + max(140, len(label) * 11), by), color, -1)
                cv2.putText(display_frame, label, (bx + 6, max(12, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            else:
                color = (120, 120, 120)
                label = f"OFF P{f_item.person_id} (Ignored)"
                cv2.rectangle(display_frame, (bx, by), (bx + bw_, by + bh_), color, 1)
                cv2.rectangle(display_frame, (bx, max(0, by - 22)), (bx + 140, by), color, -1)
                cv2.putText(display_frame, label, (bx + 6, max(12, by - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            if show_mesh and is_sel and f_item.mesh_points is not None:
                # The full mesh is 478 points over a face often only ~120px tall, so
                # fat dots would paint it solid. Scale the dot to the point count.
                radius = 1 if len(f_item.mesh_points) > 64 else 4
                for pt in f_item.mesh_points:
                    cv2.circle(display_frame, (int(pt[0]), int(pt[1])), radius, (0, 255, 255), -1)

        # Why nothing is being captured, under the score meter. Without this, a high
        # score with no saved shots looks like a broken capture button.
        if not analysis.is_qualified:
            reasons = list(getattr(analysis, "disqualified_reasons", []) or [])
            for i, reason in enumerate(reasons[:3]):
                y = 150 + i * 20
                cv2.putText(display_frame, f"- {reason}", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 190, 255), 1)

        # Composite score meter, top-left
        cv2.rectangle(display_frame, (15, 15), (380, 135), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (15, 15), (380, 135), (51, 65, 85), 1)

        score_color = (74, 222, 128) if analysis.overall_score >= 0.65 else (96, 165, 250)
        found_cnt = getattr(analysis, 'people_detected_count', 1)

        cv2.putText(display_frame, f"Active Quality Score: {int(analysis.overall_score * 100)}%",
                    (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.68, score_color, 2)
        cv2.putText(display_frame, f"People: {found_cnt} Detected (Click Face to Toggle)",
                    (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (56, 189, 248), 1)
        cv2.putText(display_frame, f"Primary Yaw: {analysis.yaw_deg:+.1f}deg Pitch: {analysis.pitch_deg:+.1f}deg",
                    (25, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1)

        if analysis.shoulder_detected:
            sh_txt = f"Shoulders: {analysis.shoulder_tilt_deg:.1f}deg ({'LEVEL' if analysis.is_shoulders_level else 'TILTED'})"
            sh_col = (74, 222, 128) if analysis.is_shoulders_level else (248, 113, 113)
        else:
            sh_txt = "Shoulders: Headshot (Out of Frame)"
            sh_col = (148, 163, 184)
        cv2.putText(display_frame, sh_txt, (25, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sh_col, 1)

    return display_frame
