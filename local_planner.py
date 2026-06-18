"""
local_planner.py
----------------
Local adaptive repair trajectory generation.

The local planner is formulated as a lightweight optimization problem.
For each detected defect, it builds a coverage cost field from the defect
boundary, principal direction, surface-normal variation, curvature, and
contact footprint.  It then searches over trajectory direction, line
spacing, expansion range, speed distribution, posture angle, and
approach/retract point placement.

The generated path is executable and deterministic, but the local geometry
is decoded from optimized variables instead of selected from a defect-type
motion library.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from defect import Defect

logger = logging.getLogger(__name__)


DEFAULT_MOTION = {
    "SCRATCH": {"robot_speed": 30.0, "overlap_ratio": 50.0, "contact_angle": 85.0},
    "PIT": {"robot_speed": 25.0, "overlap_ratio": 45.0, "contact_angle": 90.0},
    "AREA": {"robot_speed": 40.0, "overlap_ratio": 40.0, "contact_angle": 90.0},
}


def surface_z(x, y, cfg):
    return cfg.surface.curvature_x * x**2 + cfg.surface.curvature_y * y**2


def surface_normal(x, y, cfg):
    a, b = cfg.surface.curvature_x, cfg.surface.curvature_y
    n = np.array([-2 * a * x, -2 * b * y, 1.0], dtype=float)
    return n / (np.linalg.norm(n) + 1e-9)


def make_point(x, y, cfg, zone="repair", params: Dict = None) -> Dict:
    z = surface_z(x, y, cfg)
    nx, ny, nz = surface_normal(x, y, cfg)
    p = params or {"robot_speed": 30.0, "overlap_ratio": 40.0, "contact_angle": 90.0}
    return dict(
        x=float(x),
        y=float(y),
        z=float(z),
        nx=float(nx),
        ny=float(ny),
        nz=float(nz),
        zone=zone,
        robot_speed=float(p["robot_speed"]),
        overlap_ratio=float(p["overlap_ratio"]),
        contact_angle=float(p["contact_angle"]),
    )


def approach_path(start_xy, entry_xy, cfg, params, n=5):
    slow = dict(params)
    slow["robot_speed"] = params["robot_speed"] * 0.5
    return [
        make_point(
            start_xy[0] + t * (entry_xy[0] - start_xy[0]),
            start_xy[1] + t * (entry_xy[1] - start_xy[1]),
            cfg,
            zone="approach",
            params=slow,
        )
        for t in np.linspace(0, 1, n)
    ]


def retract_path(exit_xy, end_xy, cfg, params, n=5):
    slow = dict(params)
    slow["robot_speed"] = params["robot_speed"] * 0.5
    return [
        make_point(
            exit_xy[0] + t * (end_xy[0] - exit_xy[0]),
            exit_xy[1] + t * (end_xy[1] - exit_xy[1]),
            cfg,
            zone="retract",
            params=slow,
        )
        for t in np.linspace(0, 1, n)
    ]


def _local_axes(angle: float) -> Tuple[np.ndarray, np.ndarray]:
    u = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    v = np.array([-np.sin(angle), np.cos(angle)], dtype=float)
    return u, v


def _ellipse_value(x: float, y: float, defect: Defect, ax: float, ay: float, angle: float) -> float:
    dx, dy = x - defect.cx, y - defect.cy
    c, s = np.cos(angle), np.sin(angle)
    xr = c * dx + s * dy
    yr = -s * dx + c * dy
    return (xr / max(ax, 1e-6)) ** 2 + (yr / max(ay, 1e-6)) ** 2


def build_local_cost_field(defect: Defect, cfg, expand_factor: float = 1.0) -> Dict:
    z = defect.zones
    n = int(getattr(cfg.local_p, "local_field_samples", 21))
    ax = z.core_ax * expand_factor
    ay = z.core_ay * expand_factor
    xs = np.linspace(defect.cx - ax, defect.cx + ax, n)
    ys = np.linspace(defect.cy - ay, defect.cy + ay, n)

    samples, normals, weights = [], [], []
    for x in xs:
        for y in ys:
            core_val = _ellipse_value(float(x), float(y), defect, z.core_ax, z.core_ay, z.core_angle)
            trans_val = _ellipse_value(float(x), float(y), defect, ax, ay, z.core_angle)
            if trans_val <= 1.0:
                samples.append([float(x), float(y)])
                normals.append(surface_normal(float(x), float(y), cfg))
                weights.append(1.0 if core_val <= 1.0 else max(0.15, 1.0 - 0.55 * trans_val))

    samples = np.array(samples, dtype=float)
    normals = np.array(normals, dtype=float) if normals else np.zeros((0, 3))
    weights = np.array(weights, dtype=float) if weights else np.zeros(0)
    if len(normals) > 1:
        center_n = surface_normal(defect.cx, defect.cy, cfg)
        dots = np.clip(normals @ center_n, -1.0, 1.0)
        normal_var = float(np.degrees(np.arccos(dots)).mean())
    else:
        normal_var = 0.0

    curvature = abs(cfg.surface.curvature_x) + abs(cfg.surface.curvature_y)
    return {
        "samples": samples,
        "weights": weights,
        "normal_var": normal_var,
        "curvature": float(curvature),
    }


def _decode_local_segments(defect: Defect, angle: float, spacing: float, expand_factor: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Decode one local-path candidate from optimized variables."""
    z = defect.zones
    u, v = _local_axes(angle)
    half_l = z.core_ax * expand_factor
    half_w = z.core_ay * expand_factor
    s_vals = np.arange(-half_w, half_w + spacing, spacing)
    segments = []
    for i, s_val in enumerate(s_vals):
        p0 = np.array([defect.cx, defect.cy], dtype=float) + s_val * v - half_l * u
        p1 = np.array([defect.cx, defect.cy], dtype=float) + s_val * v + half_l * u
        if i % 2 == 1:
            p0, p1 = p1, p0
        segments.append((p0, p1))
    return segments


def _coverage_scores(field: Dict, segments: List[Tuple[np.ndarray, np.ndarray]], footprint: float) -> Tuple[float, float, float]:
    samples = field["samples"]
    weights = field["weights"]
    if len(samples) == 0 or not segments:
        return 0.0, 1.0, 1.0

    min_d2 = np.full(len(samples), np.inf)
    cover_radius = 0.50 * footprint
    duplicate_radius = 0.18 * footprint
    for p0, p1 in segments:
        v = p1 - p0
        vv = float(np.dot(v, v)) + 1e-9
        t = np.clip(((samples - p0) @ v) / vv, 0.0, 1.0)
        proj = p0 + t[:, None] * v
        min_d2 = np.minimum(min_d2, ((samples - proj) ** 2).sum(axis=1))

    covered = min_d2 <= cover_radius**2
    coverage = float(np.sum(weights * covered) / (np.sum(weights) + 1e-9))
    duplicate = float(np.mean(min_d2 < duplicate_radius**2))
    uncovered = np.sqrt(min_d2[~covered]) if np.any(~covered) else np.array([0.0])
    transition_smooth = float(np.mean(uncovered) / (footprint + 1e-9))
    return coverage, duplicate, transition_smooth


def _line_shape_penalty(defect: Defect, angle: float) -> float:
    """Penalize directions that fight the defect anisotropy."""
    z = defect.zones
    anisotropy = abs(z.core_ax - z.core_ay) / (max(z.core_ax, z.core_ay) + 1e-9)
    return float(anisotropy * (1.0 - abs(np.cos(angle - z.core_angle))))


def _path_length(segments: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    if not segments:
        return 0.0
    length = sum(float(np.linalg.norm(p1 - p0)) for p0, p1 in segments)
    for (_, prev_end), (next_start, _) in zip(segments[:-1], segments[1:]):
        length += float(np.linalg.norm(next_start - prev_end))
    return length


def _approach_risk(defect: Defect, angle: float, side: int, entry_offset: float, cfg) -> float:
    z = defect.zones
    u, v = _local_axes(angle)
    start = (
        np.array([defect.cx, defect.cy], dtype=float)
        - side * (z.safe_ax + cfg.local_p.approach_dist) * u
        + entry_offset * z.safe_ay * v
    )
    margin = 6.0
    outside = (
        max(0.0, margin - start[0])
        + max(0.0, start[0] - (cfg.surface.length - margin))
        + max(0.0, margin - start[1])
        + max(0.0, start[1] - (cfg.surface.width - margin))
    )
    return float(outside / 50.0)


def _reference_local_plan(defect: Defect, cfg, base_params: Dict) -> Dict:
    lp = cfg.local_p
    z = defect.zones
    footprint = float(lp.contact_footprint_width)
    spacing = max(1.0, footprint * (1.0 - base_params["overlap_ratio"] / 100.0))
    angle = z.core_angle
    if getattr(lp, "direction_strategy", "defect_axis") == "fixed_x":
        angle = 0.0
    elif getattr(lp, "direction_strategy", "defect_axis") == "curvature":
        angle = 0.0 if abs(cfg.surface.curvature_x) >= abs(cfg.surface.curvature_y) else np.pi / 2

    expand = 1.20
    field = build_local_cost_field(defect, cfg, expand)
    segments = _decode_local_segments(defect, angle, spacing, expand)
    coverage, duplicate, transition_smooth = _coverage_scores(field, segments, footprint)
    length = _path_length(segments)
    posture_cost = field["normal_var"] / 20.0 + field["curvature"] * length / 6000.0
    return {
        "score": float((1.0 - coverage) + transition_smooth + duplicate + posture_cost),
        "angle": float(angle),
        "spacing": float(spacing),
        "expand": float(expand),
        "side": 1,
        "params": dict(base_params),
        "coverage": coverage,
        "duplicate": duplicate,
        "transition_smooth": transition_smooth,
        "posture_cost": posture_cost,
        "path_length": float(length),
        "approach_risk": _approach_risk(defect, angle, 1, 0.0, cfg),
        "entry_offset": 0.0,
        "speed_taper": 0.0,
        "mode": "reference",
    }


def optimize_local_plan(defect: Defect, cfg, base_params: Dict) -> Dict:
    lp = cfg.local_p
    if not getattr(lp, "local_optimization", True):
        return _reference_local_plan(defect, cfg, base_params)

    z = defect.zones
    footprint = float(lp.contact_footprint_width)
    base_angle = z.core_angle
    best = None

    angle_offsets = (0.0,) if getattr(lp, "fixed_direction_only", False) else lp.candidate_angle_offsets
    for offset_deg in angle_offsets:
        angle = base_angle + np.radians(float(offset_deg))
        anisotropy_reward = abs(np.cos(angle - base_angle))
        for expand in lp.candidate_expand_factors:
            field = build_local_cost_field(defect, cfg, float(expand))
            for spacing_factor in lp.candidate_spacing_factors:
                spacing = max(1.0, footprint * float(spacing_factor))
                segments = _decode_local_segments(defect, angle, spacing, float(expand))
                coverage, duplicate, transition_smooth = _coverage_scores(field, segments, footprint)
                length = _path_length(segments)
                posture_cost = field["normal_var"] / 20.0 + field["curvature"] * length / 6000.0
                length_norm = length / (2.0 * max(z.trans_ax + z.trans_ay, 1e-9) * max(len(segments), 1))
                shape_penalty = _line_shape_penalty(defect, angle)
                for speed_factor in lp.candidate_speed_factors:
                    speed = base_params["robot_speed"] * float(speed_factor)
                    speed_risk = max(0.0, speed - base_params["robot_speed"]) / max(base_params["robot_speed"], 1e-9)
                    for speed_taper in getattr(lp, "candidate_speed_tapers", (0.0,)):
                        taper_cost = 0.04 * abs(float(speed_taper))
                        for entry_offset in getattr(lp, "candidate_entry_offsets", (0.0,)):
                            entry_offset = float(entry_offset)
                            entry_cost = 0.05 * abs(entry_offset)
                            for side in (-1, 1):
                                approach_risk = _approach_risk(defect, angle, side, entry_offset, cfg)
                                score = (
                                    2.40 * (1.0 - coverage)
                                    + 0.30 * transition_smooth
                                    + 0.18 * duplicate
                                    + 0.16 * posture_cost
                                    + 0.08 * length_norm
                                    + 0.16 * approach_risk
                                    + 0.08 * speed_risk
                                    + 0.06 * shape_penalty
                                    + taper_cost
                                    + entry_cost
                                    - 0.04 * anisotropy_reward
                                )
                                if best is None or score < best["score"]:
                                    params = dict(base_params)
                                    curvature_slowdown = np.clip(1.0 - 0.025 * field["normal_var"], 0.72, 1.0)
                                    params["robot_speed"] = speed * curvature_slowdown
                                    params["overlap_ratio"] = float(np.clip(100.0 * (1.0 - spacing / footprint), 20.0, 75.0))
                                    params["contact_angle"] = float(np.clip(
                                        base_params["contact_angle"] - 0.20 * field["normal_var"],
                                        78.0,
                                        92.0,
                                    ))
                                    best = {
                                        "score": float(score),
                                        "angle": float(angle),
                                        "spacing": float(spacing),
                                        "expand": float(expand),
                                        "side": int(side),
                                        "params": params,
                                        "coverage": coverage,
                                        "duplicate": duplicate,
                                        "transition_smooth": transition_smooth,
                                        "posture_cost": posture_cost,
                                        "path_length": float(length),
                                        "approach_risk": approach_risk,
                                        "entry_offset": entry_offset,
                                        "speed_taper": float(speed_taper),
                                        "mode": "optimized",
                                    }
    return best


def _with_speed_distribution(params: Dict, progress: float, taper: float) -> Dict:
    p = dict(params)
    if taper <= 1e-9:
        return p
    edge = 2.0 * abs(float(progress) - 0.5)
    p["robot_speed"] = params["robot_speed"] * (1.0 - taper * edge)
    return p


def optimized_trajectory(defect: Defect, cfg, plan: Dict) -> List[Dict]:
    lp = cfg.local_p
    angle = plan["angle"]
    spacing = plan["spacing"]
    expand = plan["expand"]
    side = plan["side"]
    params = plan["params"]
    entry_offset = float(plan.get("entry_offset", 0.0))
    speed_taper = float(plan.get("speed_taper", 0.0))
    segments = _decode_local_segments(defect, angle, spacing, expand)
    if side < 0:
        segments = [(p1, p0) for p0, p1 in segments[::-1]]
    if not segments:
        return []

    u, v_axis = _local_axes(angle)
    entry = segments[0][0]
    exit_xy = segments[-1][1]
    lateral = entry_offset * defect.zones.safe_ay * v_axis
    approach_start = entry - side * lp.approach_dist * u + lateral
    retract_end = exit_xy + side * lp.retract_dist * u + lateral
    pts = approach_path(tuple(approach_start), tuple(entry), cfg, params)

    total_segments = max(1, len(segments) - 1)
    for idx, (p0, p1) in enumerate(segments):
        if idx > 0:
            prev = segments[idx - 1][1]
            bridge_n = max(2, int(np.linalg.norm(p0 - prev) / max(spacing, 1.0)))
            for t in np.linspace(0, 1, bridge_n, endpoint=False)[1:]:
                p = prev + t * (p0 - prev)
                progress = (idx - 1 + t) / max(1, total_segments)
                pts.append(make_point(p[0], p[1], cfg, "repair", _with_speed_distribution(params, progress, speed_taper)))
        n_seg = max(3, int(np.linalg.norm(p1 - p0) / max(1.0, spacing * 0.6)))
        for t in np.linspace(0, 1, n_seg):
            p = p0 + t * (p1 - p0)
            progress = (idx + t) / max(1, total_segments)
            pts.append(make_point(p[0], p[1], cfg, "repair", _with_speed_distribution(params, progress, speed_taper)))

    pts += retract_path(tuple(exit_xy), tuple(retract_end), cfg, params)
    return pts


def _sample_repair_inside_transition(pts: List[Dict], defect: Defect) -> List[Dict]:
    """Keep repair samples inside the optimized transition repair zone."""
    filtered = []
    for p in pts:
        if p["zone"] != "repair" or defect.zones.contains_transition(p["x"], p["y"]):
            filtered.append(p)
    return filtered


class LocalPlanner:
    """Local optimization-based trajectory planner."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _motion_attributes(self, defect: Defect) -> Dict:
        params = DEFAULT_MOTION.get(defect.type.name, DEFAULT_MOTION["AREA"]).copy()
        if defect.depth_um >= self.cfg.defect.priority_high:
            params["robot_speed"] *= 0.8
            params["overlap_ratio"] = min(70.0, params["overlap_ratio"] + 10.0)
        elif defect.depth_um <= self.cfg.defect.priority_medium:
            params["robot_speed"] *= 1.1
        return params

    def plan(self, defect: Defect) -> List[Dict]:
        base_params = self._motion_attributes(defect)
        local_plan = optimize_local_plan(defect, self.cfg, base_params)
        pts = optimized_trajectory(defect, self.cfg, local_plan)
        pts = _sample_repair_inside_transition(pts, defect)
        params = local_plan["params"]
        logger.info(
            "[%s] id=%s: %d pts | theta=%.1f spacing=%.2f expand=%.2f "
            "coverage=%.3f duplicate=%.3f smooth=%.3f posture=%.3f entry=%.2f taper=%.2f "
            "speed=%.0f overlap=%.0f mode=%s",
            defect.type.name,
            defect.id,
            len(pts),
            np.degrees(local_plan["angle"]),
            local_plan["spacing"],
            local_plan["expand"],
            local_plan["coverage"],
            local_plan["duplicate"],
            local_plan["transition_smooth"],
            local_plan["posture_cost"],
            local_plan.get("entry_offset", 0.0),
            local_plan.get("speed_taper", 0.0),
            params["robot_speed"],
            params["overlap_ratio"],
            local_plan["mode"],
        )
        return pts

    def coverage_rate(self, pts: List[Dict], defect: Defect) -> float:
        repair = [(p["x"], p["y"]) for p in pts if p["zone"] == "repair"]
        if not repair:
            return 0.0
        return sum(1 for x, y in repair if defect.zones.contains_core(x, y)) / len(repair)
