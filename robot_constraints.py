"""
robot_constraints.py
--------------------
Lightweight robot-constraint proxies for constrained trajectory planning.

The simulation does not bind to a specific industrial manipulator model.
This module therefore exposes task-space feasibility checks that mirror the
robotics constraints used in deployment: IK reachability, joint-limit margin,
singularity margin, tool collision, and body collision.  A real robot can
replace these functions with model-specific IK and collision-checking APIs
without changing the planner interface.
"""

from typing import Dict, Iterable, List, Tuple

import numpy as np


def _surface_z(x: float, y: float, surface_cfg) -> float:
    return surface_cfg.curvature_x * x**2 + surface_cfg.curvature_y * y**2


def _normal_tilt_deg(n: np.ndarray) -> float:
    n = np.asarray(n, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))


def _segment_circle_clearance(p0: np.ndarray, p1: np.ndarray, obs: Dict) -> float:
    c = np.array([obs["cx"], obs["cy"]], dtype=float)
    v = p1 - p0
    vv = float(np.dot(v, v))
    if vv < 1e-12:
        dist = float(np.linalg.norm(p0 - c))
    else:
        t = float(np.clip(np.dot(c - p0, v) / vv, 0.0, 1.0))
        dist = float(np.linalg.norm((p0 + t * v) - c))
    return dist - float(obs["r"])


def point_robot_feasibility(x: float, y: float, normal: Iterable[float], cfg, obstacles=None) -> Dict:
    """Evaluate proxy robot constraints at one task-space waypoint."""
    rc = cfg.robot
    z = _surface_z(float(x), float(y), cfg.surface)
    base = np.array([rc.base_x, rc.base_y, rc.base_z], dtype=float)
    p = np.array([x, y, z], dtype=float)
    reach = float(np.linalg.norm(p - base))

    reach_violation = max(0.0, rc.min_reach - reach) / max(rc.min_reach, 1e-9)
    reach_violation += max(0.0, reach - rc.max_reach) / max(rc.max_reach, 1e-9)

    tilt = _normal_tilt_deg(np.asarray(normal, dtype=float))
    tilt_violation = max(0.0, tilt - rc.max_tool_tilt_deg) / max(rc.max_tool_tilt_deg, 1e-9)

    mid = 0.5 * (rc.min_reach + rc.max_reach)
    half = 0.5 * (rc.max_reach - rc.min_reach)
    joint_margin = float(np.clip(1.0 - abs(reach - mid) / (half + 1e-9), 0.0, 1.0))
    joint_limit_risk = max(0.0, 0.15 - joint_margin) / 0.15

    singularity_margin = float(np.clip(abs(reach - rc.min_reach) / (rc.max_reach - rc.min_reach + 1e-9), 0.0, 1.0))
    singularity_risk = max(0.0, rc.min_singularity_margin - singularity_margin) / max(rc.min_singularity_margin, 1e-9)

    tool_collision = 0.0
    body_collision = 0.0
    q = np.array([x, y], dtype=float)
    b = np.array([rc.base_x, rc.base_y], dtype=float)
    for obs in obstacles or []:
        tool_clear = float(np.linalg.norm(q - np.array([obs["cx"], obs["cy"]], dtype=float))) - float(obs["r"])
        tool_collision += max(0.0, rc.tool_radius + rc.collision_margin - tool_clear) / max(rc.tool_radius, 1e-9)
        body_clear = _segment_circle_clearance(b, q, obs)
        body_collision += max(0.0, rc.body_radius + rc.collision_margin - body_clear) / max(rc.body_radius, 1e-9)

    violation = reach_violation + tilt_violation + joint_limit_risk + singularity_risk + tool_collision + body_collision
    return {
        "ik_reach_violation": float(reach_violation),
        "tilt_violation": float(tilt_violation),
        "joint_limit_risk": float(joint_limit_risk),
        "singularity_risk": float(singularity_risk),
        "tool_collision_risk": float(tool_collision),
        "body_collision_risk": float(body_collision),
        "robot_constraint_violation": float(violation),
        "joint_margin": float(joint_margin),
        "singularity_margin": float(singularity_margin),
    }


def edge_robot_feasibility(p0: Tuple[float, float], p1: Tuple[float, float], cfg, obstacles=None, n_samples: int = 9) -> float:
    """Return a normalized edge penalty for robot feasibility along a transition."""
    vals = []
    for t in np.linspace(0.0, 1.0, n_samples):
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        vals.append(point_robot_feasibility(x, y, (0.0, 0.0, 1.0), cfg, obstacles)["robot_constraint_violation"])
    return float(np.mean(vals))


def trajectory_robot_metrics(full_path: List[Dict], cfg, obstacles=None) -> Dict:
    """Aggregate robot-constraint metrics over an assembled trajectory."""
    if not full_path:
        return {
            "robot_constraint_violation": 0.0,
            "ik_violation": 0.0,
            "joint_limit_risk": 0.0,
            "singularity_risk": 0.0,
            "tool_collision_risk": 0.0,
            "body_collision_risk": 0.0,
        }

    checks = [
        point_robot_feasibility(
            p["x"],
            p["y"],
            (p.get("nx", 0.0), p.get("ny", 0.0), p.get("nz", 1.0)),
            cfg,
            obstacles,
        )
        for p in full_path
    ]

    return {
        "robot_constraint_violation": float(np.mean([c["robot_constraint_violation"] for c in checks])),
        "ik_violation": float(np.mean([c["ik_reach_violation"] for c in checks])),
        "joint_limit_risk": float(np.mean([c["joint_limit_risk"] for c in checks])),
        "singularity_risk": float(np.mean([c["singularity_risk"] for c in checks])),
        "tool_collision_risk": float(np.mean([c["tool_collision_risk"] for c in checks])),
        "body_collision_risk": float(np.mean([c["body_collision_risk"] for c in checks])),
        "min_joint_margin": float(np.min([c["joint_margin"] for c in checks])),
        "min_singularity_margin": float(np.min([c["singularity_margin"] for c in checks])),
    }
