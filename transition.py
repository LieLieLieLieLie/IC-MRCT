"""
transition.py
-------------
缺陷间过渡路径与姿态平滑。

功能
----
1. 连接两个缺陷的退刀点与下一缺陷的进刀点（贝塞尔曲线）
2. 沿过渡段连续插值工具法向（SLERP 球面插值）
3. 检测并修正超过阈值的姿态跳变
4. 完整路径拼接（approach → repair → retract → transit → next）
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from defect import Defect

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  贝塞尔过渡曲线
# ─────────────────────────────────────────────────────────────────────────────

def cubic_bezier(
    p0: np.ndarray, p1: np.ndarray,
    p2: np.ndarray, p3: np.ndarray,
    n_pts: int = 20,
) -> np.ndarray:
    """
    三次贝塞尔曲线：p0 → (p1, p2 为控制点) → p3。
    返回 (n_pts, 2) 坐标数组。
    """
    t  = np.linspace(0, 1, n_pts)[:, None]
    c  = (1 - t) ** 3 * p0 + \
         3 * (1 - t) ** 2 * t * p1 + \
         3 * (1 - t) * t ** 2 * p2 + \
         t ** 3 * p3
    return c   # (n_pts, 2)


def transit_bezier(
    from_xy:   Tuple[float, float],
    to_xy:     Tuple[float, float],
    ctrl_ratio: float = 0.35,
    n_pts:      int   = 20,
) -> np.ndarray:
    """
    在退刀点 from_xy 和进刀点 to_xy 之间生成三次贝塞尔过渡曲线。
    控制点沿两端连线方向向内偏移 ctrl_ratio 比例。
    """
    p0   = np.array(from_xy, dtype=float)
    p3   = np.array(to_xy,   dtype=float)
    d    = p3 - p0
    p1   = p0 + ctrl_ratio * d
    p2   = p3 - ctrl_ratio * d
    return cubic_bezier(p0, p1, p2, p3, n_pts)


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


def _first_blocking_obstacle(p0: np.ndarray, p1: np.ndarray, obstacles: List[Dict], margin: float):
    worst = None
    worst_clearance = np.inf
    for obs in obstacles or []:
        clearance = _segment_circle_clearance(p0, p1, obs) - margin
        if clearance < worst_clearance:
            worst = obs
            worst_clearance = clearance
    return worst if worst is not None and worst_clearance < 0.0 else None


def _clamp_workspace(q: np.ndarray, cfg) -> np.ndarray:
    q = np.asarray(q, dtype=float).copy()
    q[0] = np.clip(q[0], 1.0, cfg.surface.length - 1.0)
    q[1] = np.clip(q[1], 1.0, cfg.surface.width - 1.0)
    return q


def _detour_waypoint(p0: np.ndarray, p1: np.ndarray, obs: Dict, cfg, margin: float) -> np.ndarray:
    c = np.array([obs["cx"], obs["cy"]], dtype=float)
    v = p1 - p0
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        v = np.array([1.0, 0.0])
        norm = 1.0
    unit = v / norm
    perp = np.array([-unit[1], unit[0]])
    radius = float(obs["r"]) + margin
    candidates = []
    midpoint = 0.5 * (p0 + p1)
    along_shift = 0.35 * radius
    for sign in (-1.0, 1.0):
        for along in (-along_shift, 0.0, along_shift):
            q = c + sign * perp * radius + unit * along
            q += 0.12 * unit * float(np.dot(midpoint - c, unit))
            q = _clamp_workspace(q, cfg)
            score = np.linalg.norm(q - p0) + np.linalg.norm(p1 - q)
            score += 120.0 * max(0.0, margin - _segment_circle_clearance(p0, q, obs))
            score += 120.0 * max(0.0, margin - _segment_circle_clearance(q, p1, obs))
            candidates.append((score, q))
    return min(candidates, key=lambda item: item[0])[1]


def _densify_polyline(poly: np.ndarray, step: float = 8.0) -> np.ndarray:
    pts = [poly[0]]
    for p0, p1 in zip(poly[:-1], poly[1:]):
        length = float(np.linalg.norm(p1 - p0))
        n = max(1, int(np.ceil(length / step)))
        for k in range(1, n + 1):
            pts.append(p0 + (p1 - p0) * (k / n))
    return np.vstack(pts)


def _push_points_outside_obstacles(path: np.ndarray, obstacles: Optional[List[Dict]], cfg, margin: float) -> np.ndarray:
    if not obstacles:
        return path
    safe = path.copy()
    for _ in range(int(getattr(cfg.transition, "obstacle_repulsion_iters", 4))):
        for i in range(1, len(safe) - 1):
            q = safe[i]
            shift = np.zeros(2)
            for obs in obstacles:
                c = np.array([obs["cx"], obs["cy"]], dtype=float)
                required = float(obs["r"]) + margin
                vec = q - c
                dist = float(np.linalg.norm(vec))
                if dist < required:
                    if dist < 1e-9:
                        prev_dir = safe[i] - safe[i - 1]
                        vec = np.array([-prev_dir[1], prev_dir[0]], dtype=float)
                        dist = float(np.linalg.norm(vec)) + 1e-9
                    shift += (required - dist + 1.0) * vec / dist
            if np.linalg.norm(shift) > 0:
                safe[i] = _clamp_workspace(q + shift, cfg)
    return safe


def _resample_polyline(poly: np.ndarray, n_pts: int) -> np.ndarray:
    if len(poly) == n_pts:
        return poly
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    dist = np.r_[0.0, np.cumsum(seg)]
    if dist[-1] < 1e-9:
        return np.repeat(poly[:1], n_pts, axis=0)
    target = np.linspace(0.0, dist[-1], n_pts)
    out = []
    for t in target:
        j = int(np.searchsorted(dist, t, side="right") - 1)
        j = min(max(j, 0), len(seg) - 1)
        frac = (t - dist[j]) / (seg[j] + 1e-9)
        out.append(poly[j] * (1.0 - frac) + poly[j + 1] * frac)
    return np.vstack(out)


def obstacle_aware_polyline(
    from_xy: Tuple[float, float],
    to_xy: Tuple[float, float],
    obstacles: Optional[List[Dict]],
    cfg,
) -> np.ndarray:
    """Build a sparse 2-D polyline that keeps transit motions outside forbidden zones."""
    waypoints = [np.array(from_xy, dtype=float), np.array(to_xy, dtype=float)]
    if not obstacles or not getattr(cfg.transition, "obstacle_avoidance", False):
        return np.vstack(waypoints)

    margin = float(getattr(cfg.transition, "obstacle_margin", 8.0))
    max_insertions = int(getattr(cfg.transition, "obstacle_max_insertions", 8))
    for _ in range(max_insertions):
        inserted = False
        for i in range(len(waypoints) - 1):
            p0, p1 = waypoints[i], waypoints[i + 1]
            obs = _first_blocking_obstacle(p0, p1, obstacles, margin)
            if obs is None:
                continue
            waypoints.insert(i + 1, _detour_waypoint(p0, p1, obs, cfg, margin))
            inserted = True
            break
        if not inserted:
            break
    return np.vstack(waypoints)


def transit_obstacle_aware_bezier(
    from_xy: Tuple[float, float],
    to_xy: Tuple[float, float],
    obstacles: Optional[List[Dict]],
    cfg,
    ctrl_ratio: float,
    n_pts: int,
) -> np.ndarray:
    poly = obstacle_aware_polyline(from_xy, to_xy, obstacles, cfg)
    margin = float(getattr(cfg.transition, "obstacle_margin", 8.0))
    if len(poly) <= 2:
        path = transit_bezier(from_xy, to_xy, ctrl_ratio, n_pts)
        path = _push_points_outside_obstacles(path, obstacles, cfg, margin)
        return path

    path = _densify_polyline(poly, step=max(3.0, margin * 0.35))
    path = _push_points_outside_obstacles(path, obstacles, cfg, margin)
    return _resample_polyline(path, n_pts)


# ─────────────────────────────────────────────────────────────────────────────
#  法向量球面插值 (SLERP)
# ─────────────────────────────────────────────────────────────────────────────

def slerp_normals(
    n0: np.ndarray, n1: np.ndarray, t_arr: np.ndarray
) -> np.ndarray:
    """
    在两个单位法向量 n0, n1 之间做球面线性插值。
    t_arr : (N,) 插值参数，∈ [0, 1]
    返回 (N, 3) 插值法向量（已归一化）。
    """
    n0 = n0 / (np.linalg.norm(n0) + 1e-9)
    n1 = n1 / (np.linalg.norm(n1) + 1e-9)
    dot = np.clip(np.dot(n0, n1), -1.0, 1.0)
    omega = np.arccos(dot)

    if omega < 1e-6:   # 几乎平行
        return np.array([n0 * (1 - t) + n1 * t for t in t_arr])

    normals = np.array([
        (np.sin((1 - t) * omega) * n0 + np.sin(t * omega) * n1) / np.sin(omega)
        for t in t_arr
    ])
    return normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
#  单段过渡路径生成
# ─────────────────────────────────────────────────────────────────────────────

def make_transit_segment(
    from_xy:   Tuple[float, float],
    to_xy:     Tuple[float, float],
    normal_from: np.ndarray,
    normal_to:   np.ndarray,
    cfg,
    lift_height: float = 5.0,
    obstacles: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    生成从 from_xy（上一缺陷退刀点）到 to_xy（下一缺陷进刀点）的
    过渡轨迹，包含：
      - 抬刀（Z 方向）
      - 贝塞尔水平路径
      - 落刀（Z 方向）
      - 沿路径的 SLERP 法向插值

    Parameters
    ----------
    lift_height : 空走抬刀高度 (mm)，叠加在曲面 z 之上
    """
    from local_planner import surface_z, surface_normal

    ctrl   = cfg.transition.bezier_ctrl_ratio
    n_pts  = cfg.transition.posture_interp_n

    t_arr = np.linspace(0, 1, n_pts)
    if cfg.transition.smooth_transition:
        xy_path = transit_obstacle_aware_bezier(from_xy, to_xy, obstacles, cfg, ctrl, n_pts)
        normals = slerp_normals(normal_from, normal_to, t_arr)  # (n_pts, 3)
    else:
        xy_path = np.linspace(np.array(from_xy, dtype=float),
                              np.array(to_xy, dtype=float), n_pts)
        normals = np.array([
            normal_from if i < n_pts // 2 else normal_to
            for i in range(n_pts)
        ], dtype=float)
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)

    pts = []
    for i, (xy, nrm) in enumerate(zip(xy_path, normals)):
        x, y = float(xy[0]), float(xy[1])
        z    = surface_z(x, y, cfg)
        # 中段抬刀：平滑模式使用 cosine 形状；消融模式保留直连特征。
        lift = lift_height * np.sin(np.pi * t_arr[i]) if cfg.transition.smooth_transition else 0.0
        pts.append(dict(
            x=x, y=y, z=z + lift,
            nx=float(nrm[0]), ny=float(nrm[1]), nz=float(nrm[2]),
            speed=50.0,    # 空走较快
            zone="transit",
        ))
    return pts


# ─────────────────────────────────────────────────────────────────────────────
#  姿态跳变检测与插入
# ─────────────────────────────────────────────────────────────────────────────

def check_posture_jump(
    pts: List[Dict],
    max_jump_deg: float = 15.0,
) -> List[int]:
    """
    检测轨迹中超过阈值的法向量跳变，返回跳变位置的索引列表。
    """
    jumps = []
    max_rad = np.radians(max_jump_deg)
    for i in range(len(pts) - 1):
        n0 = np.array([pts[i]["nx"],     pts[i]["ny"],     pts[i]["nz"]])
        n1 = np.array([pts[i+1]["nx"],   pts[i+1]["ny"],   pts[i+1]["nz"]])
        cos_a = np.clip(np.dot(n0, n1), -1, 1)
        if np.arccos(cos_a) > max_rad:
            jumps.append(i)
    return jumps


# ─────────────────────────────────────────────────────────────────────────────
#  完整路径拼接器
# ─────────────────────────────────────────────────────────────────────────────

class TransitionPlanner:
    """
    将有序的缺陷局部轨迹拼接为完整的机器人路径。

    输入
    ----
    ordered_defects : 已排序的缺陷列表
    local_trajs     : 对应的局部轨迹列表（List[List[Dict]]）

    输出
    ----
    full_path : 完整路径（List[Dict]），含修复段和过渡段
    stats     : 统计信息字典
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def assemble(
        self,
        ordered_defects: List[Defect],
        local_trajs:     List[List[Dict]],
        robot_start:     Tuple[float, float] = (0.0, 0.0),
        obstacles:       Optional[List[Dict]] = None,
    ) -> Tuple[List[Dict], Dict]:
        """
        拼接完整路径并统计关键指标。
        """
        if not ordered_defects:
            return [], {}

        cfg           = self.cfg
        full_path     = []
        n_transit_pts = 0
        n_repair_pts  = 0
        n_approach    = 0
        max_jump      = 0.0

        # 上一缺陷的退刀点和法向
        prev_retract_xy = robot_start
        prev_normal     = np.array([0.0, 0.0, 1.0])

        for defect, traj in zip(ordered_defects, local_trajs):
            if not traj:
                continue

            # 当前缺陷进刀点
            approach_pts = [p for p in traj if p["zone"] == "approach"]
            entry_xy = ((approach_pts[0]["x"], approach_pts[0]["y"])
                        if approach_pts else (defect.cx, defect.cy))
            entry_normal = (np.array([approach_pts[0]["nx"],
                                      approach_pts[0]["ny"],
                                      approach_pts[0]["nz"]])
                            if approach_pts
                            else defect.surface_normal(cfg.surface))

            # 生成过渡段（上一退刀 → 当前进刀）
            transit = make_transit_segment(
                from_xy      = prev_retract_xy,
                to_xy        = entry_xy,
                normal_from  = prev_normal,
                normal_to    = entry_normal,
                cfg          = cfg,
                lift_height  = cfg.local_p.lift_height,
                obstacles    = obstacles,
            )
            full_path    += transit
            n_transit_pts += len(transit)

            # 追加当前缺陷局部轨迹
            full_path  += traj
            n_repair_pts  += sum(1 for p in traj if p["zone"] == "repair")
            n_approach    += sum(1 for p in traj if p["zone"] == "approach")

            # 更新退刀点
            retract_pts = [p for p in traj if p["zone"] == "retract"]
            if retract_pts:
                last = retract_pts[-1]
                prev_retract_xy = (last["x"], last["y"])
                prev_normal     = np.array([last["nx"], last["ny"], last["nz"]])
            else:
                prev_retract_xy = (defect.cx, defect.cy)
                prev_normal     = defect.surface_normal(cfg.surface)

        # 姿态跳变统计
        jump_idxs = check_posture_jump(
            full_path, cfg.transition.max_posture_jump)
        if full_path:
            from local_planner import surface_normal as sn
            normals = np.array([[p["nx"], p["ny"], p["nz"]]
                                 for p in full_path])
            dots    = np.clip((normals[:-1] * normals[1:]).sum(axis=1), -1, 1)
            angles  = np.degrees(np.arccos(dots))
            max_jump = float(angles.max()) if len(angles) > 0 else 0.0

        # 路程统计
        xy = np.array([[p["x"], p["y"]] for p in full_path])
        total_dist  = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        transit_dist = 0.0
        i = 0
        while i < len(full_path):
            j = i + 1
            while j < len(full_path) and full_path[j]["zone"] == full_path[i]["zone"]:
                j += 1
            if full_path[i]["zone"] == "transit" and j - i > 1:
                transit_xy = np.array([[p["x"], p["y"]] for p in full_path[i:j]], dtype=float)
                transit_dist += float(np.linalg.norm(np.diff(transit_xy, axis=0), axis=1).sum())
            i = j

        stats = {
            "total_points":   len(full_path),
            "repair_points":  n_repair_pts,
            "transit_points": n_transit_pts,
            "approach_points":n_approach,
            "total_dist_mm":  total_dist,
            "transit_dist_mm":transit_dist,
            "idle_ratio":     transit_dist / (total_dist + 1e-9),
            "n_posture_jumps":len(jump_idxs),
            "max_posture_jump_deg": max_jump,
        }

        logger.info(
            f"路径拼接完成 | 总点数={stats['total_points']} | "
            f"修复={n_repair_pts} | 过渡={n_transit_pts} | "
            f"空走比={stats['idle_ratio']:.2%} | "
            f"最大姿态跳变={max_jump:.1f}°"
        )
        return full_path, stats
