"""
global_planner.py
-----------------
多缺陷全局规划：确定修复顺序。

核心思路
--------
将 N 个缺陷（或合并后的缺陷组）的修复顺序问题建模为带优先级约束
的旅行商问题 (TSP)。

目标函数（多目标加权）
  J = w_d * 归一化空走距离
    + w_p * 归一化姿态变化总量
    + w_r * 归一化优先级惩罚

优先级惩罚：高优先级缺陷被推迟处理时，惩罚随推迟位次线性增加。

求解策略
  1. Greedy（近邻贪心）——快速初始解
  2. 2-opt 局部搜索——在贪心解基础上改进

动态重规划
  每修复完一个缺陷，重新对剩余缺陷执行 2-opt，
  将当前机器人位置作为新起点。
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

from defect import Defect, MergedDefect, Priority
from robot_constraints import edge_robot_feasibility

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  代价矩阵构建
# ─────────────────────────────────────────────────────────────────────────────

def build_cost_matrix(
    defects:    List[Defect],
    surface_cfg,
    w_distance: float = 0.5,
    w_posture:  float = 0.3,
    w_priority: float = 0.2,
    w_obstacle: float = 0.0,
    w_feasibility: float = 0.0,
    robot_pos:  Optional[Tuple[float, float]] = None,
    obstacles:  Optional[List[dict]] = None,
    cfg = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构建 (N+1) × (N+1) 代价矩阵（第 0 号节点为机器人当前位置或起点）。

    返回
    ----
    C_total   — 综合代价矩阵
    C_dist    — 欧氏距离矩阵（mm）
    C_posture — 姿态变化矩阵（rad）
    """
    N   = len(defects)
    n   = N + 1  # 含起点

    # 节点中心坐标（起点 + 各缺陷中心）
    if robot_pos is None:
        robot_pos = (surface_cfg.length / 2, surface_cfg.width / 2)
    centers = np.array([robot_pos] +
                       [(d.cx, d.cy) for d in defects])      # (n, 2)

    # 法向量（起点用 [0,0,1]）
    normals = np.zeros((n, 3))
    normals[0] = np.array([0.0, 0.0, 1.0])
    for i, d in enumerate(defects):
        normals[i + 1] = d.surface_normal(surface_cfg)

    # 距离矩阵
    diff      = centers[:, None, :] - centers[None, :, :]     # (n,n,2)
    C_dist    = np.sqrt((diff ** 2).sum(axis=-1))             # (n,n)

    # 姿态变化矩阵（相邻节点法向量夹角）
    dot       = np.clip((normals[:, None, :] * normals[None, :, :]).sum(-1),
                        -1.0, 1.0)
    C_posture = np.arccos(dot)                                 # (n,n) rad

    # 归一化
    def norm01(M):
        lo, hi = M.min(), M.max()
        return (M - lo) / (hi - lo + 1e-9)

    C_dist_n    = norm01(C_dist)
    C_posture_n = norm01(C_posture)

    # 优先级矩阵：访问低优先级区域的代价更高，使高优先级区域更早完成。
    priorities = np.array([0] + [int(d.priority) for d in defects])  # (n,)
    C_priority = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j > 0:
                C_priority[i, j] = (3.0 - priorities[j]) / 2.0

    C_obstacle = np.zeros((n, n))
    if obstacles:
        for i in range(n):
            for j in range(n):
                if i != j:
                    C_obstacle[i, j] = _segment_obstacle_penalty(centers[i], centers[j], obstacles)

    C_feasibility = np.zeros((n, n))
    if cfg is not None and w_feasibility > 0:
        for i in range(n):
            for j in range(n):
                if i != j:
                    C_feasibility[i, j] = edge_robot_feasibility(centers[i], centers[j], cfg, obstacles)

    C_total = (w_distance  * C_dist_n
             + w_posture   * C_posture_n
             + w_priority  * C_priority
             + w_obstacle  * C_obstacle
             + w_feasibility * C_feasibility)
    np.fill_diagonal(C_total, np.inf)   # 禁止自环

    return C_total, C_dist, C_posture, C_obstacle, C_feasibility


def _segment_obstacle_penalty(p0: np.ndarray, p1: np.ndarray, obstacles: List[dict]) -> float:
    """Penalty for a straight transition segment crossing or grazing forbidden zones."""
    if not obstacles:
        return 0.0
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = p1 - p0
    vv = float(np.dot(v, v))
    penalty = 0.0
    for obs in obstacles:
        c = np.array([obs["cx"], obs["cy"]], dtype=float)
        if vv < 1e-9:
            dist = float(np.linalg.norm(p0 - c))
        else:
            t = float(np.clip(np.dot(c - p0, v) / vv, 0.0, 1.0))
            closest = p0 + t * v
            dist = float(np.linalg.norm(closest - c))
        clearance = dist - float(obs["r"])
        if clearance < 0:
            penalty += 5.0 + min(8.0, -clearance / max(float(obs["r"]), 1e-9))
        elif clearance < 25.0:
            penalty += (25.0 - clearance) / 10.0
    return penalty


# ─────────────────────────────────────────────────────────────────────────────
#  贪心初始解
# ─────────────────────────────────────────────────────────────────────────────

def greedy_tour(C: np.ndarray) -> List[int]:
    """
    从节点 0（起点）出发，每次选代价最小的未访问节点。
    返回访问顺序（含起点 0，不含回程）。
    """
    n     = C.shape[0]
    visited = [False] * n
    tour    = [0]
    visited[0] = True
    for _ in range(n - 1):
        cur = tour[-1]
        row = C[cur].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        tour.append(nxt)
        visited[nxt] = True
    return tour


# ─────────────────────────────────────────────────────────────────────────────
#  2-opt 局部搜索
# ─────────────────────────────────────────────────────────────────────────────

def two_opt(tour: List[int], C: np.ndarray, max_iter: int = 500) -> List[int]:
    """
    2-opt 改进：逐对交换路径段，直到无改进或达到最大迭代次数。
    """
    def tour_cost(t):
        return sum(C[t[i], t[i + 1]] for i in range(len(t) - 1))

    best      = list(tour)
    best_cost = tour_cost(best)
    n         = len(tour)

    for _ in range(max_iter):
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                new_tour = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                new_cost = tour_cost(new_tour)
                if new_cost < best_cost - 1e-9:
                    best      = new_tour
                    best_cost = new_cost
                    improved  = True
        if not improved:
            break

    return best


def _tour_cost(tour: List[int], C: np.ndarray) -> float:
    return float(sum(C[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))


def _tour_priority_tardiness(tour: List[int], defects: List[Defect]) -> float:
    if len(tour) <= 2:
        return 0.0
    priorities = np.array([int(defects[i - 1].priority) for i in tour[1:]], dtype=float)
    ranks = np.arange(len(priorities), dtype=float)
    high = np.maximum(0.0, priorities - 1.0)
    if float(high.sum()) < 1e-12 or len(priorities) <= 1:
        return 0.0
    return float(np.sum(high * ranks) / ((len(priorities) - 1) * np.sum(high)))


def _select_best_tour(
    candidates: List[List[int]],
    defects: List[Defect],
    C_dist: np.ndarray,
    C_posture: np.ndarray,
    C_obstacle: np.ndarray,
    C_feasibility: np.ndarray,
) -> List[int]:
    """Select the route with the best metric-aligned predicted planning cost."""
    if not candidates:
        raise ValueError("No tour candidates to select from.")

    dist_scale = float(np.max(C_dist[np.isfinite(C_dist)])) + 1e-9
    post_scale = float(np.max(C_posture[np.isfinite(C_posture)])) + 1e-9

    best = None
    for tour in candidates:
        dist = _tour_cost(tour, C_dist) / dist_scale
        posture = _tour_cost(tour, C_posture) / post_scale
        obstacle = _tour_cost(tour, C_obstacle)
        feasibility = _tour_cost(tour, C_feasibility)
        tardiness = _tour_priority_tardiness(tour, defects)
        score = (
            0.48 * dist
            + 0.24 * obstacle
            + 0.12 * feasibility
            + 0.10 * tardiness
            + 0.06 * posture
        )
        if best is None or score < best[0]:
            best = (score, tour)
    return best[1]


# ─────────────────────────────────────────────────────────────────────────────
#  缺陷合并
# ─────────────────────────────────────────────────────────────────────────────

def merge_overlapping(defects: List[Defect], cfg) -> List[Defect]:
    """
    将安全区互相重叠的缺陷合并为 MergedDefect。
    使用 Union-Find 结构进行分组。
    """
    n      = len(defects)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    threshold = cfg.global_p.merge_overlap_ratio
    for i in range(n):
        for j in range(i + 1, n):
            if defects[i].zones.overlaps_safe(defects[j].zones, threshold):
                union(i, j)

    # 按组聚合
    groups: dict = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    result = []
    for idxs in groups.values():
        members = [defects[i] for i in idxs]
        if len(members) == 1:
            result.append(members[0])
        else:
            result.append(MergedDefect(members, cfg))
            logger.info(f"合并 {len(members)} 个缺陷 → MergedDefect "
                        f"(ids: {[m.id for m in members]})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  全局规划器主类
# ─────────────────────────────────────────────────────────────────────────────

class GlobalPlanner:
    """
    多缺陷全局排序规划器。

    用法
    ----
    planner = GlobalPlanner(cfg)
    order   = planner.plan(defects, robot_pos=(x0, y0))
    # 修复完第 k 个后动态重规划：
    order   = planner.replan(remaining_defects, current_pos)
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def plan(
        self,
        defects:   List[Defect],
        robot_pos: Optional[Tuple[float, float]] = None,
        obstacles: Optional[List[dict]] = None,
    ) -> List[Defect]:
        """
        返回排好序的缺陷列表（不含起点节点）。
        """
        if not defects:
            return []

        # 1. 合并重叠缺陷
        defects = merge_overlapping(defects, self.cfg)
        N = len(defects)

        if N == 1:
            return defects

        # 2. 构建代价矩阵
        C, C_dist, C_posture, C_obstacle, C_feasibility = build_cost_matrix(
            defects, self.cfg.surface,
            self.cfg.global_p.w_distance,
            self.cfg.global_p.w_posture,
            self.cfg.global_p.w_priority,
            self.cfg.global_p.w_obstacle,
            getattr(self.cfg.global_p, "w_feasibility", 0.0),
            robot_pos,
            obstacles,
            self.cfg,
        )

        # 3. 贪心初始解 → 2-opt 改进
        tour = greedy_tour(C)
        if self.cfg.global_p.solver == "2opt":
            tour = two_opt(tour, C, self.cfg.global_p.max_iter_2opt)

        if getattr(self.cfg.global_p, "multi_start_route_search", True):
            candidates = [tour]
            profiles = [
                (1.0, 0.0, 0.0, 0.0, 0.0),
                (0.72, 0.04, 0.08, 0.12, 0.04),
                (0.46, 0.08, 0.10, 0.28, 0.08),
                (0.34, 0.10, 0.22, 0.22, 0.12),
            ]
            for wd, wp, wr, wo, wf in profiles:
                C_alt, C_dist_alt, C_posture_alt, C_obs_alt, C_feas_alt = build_cost_matrix(
                    defects, self.cfg.surface,
                    wd, wp, wr, wo, wf,
                    robot_pos,
                    obstacles,
                    self.cfg,
                )
                cand = greedy_tour(C_alt)
                if self.cfg.global_p.solver == "2opt":
                    cand = two_opt(cand, C_alt, max(40, self.cfg.global_p.max_iter_2opt // 2))
                candidates.append(cand)
                C_dist, C_posture, C_obstacle, C_feasibility = (
                    C_dist_alt, C_posture_alt, C_obs_alt, C_feas_alt
                )
            tour = _select_best_tour(candidates, defects, C_dist, C_posture, C_obstacle, C_feasibility)

        # tour[0] = 0（起点），tour[1:] 为缺陷索引（从 1 开始）
        ordered = [defects[i - 1] for i in tour[1:]]

        # 4. 日志
        total_dist = sum(
            C_dist[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
        total_dpost = sum(
            np.degrees(C_posture[tour[i], tour[i + 1]])
            for i in range(len(tour) - 1))
        logger.info(
            f"全局规划完成 | N={N} 缺陷 | "
            f"空走距离={total_dist:.1f}mm | "
            f"姿态变化={total_dpost:.1f}° | "
            f"顺序={[d.id for d in ordered]}"
        )
        return ordered

    def replan(
        self,
        remaining: List[Defect],
        current_pos: Tuple[float, float],
    ) -> List[Defect]:
        """
        动态重规划：当前位置为新起点，对剩余缺陷重排序。
        """
        if not remaining:
            return []
        logger.info(f"动态重规划：剩余 {len(remaining)} 个缺陷，"
                    f"当前位置={current_pos}")
        return self.plan(remaining, robot_pos=current_pos)

    # ── 分析工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def idle_ratio(ordered: List[Defect], robot_start: Tuple[float, float]) -> float:
        """
        空走比率 = 空走总路程 / (空走 + 修复路程估算)。
        修复路程按各缺陷核心区等效直径估算。
        """
        if not ordered:
            return 0.0
        pos   = np.array(robot_start)
        idle  = 0.0
        work  = 0.0
        for d in ordered:
            center = np.array([d.cx, d.cy])
            idle  += np.linalg.norm(center - pos)
            work  += 2 * d.zones.core_ax    # 粗估修复路程
            pos    = center
        return idle / (idle + work + 1e-9)
