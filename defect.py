"""
defect.py
---------
缺陷建模模块。

缺陷类型
--------
  Scratch  — 划痕：线段形，有长度、宽度、方向角
  Pit      — 凹坑：椭圆形，有长短轴
  AreaDefect — 面积型超差：凸多边形轮廓

三区结构（每种缺陷统一）
------------------------
  核心修复区（core）    ← 缺陷实际范围
  过渡修复区（transition）← 向外均匀外扩
  安全区（safe）         ← 进退刀缓冲区

优先级
------
  HIGH (3) > MEDIUM (2) > LOW (1)
  由深度决定，或可由外部指定。
"""

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Optional
import uuid


# ─────────────────────────────────────────────────────────────────────────────
#  枚举
# ─────────────────────────────────────────────────────────────────────────────

class DefectType(IntEnum):
    SCRATCH    = 1   # 划痕
    PIT        = 2   # 凹坑
    AREA       = 3   # 面积型


class Priority(IntEnum):
    LOW    = 1
    MEDIUM = 2
    HIGH   = 3


# ─────────────────────────────────────────────────────────────────────────────
#  三区包围盒（统一接口）
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DefectZones:
    """
    三区几何描述（以椭圆近似，便于解析计算）。
    cx, cy      — 中心坐标 (mm)
    ax, ay      — 半长轴（X/Y 方向）
    angle       — 主轴与 X 轴夹角 (rad)
    """
    # 核心修复区
    core_cx:   float = 0.0
    core_cy:   float = 0.0
    core_ax:   float = 5.0
    core_ay:   float = 5.0
    core_angle: float = 0.0

    # 过渡修复区
    trans_ax:  float = 7.5
    trans_ay:  float = 7.5

    # 安全区（进退刀）
    safe_ax:   float = 11.0
    safe_ay:   float = 11.0

    def contains_core(self, x: float, y: float) -> bool:
        return self._in_ellipse(x, y, self.core_ax, self.core_ay)

    def contains_transition(self, x: float, y: float) -> bool:
        return self._in_ellipse(x, y, self.trans_ax, self.trans_ay)

    def contains_safe(self, x: float, y: float) -> bool:
        return self._in_ellipse(x, y, self.safe_ax, self.safe_ay)

    def _in_ellipse(self, x: float, y: float, ax: float, ay: float) -> bool:
        dx = x - self.core_cx
        dy = y - self.core_cy
        c, s = np.cos(self.core_angle), np.sin(self.core_angle)
        xr =  c * dx + s * dy
        yr = -s * dx + c * dy
        return (xr / ax) ** 2 + (yr / ay) ** 2 <= 1.0

    @property
    def center(self) -> Tuple[float, float]:
        return self.core_cx, self.core_cy

    def approach_point(self, approach_dist: float) -> Tuple[float, float]:
        """沿主轴负方向的进刀点。"""
        c, s = np.cos(self.core_angle), np.sin(self.core_angle)
        d = self.safe_ax + approach_dist
        return self.core_cx - c * d, self.core_cy - s * d

    def retract_point(self, retract_dist: float) -> Tuple[float, float]:
        """沿主轴正方向的退刀点。"""
        c, s = np.cos(self.core_angle), np.sin(self.core_angle)
        d = self.safe_ax + retract_dist
        return self.core_cx + c * d, self.core_cy + s * d

    def overlaps_safe(self, other: "DefectZones", threshold: float = 0.3) -> bool:
        """判断两个安全区是否重叠（用于合并决策）。"""
        dist = np.hypot(self.core_cx - other.core_cx,
                        self.core_cy - other.core_cy)
        return dist < (self.safe_ax + other.safe_ax) * (1.0 - threshold)


# ─────────────────────────────────────────────────────────────────────────────
#  基类
# ─────────────────────────────────────────────────────────────────────────────

class Defect:
    """所有缺陷类型的基类。"""

    def __init__(
        self,
        cx: float, cy: float,
        depth_um: float,
        defect_type: DefectType,
        cfg,
        defect_id: Optional[str] = None,
    ):
        self.id          = defect_id or str(uuid.uuid4())[:8]
        self.cx          = cx          # mm
        self.cy          = cy          # mm
        self.depth_um    = depth_um    # 深度（严重程度指标）
        self.type        = defect_type
        self.cfg         = cfg
        self.repaired    = False
        self.zones: DefectZones = self._build_zones()
        self.priority: Priority = self._assign_priority()

    def _build_zones(self) -> DefectZones:
        raise NotImplementedError

    def _assign_priority(self) -> Priority:
        if self.depth_um >= self.cfg.defect.priority_high:
            return Priority.HIGH
        elif self.depth_um >= self.cfg.defect.priority_medium:
            return Priority.MEDIUM
        return Priority.LOW

    def surface_normal(self, surface_cfg) -> np.ndarray:
        """在缺陷中心处的曲面法向（解析计算二次曲面梯度）。"""
        a, b = surface_cfg.curvature_x, surface_cfg.curvature_y
        nx = -2 * a * self.cx
        ny = -2 * b * self.cy
        nz = 1.0
        n  = np.array([nx, ny, nz])
        return n / np.linalg.norm(n)

    def __repr__(self):
        return (f"<{self.type.name} id={self.id} "
                f"center=({self.cx:.1f},{self.cy:.1f}) "
                f"depth={self.depth_um:.0f}μm priority={self.priority.name}>")


# ─────────────────────────────────────────────────────────────────────────────
#  划痕 (Scratch)
# ─────────────────────────────────────────────────────────────────────────────

class Scratch(Defect):
    """
    线形划痕。
    length  — 划痕长度 (mm)
    width   — 划痕宽度 (mm)
    angle   — 划痕方向与 X 轴夹角 (rad)
    """

    def __init__(self, cx, cy, depth_um, length, width, angle, cfg):
        self.length = length
        self.width  = width
        self.angle  = angle   # rad
        super().__init__(cx, cy, depth_um, DefectType.SCRATCH, cfg)

    def _build_zones(self) -> DefectZones:
        cfg = self.cfg.defect
        # 核心区：以划痕长度为长轴、宽度为短轴的椭圆
        ax_core = self.length / 2
        ay_core = self.width  / 2
        return DefectZones(
            core_cx    = self.cx,
            core_cy    = self.cy,
            core_ax    = ax_core,
            core_ay    = ay_core,
            core_angle = self.angle,
            trans_ax   = ax_core * cfg.transition_expand,
            trans_ay   = ay_core * cfg.transition_expand,
            safe_ax    = ax_core * cfg.safe_expand,
            safe_ay    = ay_core * cfg.safe_expand,
        )

    @property
    def main_axis(self) -> np.ndarray:
        """划痕主轴单位向量。"""
        return np.array([np.cos(self.angle), np.sin(self.angle)])


# ─────────────────────────────────────────────────────────────────────────────
#  凹坑 (Pit)
# ─────────────────────────────────────────────────────────────────────────────

class Pit(Defect):
    """
    椭圆形凹坑。
    radius_a, radius_b — 长轴/短轴半径 (mm)
    angle              — 长轴与 X 轴夹角 (rad)
    """

    def __init__(self, cx, cy, depth_um, radius_a, radius_b, angle, cfg):
        self.radius_a = radius_a
        self.radius_b = radius_b
        self.angle    = angle
        super().__init__(cx, cy, depth_um, DefectType.PIT, cfg)

    def _build_zones(self) -> DefectZones:
        cfg = self.cfg.defect
        return DefectZones(
            core_cx    = self.cx,
            core_cy    = self.cy,
            core_ax    = self.radius_a,
            core_ay    = self.radius_b,
            core_angle = self.angle,
            trans_ax   = self.radius_a * cfg.transition_expand,
            trans_ay   = self.radius_b * cfg.transition_expand,
            safe_ax    = self.radius_a * cfg.safe_expand,
            safe_ay    = self.radius_b * cfg.safe_expand,
        )

    @property
    def equiv_radius(self) -> float:
        """等效半径（用于优先级和步距估算）。"""
        return np.sqrt(self.radius_a * self.radius_b)


# ─────────────────────────────────────────────────────────────────────────────
#  面积型超差 (AreaDefect)
# ─────────────────────────────────────────────────────────────────────────────

class AreaDefect(Defect):
    """
    不规则面积型超差（凸多边形轮廓）。
    contour — (N, 2) 顶点坐标 (mm)，已排序
    """

    def __init__(self, cx, cy, depth_um, contour: np.ndarray, cfg):
        self.contour = contour   # (N, 2)
        super().__init__(cx, cy, depth_um, DefectType.AREA, cfg)

    def _build_zones(self) -> DefectZones:
        cfg = self.cfg.defect
        # 用最小外接椭圆近似（以 PCA 主方向为轴）
        pts  = self.contour - self.contour.mean(axis=0)
        cov  = np.cov(pts.T)
        evals, evecs = np.linalg.eigh(cov)
        # 长轴对应最大特征值
        order  = np.argsort(evals)[::-1]
        evals  = evals[order]
        evecs  = evecs[:, order]
        angle  = np.arctan2(evecs[1, 0], evecs[0, 0])
        ax     = np.sqrt(evals[0]) * 2.0   # 2 sigma → 覆盖约 95%
        ay     = np.sqrt(evals[1]) * 2.0
        ax     = max(ax, 3.0)
        ay     = max(ay, 3.0)
        return DefectZones(
            core_cx    = self.cx,
            core_cy    = self.cy,
            core_ax    = ax,
            core_ay    = ay,
            core_angle = angle,
            trans_ax   = ax * cfg.transition_expand,
            trans_ay   = ay * cfg.transition_expand,
            safe_ax    = ax * cfg.safe_expand,
            safe_ay    = ay * cfg.safe_expand,
        )

    @property
    def area(self) -> float:
        """多边形面积（Shoelace 公式）。"""
        x, y = self.contour[:, 0], self.contour[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# ─────────────────────────────────────────────────────────────────────────────
#  合并缺陷
# ─────────────────────────────────────────────────────────────────────────────

class MergedDefect(Defect):
    """
    将安全区重叠的多个缺陷合并为一个虚拟缺陷区域处理。
    以最高优先级、最大等效半径为参数。
    """

    def __init__(self, members: List[Defect], cfg):
        self.members = members
        # 加权中心
        cx = np.mean([d.cx for d in members])
        cy = np.mean([d.cy for d in members])
        depth = max(d.depth_um for d in members)
        # 包围所有成员的最小外接椭圆
        all_cx = [d.zones.core_cx for d in members]
        all_cy = [d.zones.core_cy for d in members]
        spread_x = max(abs(cx - x) + d.zones.safe_ax
                       for d, x in zip(members, all_cx))
        spread_y = max(abs(cy - y) + d.zones.safe_ay
                       for d, y in zip(members, all_cy))
        self._spread_x = spread_x
        self._spread_y = spread_y
        super().__init__(cx, cy, depth, DefectType.AREA, cfg,
                         defect_id="merged_" + "_".join(d.id for d in members))

    def _build_zones(self) -> DefectZones:
        cfg = self.cfg.defect
        return DefectZones(
            core_cx    = self.cx,
            core_cy    = self.cy,
            core_ax    = self._spread_x,
            core_ay    = self._spread_y,
            core_angle = 0.0,
            trans_ax   = self._spread_x * cfg.transition_expand,
            trans_ay   = self._spread_y * cfg.transition_expand,
            safe_ax    = self._spread_x * cfg.safe_expand,
            safe_ay    = self._spread_y * cfg.safe_expand,
        )
