"""
dataset.py
----------
合成缺陷场景数据集生成器。

生成内容
--------
每个"场景"包含：
  - 曲面参数（随机化曲率）
  - N 个随机缺陷（类型、位置、形态、深度随机）
  - 机器人初始位置

数据集用途
----------
1. 算法评测（排序质量、覆盖率、空走比、姿态平滑度）
2. 基线方法对比（随机顺序、固定顺序）
3. 超参数调优

原数据集关联
------------
`Blade_belt_grinding_trajectory_dataset.xlsx` 中包含
local_curvature / contact_angle / normal_force 等工艺参数，
我们将其统计分布（均值/标准差）用于随机化曲面曲率和
缺陷深度的合理范围。
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from defect import Scratch, Pit, AreaDefect, Defect, DefectType


# ── 原数据集统计（离线统计得到，硬编码为常数）────────────────────────────────
_CURV_MEAN = 0.0053      # mm^-1，来自数据集 local_curvature 列均值
_CURV_STD  = 0.0021
_DEPTH_MIN = 10.0        # μm，缺陷深度范围（合理工程范围）
_DEPTH_MAX = 150.0


# ─────────────────────────────────────────────────────────────────────────────
#  单场景生成
# ─────────────────────────────────────────────────────────────────────────────

def _random_scratch(rng: np.random.Generator, cfg) -> Scratch:
    sc = cfg.surface
    cx     = rng.uniform(30, sc.length - 30)
    cy     = rng.uniform(20, sc.width  - 20)
    depth  = rng.uniform(_DEPTH_MIN, _DEPTH_MAX)
    length = rng.uniform(10, 60)    # mm
    width  = rng.uniform(1,  8)     # mm
    angle  = rng.uniform(0, np.pi)  # rad
    return Scratch(cx, cy, depth, length, width, angle, cfg)


def _random_pit(rng: np.random.Generator, cfg) -> Pit:
    sc = cfg.surface
    cx      = rng.uniform(20, sc.length - 20)
    cy      = rng.uniform(15, sc.width  - 15)
    depth   = rng.uniform(_DEPTH_MIN, _DEPTH_MAX)
    r_a     = rng.uniform(4, 20)    # 长轴半径 mm
    r_b     = rng.uniform(3, r_a)   # 短轴半径 mm（≤ 长轴）
    angle   = rng.uniform(0, np.pi)
    return Pit(cx, cy, depth, r_a, r_b, angle, cfg)


def _random_area(rng: np.random.Generator, cfg) -> AreaDefect:
    sc = cfg.surface
    cx    = rng.uniform(30, sc.length - 30)
    cy    = rng.uniform(20, sc.width  - 20)
    depth = rng.uniform(_DEPTH_MIN, _DEPTH_MAX)
    # 随机凸多边形（用极坐标方法生成）
    n_verts = rng.integers(5, 10)
    r_base  = rng.uniform(8, 25)     # 等效半径 mm
    angles  = np.sort(rng.uniform(0, 2 * np.pi, n_verts))
    rs      = r_base * rng.uniform(0.6, 1.0, n_verts)
    contour = np.column_stack([
        cx + rs * np.cos(angles),
        cy + rs * np.sin(angles),
    ])
    return AreaDefect(cx, cy, depth, contour, cfg)


def _random_obstacles(rng: np.random.Generator, cfg) -> List[Dict]:
    """Generate circular forbidden regions that emulate clamps or sensor shadows."""
    sc = cfg.surface
    lo, hi = cfg.dataset.obstacles_per_scene
    n_obs = int(rng.integers(lo, hi + 1))
    obstacles = []
    for _ in range(n_obs):
        r = float(rng.uniform(*cfg.dataset.obstacle_radius_range))
        x_margin = min(sc.length / 2 - 1.0, r + 35.0)
        y_margin = min(sc.width / 2 - 1.0, r + 25.0)
        obstacles.append({
            "cx": float(rng.uniform(x_margin, sc.length - x_margin)),
            "cy": float(rng.uniform(y_margin, sc.width - y_margin)),
            "r": r,
        })
    return obstacles


def _outside_obstacles(defect: Defect, obstacles: List[Dict], margin: float = 35.0) -> bool:
    for obs in obstacles:
        dist = np.hypot(defect.cx - obs["cx"], defect.cy - obs["cy"])
        defect_radius = max(defect.zones.safe_ax, defect.zones.safe_ay)
        if dist < obs["r"] + defect_radius + margin:
            return False
    return True


def _point_outside_obstacles(x: float, y: float, obstacles: List[Dict], margin: float = 30.0) -> bool:
    for obs in obstacles:
        if np.hypot(x - obs["cx"], y - obs["cy"]) < obs["r"] + margin:
            return False
    return True


def generate_scene(
    cfg,
    rng: np.random.Generator,
    n_defects: int,
) -> Dict:
    """
    生成单个缺陷场景。

    Returns
    -------
    dict with keys:
      defects      : List[Defect]
      robot_start  : (x, y) mm
      surface_cfg  : SurfaceConfig（随机化曲率）
      n_defects    : int
    """
    from config import SurfaceConfig
    import copy

    # 随机化曲面曲率。多数场景被设为高曲率/强姿态变化压力测试。
    surface_cfg = copy.deepcopy(cfg.surface)
    if rng.random() < cfg.dataset.high_curvature_probability:
        surface_cfg.curvature_x = float(rng.uniform(0.006, 0.018))
        surface_cfg.curvature_y = float(rng.uniform(0.008, 0.024))
    else:
        surface_cfg.curvature_x = max(0, float(rng.normal(_CURV_MEAN, _CURV_STD)))
        surface_cfg.curvature_y = max(0, float(rng.normal(_CURV_MEAN * 1.2, _CURV_STD)))

    # 临时 cfg（含新曲面参数）
    import copy as _copy
    scene_cfg = _copy.deepcopy(cfg)
    scene_cfg.surface = surface_cfg

    obstacles = _random_obstacles(rng, scene_cfg)

    # 随机缺陷类型分布
    probs  = [cfg.dataset.prob_scratch,
              cfg.dataset.prob_pit,
              cfg.dataset.prob_area]
    types  = [DefectType.SCRATCH, DefectType.PIT, DefectType.AREA]
    chosen = rng.choice(types, size=n_defects, p=probs)

    defects: List[Defect] = []
    for dtype in chosen:
        defect = None
        for _ in range(40):
            if dtype == DefectType.SCRATCH:
                defect = _random_scratch(rng, scene_cfg)
            elif dtype == DefectType.PIT:
                defect = _random_pit(rng, scene_cfg)
            else:
                defect = _random_area(rng, scene_cfg)
            if _outside_obstacles(defect, obstacles):
                break
        defects.append(defect)

    # 强化优先级时效压力：远距离散布的深缺陷需要尽早处理。
    for defect in defects:
        boundary_pressure = abs(defect.cx - surface_cfg.length / 2) / (surface_cfg.length / 2)
        if rng.random() < 0.35 + 0.35 * boundary_pressure:
            defect.depth_um = float(rng.uniform(max(defect.depth_um, 85.0), _DEPTH_MAX))
            defect.priority = defect._assign_priority()

    # 机器人起始位置（随机，模拟不同装夹）
    robot_start = None
    for _ in range(80):
        x0 = float(rng.uniform(0, 30))
        y0 = float(rng.uniform(0, cfg.surface.width))
        if _point_outside_obstacles(x0, y0, obstacles):
            robot_start = (x0, y0)
            break
    if robot_start is None:
        robot_start = (0.0, float(cfg.surface.width / 2))

    return {
        "defects":     defects,
        "robot_start": robot_start,
        "surface_cfg": surface_cfg,
        "obstacles":   obstacles,
        "n_defects":   n_defects,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  数据集生成主函数
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(cfg, verbose: bool = True) -> List[Dict]:
    """
    生成 cfg.dataset.n_scenes 个场景，保存为 .pkl。

    Returns
    -------
    List[Dict]，每个元素是一个 generate_scene 的返回值。
    """
    rng    = np.random.default_rng(cfg.dataset.random_seed)
    scenes = []
    lo, hi = cfg.dataset.defects_per_scene

    for i in range(cfg.dataset.n_scenes):
        n = int(rng.integers(lo, hi + 1))
        scene = generate_scene(cfg, rng, n)
        scenes.append(scene)
        if verbose and (i + 1) % 50 == 0:
            print(f"  生成场景 {i+1}/{cfg.dataset.n_scenes} "
                  f"(n_defects={n})")

    os.makedirs(os.path.dirname(cfg.dataset.save_path), exist_ok=True)
    with open(cfg.dataset.save_path, "wb") as f:
        pickle.dump(scenes, f)
    if verbose:
        print(f"数据集已保存: {cfg.dataset.save_path} "
              f"({len(scenes)} 场景)")
    return scenes


def load_dataset(path: str) -> List[Dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  场景统计摘要
# ─────────────────────────────────────────────────────────────────────────────

def dataset_summary(scenes: List[Dict]) -> Dict:
    n_total   = len(scenes)
    n_defects = [s["n_defects"] for s in scenes]
    types     = []
    depths    = []
    for s in scenes:
        for d in s["defects"]:
            types.append(d.type.name)
            depths.append(d.depth_um)

    from collections import Counter
    type_cnt = Counter(types)

    return {
        "n_scenes":        n_total,
        "defects_per_scene_mean": np.mean(n_defects),
        "defects_per_scene_std":  np.std(n_defects),
        "type_distribution": dict(type_cnt),
        "depth_mean_um":   np.mean(depths),
        "depth_std_um":    np.std(depths),
        "depth_min_um":    np.min(depths),
        "depth_max_um":    np.max(depths),
    }
