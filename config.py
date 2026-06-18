"""
config.py
---------
机器人打磨多缺陷修复轨迹规划框架 — 全局配置。
"""

from dataclasses import dataclass, field
from typing import Tuple


# ── 曲面与工作空间 ────────────────────────────────────────────────────────────
@dataclass
class SurfaceConfig:
    length: float = 300.0          # mm，曲面长度（X 方向）
    width:  float = 120.0          # mm，曲面宽度（Y 方向）
    # 二次曲面参数：z = a*x^2 + b*y^2（模拟叶片/车身曲面）
    curvature_x: float = 0.0003    # mm^-1
    curvature_y: float = 0.0006    # mm^-1
    grid_res:    float = 0.5       # mm，法向场采样分辨率


# ── 缺陷建模 ─────────────────────────────────────────────────────────────────
@dataclass
class DefectConfig:
    # 三区外扩比例（相对缺陷等效半径）
    core_expand:       float = 1.0   # 核心修复区 = 缺陷本身
    transition_expand: float = 1.5   # 过渡区外扩倍率
    safe_expand:       float = 2.2   # 安全/进退刀区外扩倍率

    # 缺陷严重程度阈值（深度，μm）
    priority_high:   float = 80.0   # 深度 > 此值 → 高优先级
    priority_medium: float = 30.0   # 深度 > 此值 → 中优先级


# ── 全局规划（多缺陷排序）────────────────────────────────────────────────────
@dataclass
class GlobalPlannerConfig:
    # Coupled edge-cost weights: distance, posture, priority,
    # forbidden-zone risk, and robot feasibility.
    w_distance:    float = 0.34
    w_posture:     float = 0.12
    w_priority:    float = 0.12
    w_obstacle:    float = 0.28
    w_feasibility: float = 0.14
    multi_start_route_search: bool = True

    # 合并策略：两个缺陷外扩安全区重叠 → 合并为一个
    merge_overlap_ratio: float = 0.75

    # 求解器："greedy" | "2opt" | "or_tools"
    solver: str = "2opt"
    max_iter_2opt: int = 120

    # 动态重规划：每修复完一个缺陷后重排剩余顺序
    dynamic_replan: bool = True


@dataclass
class RobotConstraintConfig:
    # Lightweight task-space robot feasibility envelope used for simulation.
    # A real deployment can replace these proxy checks with robot-specific IK,
    # joint-limit, singularity, and collision checkers.
    base_x: float = -35.0
    base_y: float = 60.0
    base_z: float = 40.0
    min_reach: float = 25.0
    max_reach: float = 360.0
    max_tool_tilt_deg: float = 42.0
    max_normal_step_deg: float = 12.0
    min_singularity_margin: float = 0.08
    tool_radius: float = 8.0
    body_radius: float = 18.0
    collision_margin: float = 6.0


# ── 局部轨迹生成 ──────────────────────────────────────────────────────────────
@dataclass
class LocalPlannerConfig:
    # Nominal steps kept only for compatibility with old scripts. The proposed
    # local planner selects spacing from the contact footprint by optimization.
    step_scratch:  float = 1.5
    step_pit:      float = 2.0
    step_area:     float = 2.5

    # 进退刀参数
    approach_dist: float = 8.0     # mm，进刀点距边界距离
    retract_dist:  float = 8.0     # mm，退刀点距边界距离
    lift_height:   float = 5.0     # mm，空走抬刀高度

    # Reference direction used only when local_optimization=False.
    # The proposed method searches trajectory direction as an optimization
    # variable around the defect principal direction and surface state.
    direction_strategy: str = "defect_axis"

    # 法向姿态插值步长 (mm)
    normal_interp_step: float = 2.0

    # Kept for old scripts; not used by the adaptive planner.
    spiral_turns: int = 3

    # Local optimization variables for morphology-adaptive coverage.
    contact_footprint_width: float = 6.0
    candidate_angle_offsets: Tuple = (-25.0, -12.0, 0.0, 12.0, 25.0, 90.0)
    candidate_spacing_factors: Tuple = (0.28, 0.38, 0.50, 0.65)
    candidate_expand_factors: Tuple = (1.10, 1.25, 1.40)
    candidate_speed_factors: Tuple = (0.85, 1.00)
    candidate_entry_offsets: Tuple = (-0.25, 0.0, 0.25)
    candidate_speed_tapers: Tuple = (0.0, 0.18)
    local_field_samples: int = 17
    local_optimization: bool = True
    # When True: optimize spacing/expansion/speed/entry but fix direction to
    # the defect principal axis (offset_deg=0 only). Isolates the direction
    # search contribution from the rest of local optimization.
    fixed_direction_only: bool = False


# ── 过渡路径 ──────────────────────────────────────────────────────────────────
@dataclass
class TransitionConfig:
    # 贝塞尔过渡段控制点距离（相对两缺陷距离的比例）
    bezier_ctrl_ratio: float = 0.35
    # 姿态插值点数（过渡段上均匀插值）
    posture_interp_n:  int   = 20
    # 最大允许姿态跳变角度 (deg)，超过则插入额外过渡点
    max_posture_jump:  float = 15.0
    # 是否使用贝塞尔路径和 SLERP 法向插值进行平滑过渡
    smooth_transition: bool = True
    # Waypoint-level detours around circular forbidden zones.
    obstacle_avoidance: bool = True
    obstacle_margin: float = 8.0
    obstacle_max_insertions: int = 8
    obstacle_repulsion_iters: int = 2


# ── 仿真数据集 ────────────────────────────────────────────────────────────────
@dataclass
class DatasetConfig:
    n_scenes:          int   = 1000      # 生成场景数
    defects_per_scene: Tuple = (8, 24)   # 每场景缺陷数范围
    random_seed:       int   = 42
    save_path:         str   = "dataset/repair_dataset.pkl"

    # 缺陷类型分布概率
    prob_scratch:  float = 0.35
    prob_pit:      float = 0.40
    prob_area:     float = 0.25

    # 压力测试场景：障碍/禁入区与高曲率姿态变化
    obstacles_per_scene: Tuple = (3, 7)
    obstacle_radius_range: Tuple = (12.0, 28.0)
    high_curvature_probability: float = 0.80


# ── 实验 ──────────────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    results_dir: str  = "results"
    seed:        int  = 42
    verbose:     bool = True


# ── 主配置 ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    surface:    SurfaceConfig    = field(default_factory=SurfaceConfig)
    defect:     DefectConfig     = field(default_factory=DefectConfig)
    global_p:   GlobalPlannerConfig = field(default_factory=GlobalPlannerConfig)
    robot:      RobotConstraintConfig = field(default_factory=RobotConstraintConfig)
    local_p:    LocalPlannerConfig  = field(default_factory=LocalPlannerConfig)
    transition: TransitionConfig    = field(default_factory=TransitionConfig)
    dataset:    DatasetConfig       = field(default_factory=DatasetConfig)
    experiment: ExperimentConfig    = field(default_factory=ExperimentConfig)


def get_config() -> Config:
    return Config()
