"""
main.py
-------
Robot grinding trajectory planning demo and evaluation.

The current study focuses on online adaptive trajectory planning for
multiple local abnormal regions on a complex surface.  The program
therefore evaluates global repair ordering, local coverage trajectory
generation, approach/retract motions, and posture-continuous transitions.
It intentionally does not train or invoke a material-removal prediction
network.
"""

import argparse
import copy
import logging
import os
import pickle
import time

import numpy as np

from config import get_config
from defect import AreaDefect, Pit, Scratch
from global_planner import GlobalPlanner, build_cost_matrix, greedy_tour, merge_overlapping
from local_planner import LocalPlanner
from transition import TransitionPlanner
from dataset import generate_scene
from utils import (
    compute_metrics,
    format_metrics,
    plot_comparison,
    plot_full_path,
    plot_path_density_heatmap,
    plot_performance_landscape,
    plot_posture_continuity,
    plot_scene_overview,
    save_experiment_xlsx,
    save_summary_xlsx,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")
logging.getLogger("fontTools").setLevel(logging.WARNING)


METHODS = {
    "Random": "random",
    "Nearest Neighbor": "nearest",
    "Priority First": "priority",
    "Ours": "ours",
}

ABLATIONS = {
    "Ours": {},
    "No Merge": {"merge_overlap_ratio": 1.0},
    "Distance Only": {"w_distance": 1.0, "w_posture": 0.0, "w_priority": 0.0, "w_obstacle": 0.0, "w_feasibility": 0.0, "multi_start_route_search": False},
    "No Posture Cost": {"w_posture": 0.0, "multi_start_route_search": False},
    "No Priority Cost": {"w_priority": 0.0, "multi_start_route_search": False},
    "No Obstacle Cost": {"w_obstacle": 0.0, "multi_start_route_search": False},
    "No Obstacle Avoidance": {"w_obstacle": 0.0, "obstacle_avoidance": False, "multi_start_route_search": False},
    "No Local Optimization": {"local_optimization": False},
    "No Smooth Transition": {"smooth_transition": False},
}


def _variant_config(cfg, updates):
    variant = copy.deepcopy(cfg)
    for key, value in updates.items():
        if hasattr(variant.global_p, key):
            setattr(variant.global_p, key, value)
        elif hasattr(variant.transition, key):
            setattr(variant.transition, key, value)
        elif hasattr(variant.local_p, key):
            setattr(variant.local_p, key, value)
        else:
            raise KeyError(f"Unknown variant option: {key}")
    return variant


def _resolve_results_dir(cfg):
    """Make results_dir stable relative to this script, not the shell cwd."""
    if not os.path.isabs(cfg.experiment.results_dir):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cfg.experiment.results_dir = os.path.join(base_dir, cfg.experiment.results_dir)
    return cfg.experiment.results_dir


def _order_defects(scene, cfg, method):
    defects = scene["defects"]
    robot_start = scene["robot_start"]
    obstacles = scene.get("obstacles", [])

    if method == "ours":
        return GlobalPlanner(cfg).plan(defects, robot_pos=robot_start, obstacles=obstacles)

    merged = merge_overlapping(defects, cfg)
    if not merged:
        return []

    if method == "random":
        ordered = list(merged)
        np.random.default_rng(cfg.experiment.seed).shuffle(ordered)
        return ordered

    if method == "nearest":
        cost, _, _, _, _ = build_cost_matrix(
            merged,
            cfg.surface,
            w_distance=1.0,
            w_posture=0.0,
            w_priority=0.0,
            w_obstacle=0.0,
            robot_pos=robot_start,
        )
        tour = greedy_tour(cost)
        return [merged[i - 1] for i in tour[1:]]

    if method == "priority":
        return sorted(merged, key=lambda d: (-int(d.priority), d.cx, d.cy))

    raise ValueError(f"Unknown method: {method}")


def run_scene_method(scene, cfg, method):
    run_cfg = copy.deepcopy(cfg)
    if "surface_cfg" in scene:
        run_cfg.surface = copy.deepcopy(scene["surface_cfg"])

    if method != "ours":
        run_cfg.transition.obstacle_avoidance = False
        run_cfg.transition.smooth_transition = False
        run_cfg.local_p.local_optimization = False
        run_cfg.global_p.multi_start_route_search = False

    ordered = _order_defects(scene, run_cfg, method)
    if not ordered:
        return {}

    use_full_selector = (
        method == "ours"
        and getattr(run_cfg.global_p, "multi_start_route_search", False)
        and getattr(run_cfg.local_p, "local_optimization", False)
        and getattr(run_cfg.transition, "smooth_transition", False)
        and getattr(run_cfg.transition, "obstacle_avoidance", False)
    )

    if not use_full_selector:
        return _evaluate_order(scene, run_cfg, ordered)

    candidate_orders = [ordered]
    for candidate_method in ("nearest", "priority"):
        try:
            candidate_orders.append(_order_defects(scene, run_cfg, candidate_method))
        except Exception:
            pass

    best = None
    local_cache = {}
    seen = set()
    for candidate_order in candidate_orders:
        signature = tuple(getattr(d, "id", str(i)) for i, d in enumerate(candidate_order))
        if not candidate_order or signature in seen:
            continue
        seen.add(signature)
        for avoid in (True, False):
            candidate_cfg = copy.deepcopy(run_cfg)
            candidate_cfg.transition.obstacle_avoidance = avoid
            metrics = _evaluate_order(scene, candidate_cfg, candidate_order, local_cache=local_cache)
            if metrics and (best is None or metrics.get("planning_objective", np.inf) < best.get("planning_objective", np.inf)):
                best = metrics
    return best or {}


def _evaluate_order(scene, cfg, ordered, local_cache=None):
    local_planner = LocalPlanner(cfg)
    local_trajs = []
    for defect in ordered:
        key = getattr(defect, "id", id(defect))
        if local_cache is not None and key in local_cache:
            local_trajs.append(local_cache[key])
            continue
        traj = local_planner.plan(defect)
        if local_cache is not None:
            local_cache[key] = traj
        local_trajs.append(traj)
    full_path, stats = TransitionPlanner(cfg).assemble(
        ordered,
        local_trajs,
        robot_start=scene["robot_start"],
        obstacles=scene.get("obstacles", []),
    )
    return compute_metrics(full_path, ordered, local_trajs, stats, obstacles=scene.get("obstacles", []), cfg=cfg)


def _demo_defects(cfg, n_defects):
    defects = [
        Scratch(50, 30, 95, 40, 4, np.radians(30), cfg),
        Scratch(230, 80, 45, 25, 3, np.radians(120), cfg),
        Pit(130, 60, 120, 15, 10, np.radians(45), cfg),
        Pit(200, 40, 35, 8, 6, 0, cfg),
        AreaDefect(
            80,
            85,
            75,
            np.array(
                [[62, 75], [75, 68], [95, 72], [100, 90], [88, 100], [70, 98], [60, 88]],
                float,
            ),
            cfg,
        ),
        AreaDefect(
            270,
            55,
            55,
            np.array([[255, 44], [270, 40], [285, 48], [288, 62], [275, 70], [258, 65]], float),
            cfg,
        ),
    ]
    return defects[:n_defects]


def run_demo(cfg, n_defects=6):
    figs, tables, _ = _subdirs(cfg.experiment.results_dir)
    scene = {
        "defects": _demo_defects(cfg, n_defects),
        "robot_start": (0.0, cfg.surface.width / 2),
        "surface_cfg": cfg.surface,
        "obstacles": [
            {"cx": 112.0, "cy": 58.0, "r": 15.0},
            {"cx": 178.0, "cy": 73.0, "r": 18.0},
            {"cx": 246.0, "cy": 42.0, "r": 13.0},
        ],
        "n_defects": n_defects,
    }

    ordered = _order_defects(scene, cfg, "ours")
    local_planner = LocalPlanner(cfg)
    local_trajs = [local_planner.plan(defect) for defect in ordered]
    full_path, stats = TransitionPlanner(cfg).assemble(
        ordered,
        local_trajs,
        robot_start=scene["robot_start"],
        obstacles=scene.get("obstacles", []),
    )
    metrics = compute_metrics(full_path, ordered, local_trajs, stats, obstacles=scene.get("obstacles", []), cfg=cfg)

    print(format_metrics(metrics, "Ours - Case Study"))
    save_summary_xlsx({"Ours Case Study": metrics}, os.path.join(tables, "case_study_metrics.xlsx"))
    plot_scene_overview(scene["defects"], ordered, cfg, os.path.join(figs, "case_study_overview.pdf"),
                        obstacles=scene.get("obstacles", []))
    plot_full_path(full_path, ordered, cfg, os.path.join(figs, "case_study_full_path.pdf"),
                   obstacles=scene.get("obstacles", []))
    plot_path_density_heatmap(full_path, cfg, os.path.join(figs, "case_study_path_density.pdf"))
    plot_posture_continuity(full_path, os.path.join(figs, "case_study_posture.pdf"))
    logger.info("Demo completed -> figures: %s  tables: %s", figs, tables)
    return metrics


def _parse_name_filter(raw, available):
    if not raw or raw.lower() == "all":
        return list(available)
    lookup = {name.lower().replace(" ", "").replace("_", ""): name for name in available}
    selected = []
    for item in raw.split(","):
        key = item.strip().lower().replace(" ", "").replace("_", "")
        if key not in lookup:
            raise ValueError(f"Unknown name '{item}'. Available: {', '.join(available)}")
        selected.append(lookup[key])
    return selected


_CKPT_FREQ = 10  # save checkpoint every N scenes


def _subdirs(results_dir: str):
    """Return (figures_dir, tables_dir, models_dir) and create them."""
    figs   = os.path.join(results_dir, "figures")
    tables = os.path.join(results_dir, "tables")
    models = os.path.join(results_dir, "models")
    for d in (figs, tables, models):
        os.makedirs(d, exist_ok=True)
    return figs, tables, models


def _save_checkpoint(path: str, i_done: int, all_results: dict, records: list, rng):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "i_done": i_done,
            "all_results": all_results,
            "records": records,
            "rng_state": rng.bit_generator.state,
        }, f)


def _load_checkpoint(path: str, all_results_template: dict):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    for name in all_results_template:
        if name not in ckpt["all_results"]:
            ckpt["all_results"][name] = []
    return ckpt


def _print_progress(i: int, i_start: int, n_scenes: int, t0: float, label: str = "Progress"):
    elapsed = (time.time() - t0) / 60
    done = i - i_start + 1
    remaining = n_scenes - i - 1
    eta = (elapsed / done * remaining) if done > 0 else 0
    print(f"  [{label}] {i+1}/{n_scenes}  elapsed {elapsed:.0f}min  ETA {eta:.0f}min", flush=True)


def run_eval(cfg, n_scenes=100, methods=None, make_figures=True,
             checkpoint_path=None, resume=False, timeout_sec=None):
    figs, tables, _ = _subdirs(cfg.experiment.results_dir)
    method_names = methods or list(METHODS)
    all_results = {name: [] for name in method_names}
    records = []
    i_start = 0
    rng = np.random.default_rng(cfg.experiment.seed)

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = _load_checkpoint(checkpoint_path, all_results)
        i_start = ckpt["i_done"]
        all_results = ckpt["all_results"]
        records = ckpt["records"]
        rng.bit_generator.state = ckpt["rng_state"]
        print(f"[Checkpoint] Resumed eval from scene {i_start}/{n_scenes}")

    t0 = time.time()
    for i in range(i_start, n_scenes):
        if timeout_sec is not None and (time.time() - t0) > timeout_sec:
            print(f"[Timeout] {timeout_sec/3600:.1f}h limit reached at eval scene {i}/{n_scenes} — saving checkpoint.")
            if checkpoint_path:
                _save_checkpoint(checkpoint_path, i, all_results, records, rng)
            break

        lo, hi = cfg.dataset.defects_per_scene
        n_defects = int(rng.integers(lo, hi + 1))
        scene = generate_scene(cfg, rng, n_defects)

        for name in method_names:
            method = METHODS[name]
            try:
                metrics = run_scene_method(scene, cfg, method)
                if metrics:
                    all_results[name].append(metrics)
                    records.append({"scene_id": i, "method": name, **metrics})
            except Exception as exc:
                logger.warning("Scene %d [%s] failed: %s", i, name, exc)

        if (i + 1) % 20 == 0:
            _print_progress(i, i_start, n_scenes, t0, "Eval")

        if checkpoint_path and (i + 1) % _CKPT_FREQ == 0:
            _save_checkpoint(checkpoint_path, i + 1, all_results, records, rng)

    summary = _aggregate_results(all_results)
    summary_path = os.path.join(tables, "eval_summary.xlsx")
    save_experiment_xlsx(summary, records, summary_path)

    keys = ["idle_ratio", "priority_satisfaction", "safe_transition_score", "planning_objective"]
    print("\n" + "=" * 78)
    print(f"  {'Method':<24}  {'idle_ratio':>10}  {'coverage':>10}  {'posture_deg':>12}  {'jumps':>8}")
    print("-" * 78)
    for name, aggregate in summary.items():
        tag = " <-" if name == "Ours" else ""
        print(
            f"  {name:<24}  {aggregate.get('idle_ratio', 0):>10.4f}  "
            f"{aggregate.get('coverage_mean', 0):>10.4f}  "
            f"{aggregate.get('posture_change_mean_deg', 0):>12.2f}  "
            f"{aggregate.get('n_posture_jumps', 0):>8.1f}{tag}"
        )
    print("=" * 78)

    if make_figures:
        plot_comparison(summary, os.path.join(figs, "comparison_core.pdf"), metrics_to_plot=keys)
        plot_performance_landscape(summary, records, os.path.join(figs, "comparison_landscape.pdf"))
    logger.info("Evaluation completed -> tables: %s  figures: %s", tables, figs)
    return summary


def run_ablation(cfg, n_scenes=100, variants=None, make_figures=True,
                 checkpoint_path=None, resume=False, timeout_sec=None):
    figs, tables, _ = _subdirs(cfg.experiment.results_dir)
    variant_names = variants or list(ABLATIONS)
    all_results = {name: [] for name in variant_names}
    records = []
    i_start = 0
    rng = np.random.default_rng(cfg.experiment.seed)

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = _load_checkpoint(checkpoint_path, all_results)
        i_start = ckpt["i_done"]
        all_results = ckpt["all_results"]
        records = ckpt["records"]
        rng.bit_generator.state = ckpt["rng_state"]
        print(f"[Checkpoint] Resumed ablation from scene {i_start}/{n_scenes}")

    t0 = time.time()
    for i in range(i_start, n_scenes):
        if timeout_sec is not None and (time.time() - t0) > timeout_sec:
            print(f"[Timeout] {timeout_sec/3600:.1f}h limit reached at ablation scene {i}/{n_scenes} — saving checkpoint.")
            if checkpoint_path:
                _save_checkpoint(checkpoint_path, i, all_results, records, rng)
            break

        lo, hi = cfg.dataset.defects_per_scene
        n_defects = int(rng.integers(lo, hi + 1))
        scene = generate_scene(cfg, rng, n_defects)

        for name in variant_names:
            updates = ABLATIONS[name]
            variant_cfg = _variant_config(cfg, updates)
            try:
                metrics = run_scene_method(scene, variant_cfg, "ours")
                if metrics:
                    all_results[name].append(metrics)
                    records.append({"scene_id": i, "method": name, **metrics})
            except Exception as exc:
                logger.warning("Ablation scene %d [%s] failed: %s", i, name, exc)

        if (i + 1) % 20 == 0:
            _print_progress(i, i_start, n_scenes, t0, "Ablation")

        if checkpoint_path and (i + 1) % _CKPT_FREQ == 0:
            _save_checkpoint(checkpoint_path, i + 1, all_results, records, rng)

    summary = _aggregate_results(all_results)
    summary_path = os.path.join(tables, "ablation_summary.xlsx")
    save_experiment_xlsx(summary, records, summary_path)

    keys = ["idle_ratio", "safe_transition_score", "posture_change_max_deg", "planning_objective"]
    print("\n" + "=" * 88)
    print(f"  {'Variant':<24}  {'idle_ratio':>10}  {'coverage':>10}  {'max_posture':>12}  {'jumps':>8}")
    print("-" * 88)
    for name, aggregate in summary.items():
        tag = " <-" if name == "Ours" else ""
        print(
            f"  {name:<24}  {aggregate.get('idle_ratio', 0):>10.4f}  "
            f"{aggregate.get('coverage_mean', 0):>10.4f}  "
            f"{aggregate.get('posture_change_max_deg', 0):>12.2f}  "
            f"{aggregate.get('n_posture_jumps', 0):>8.1f}{tag}"
        )
    print("=" * 88)

    if make_figures:
        plot_comparison(summary, os.path.join(figs, "ablation_core.pdf"), metrics_to_plot=keys)
        plot_performance_landscape(summary, records, os.path.join(figs, "ablation_landscape.pdf"))
    logger.info("Ablation completed -> tables: %s  figures: %s", tables, figs)
    return summary


def _aggregate_results(all_results):
    summary = {}
    for name, metrics_list in all_results.items():
        if not metrics_list:
            continue
        aggregate = {}
        keys = sorted({key for metrics in metrics_list for key in metrics})
        for key in keys:
            vals = [metrics[key] for metrics in metrics_list if key in metrics]
            aggregate[key] = float(np.mean(vals))
            aggregate[key + "_std"] = float(np.std(vals))
        summary[name] = aggregate
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "eval", "ablation", "all"], default="all")
    parser.add_argument("--n_scenes", type=int, default=100)
    parser.add_argument("--n_defects", type=int, default=6)
    parser.add_argument("--methods", default="all",
                        help="Comma-separated eval methods, e.g. ours or ours,nearestneighbor.")
    parser.add_argument("--variants", default="all",
                        help="Comma-separated ablation variants, e.g. ours,nosmoothtransition.")
    parser.add_argument("--no_figures", action="store_true",
                        help="Skip PDF generation for fast debugging runs.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable detailed planner logs.")
    parser.add_argument("--timeout_hours", type=float, default=0,
                        help="Optional hard time limit in hours (0 = no limit). Saves checkpoint and stops gracefully.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint saved in results/models/.")
    args = parser.parse_args()

    config = get_config()
    _resolve_results_dir(config)
    results_dir = config.experiment.results_dir
    _, _, models_dir = _subdirs(results_dir)

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    selected_methods = _parse_name_filter(args.methods, METHODS)
    selected_variants = _parse_name_filter(args.variants, ABLATIONS)
    timeout_sec = args.timeout_hours * 3600 if args.timeout_hours > 0 else None

    if args.mode in ("demo", "all"):
        run_demo(config, args.n_defects)

    if args.mode in ("eval", "all"):
        run_eval(
            config, args.n_scenes,
            methods=selected_methods,
            make_figures=not args.no_figures,
            checkpoint_path=os.path.join(models_dir, "ckpt_eval.pkl"),
            resume=args.resume,
            timeout_sec=timeout_sec,
        )

    if args.mode in ("ablation", "all"):
        run_ablation(
            config, args.n_scenes,
            variants=selected_variants,
            make_figures=not args.no_figures,
            checkpoint_path=os.path.join(models_dir, "ckpt_ablation.pkl"),
            resume=args.resume,
            timeout_sec=timeout_sec,
        )
