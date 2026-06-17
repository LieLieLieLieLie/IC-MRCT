# IC-MRCT

**Inspection-Conditioned Multi-Region Constrained Robot Trajectory Planning for Complex Curved Surface Repair**

This repository contains the simulation code accompanying the paper. IC-MRCT is a trajectory planning framework that takes post-inspection defect maps as input and generates a complete, kinematically feasible robot repair trajectory — covering global region ordering, local coverage-path generation, and posture-continuous inter-region transitions.

---

## Overview

The framework processes three types of defects (scratches, pits, irregular area defects) on a curved surface and plans an executable trajectory through a six-stage pipeline:

1. **Inspection-conditioned input** — defect instances with morphological class, severity, and priority
2. **Three-zone region representation** — distance-field expansion into core / transition / safety zones
3. **Constraint-coupled global sequence planning** — multi-objective ordering under idle travel, posture, priority, forbidden-zone, and feasibility costs
4. **Morphology-adaptive local repair planning** — scan-line coverage optimised per defect geometry
5. **Approach, retract, and transition planning** — Bézier arcs with SLERP normal interpolation
6. **Robot trajectory synthesis** — joint-limit and collision verification

---

## Requirements

- Python ≥ 3.9
- Dependencies:

```bash
pip install -r requirements.txt
```

| Package | Version |
|---|---|
| numpy | ≥ 1.26.4 |
| matplotlib | ≥ 3.5.1 |
| openpyxl | ≥ 3.0.10 |

---

## Usage

### Quick demo (single case study scene)

```bash
python main.py --mode demo
```

Generates case study figures and metrics in `results/`.

### Baseline comparison (100 scenes)

```bash
python main.py --mode eval --n_scenes 100
```

Compares IC-MRCT against Random, Nearest Neighbor, and Priority First baselines.

### Ablation study (100 scenes)

```bash
python main.py --mode ablation --n_scenes 100
```

Evaluates 9 variants by disabling individual components.

### Full experiment

```bash
python main.py --mode all --n_scenes 100
```

### Additional options

| Flag | Description |
|---|---|
| `--no_figures` | Skip PDF figure generation (faster debugging) |
| `--verbose` | Enable detailed planner logs |
| `--timeout_hours N` | Stop gracefully after N hours and save checkpoint |
| `--resume` | Resume from the last saved checkpoint |
| `--methods` | Comma-separated subset of eval methods, e.g. `ours,nearestneighbor` |
| `--variants` | Comma-separated subset of ablation variants, e.g. `ours,nosmoothtransition` |

---

## Output

All results are written to `results/`:

```
results/
├── figures/          # PDF plots (comparison, ablation, case study)
├── tables/           # Excel summary sheets
└── models/           # Checkpoint files for resumable runs
```

---

## Repository Structure

```
├── main.py               # Entry point: demo / eval / ablation modes
├── config.py             # All hyperparameters (surface, planner, transition, dataset)
├── defect.py             # Defect classes: Scratch, Pit, AreaDefect
├── dataset.py            # Synthetic scene generator
├── global_planner.py     # Multi-objective global sequence planner
├── local_planner.py      # Morphology-adaptive local coverage planner
├── transition.py         # Bézier + SLERP transition planner
├── robot_constraints.py  # Kinematic feasibility checker
├── utils.py              # Metrics, plotting, Excel export
└── requirements.txt
```

---

## Baselines

| Method | Description |
|---|---|
| Random | Random visitation order |
| Nearest Neighbor | Greedy distance-only sequencing |
| Priority First | Sort by defect priority then position |
| **IC-MRCT (Ours)** | Constraint-coupled multi-objective planning |

## Ablation Variants

| Variant | Disabled component |
|---|---|
| No Merge | Region merging |
| Distance Only | All costs except distance |
| No Posture Cost | Posture-change penalty |
| No Priority Cost | Priority weighting |
| No Obstacle Cost | Forbidden-zone cost term |
| No Obstacle Avoidance | Transition obstacle detour |
| No Local Optimization | Morphology-adaptive local planner |
| No Smooth Transition | Bézier + SLERP transition |
