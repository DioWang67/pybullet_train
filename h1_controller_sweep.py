"""
Coarse controller sweep for H1.

Usage:
    python h1_controller_sweep.py
    python h1_controller_sweep.py --smoke
"""

from __future__ import annotations

import argparse
import itertools

from h1_controller_tools import ControllerParams, default_pose_from_cfg, evaluate_controller, load_cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a very small sweep first.")
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def value_grid(smoke: bool) -> dict[str, list[float]]:
    if smoke:
        return {
            "stand_kp": [100.0, 120.0],
            "stand_kd": [8.0],
            "pitch_kp": [45.0, 65.0],
            "pitch_kd": [8.0],
            "roll_kp": [10.0, 14.0],
            "roll_kd": [2.0],
        }
    return {
        "stand_kp": [100.0, 110.0, 120.0],
        "stand_kd": [8.0, 9.0, 10.0],
        "pitch_kp": [45.0, 55.0, 65.0],
        "pitch_kd": [6.0, 8.0],
        "roll_kp": [10.0, 12.0, 14.0],
        "roll_kd": [1.5, 2.0, 2.5],
    }


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_cfg()
    pose = default_pose_from_cfg(cfg)
    grid = value_grid(args.smoke)
    keys = list(grid)

    results = []
    total = 1
    for values in grid.values():
        total *= len(values)

    print(f"evaluating {total} controller candidates")
    for index, combo in enumerate(itertools.product(*(grid[key] for key in keys)), start=1):
        params = dict(zip(keys, combo))
        controller = ControllerParams(**params)
        result = evaluate_controller(pose=pose, controller=controller, render=False)
        results.append(result)
        print(
            f"[{index:03d}/{total:03d}]"
            f" score={result.total_score:+.3f}"
            f" stand={result.stand.score:+.2f}"
            f" shift={result.weight_shift.score:+.2f}"
            f" lift={result.lift_left.score:+.2f}"
            f" params={controller}"
        )

    results.sort(key=lambda item: item.total_score, reverse=True)
    print("")
    print(f"top {min(args.top_k, len(results))} candidates")
    for rank, result in enumerate(results[: args.top_k], start=1):
        print(
            f"{rank:02d}"
            f" total={result.total_score:+.3f}"
            f" stand={result.stand.score:+.2f}"
            f" shift={result.weight_shift.score:+.2f}"
            f" lift={result.lift_left.score:+.2f}"
            f" params={result.controller}"
        )


if __name__ == "__main__":
    main()
