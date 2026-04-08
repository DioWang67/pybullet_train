"""
Evaluate the current H1 low-level controller in DIRECT mode.

Usage:
    python h1_controller_eval.py
    python h1_controller_eval.py --json
    python h1_controller_eval.py --render
"""

from __future__ import annotations

import argparse
import json

from h1_controller_tools import ControllerParams, evaluate_controller, result_to_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--stand-kp", type=float, default=ControllerParams.stand_kp)
    parser.add_argument("--stand-kd", type=float, default=ControllerParams.stand_kd)
    parser.add_argument("--pitch-kp", type=float, default=ControllerParams.pitch_kp)
    parser.add_argument("--pitch-kd", type=float, default=ControllerParams.pitch_kd)
    parser.add_argument("--roll-kp", type=float, default=ControllerParams.roll_kp)
    parser.add_argument("--roll-kd", type=float, default=ControllerParams.roll_kd)
    return parser


def print_task(name: str, metrics) -> None:
    print(
        f"{name:<12}"
        f" score={metrics.score:+.3f}"
        f" survived={int(metrics.survived)}"
        f" time={metrics.elapsed:.2f}/{metrics.duration:.2f}s"
        f" z={metrics.final_z:.3f}"
        f" max_pitch={metrics.max_abs_pitch:.3f}"
        f" max_roll={metrics.max_abs_roll:.3f}"
        f" y_span={metrics.y_span:.3f}"
        f" clearance={metrics.swing_clearance:.3f}"
        f" single={metrics.single_support_ratio:.2f}"
        f" expected={metrics.expected_support_ratio:.2f}"
    )


def main() -> None:
    args = build_parser().parse_args()
    controller = ControllerParams(
        stand_kp=args.stand_kp,
        stand_kd=args.stand_kd,
        pitch_kp=args.pitch_kp,
        pitch_kd=args.pitch_kd,
        roll_kp=args.roll_kp,
        roll_kd=args.roll_kd,
    )
    result = evaluate_controller(controller=controller, render=args.render)

    if args.json:
        print(json.dumps(result_to_dict(result), indent=2))
        return

    print("pose:", result.pose)
    print("controller:", result.controller)
    print_task("stand", result.stand)
    print_task("weight_shift", result.weight_shift)
    print_task("lift_left", result.lift_left)
    print(f"total_score={result.total_score:+.3f}")


if __name__ == "__main__":
    main()
