"""
Quick motion debug viewer for Unitree H1.

Usage:
    python h1_motion_debug.py --mode stand
    python h1_motion_debug.py --mode squat
    python h1_motion_debug.py --mode march
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pybullet as p

from config import ConfigManager
from simulators import PyBulletRobotSimulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["stand", "squat", "weight_shift", "lift_left", "lift_right", "march_soft"],
        default="stand",
    )
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--speed", type=float, default=1.0)
    return parser


def make_pose(mode: str, stand: np.ndarray, t: float, speed: float) -> np.ndarray:
    pose = stand.copy()
    phase = speed * t

    if mode == "stand":
        return pose

    if mode == "squat":
        wave = 0.5 * (1.0 - math.cos(2.0 * math.pi * 0.35 * phase))
        pose[2] += 0.12 * wave
        pose[3] -= 0.45 * wave
        pose[4] += 0.26 * wave
        pose[7] += 0.12 * wave
        pose[8] -= 0.45 * wave
        pose[9] += 0.26 * wave
        return pose

    shift_left = stand.copy()
    shift_left[1] += -0.04
    shift_left[6] += -0.04
    shift_left[4] += 0.08
    shift_left[9] += 0.08

    shift_right = stand.copy()
    shift_right[1] += -0.12
    shift_right[6] += -0.12

    if mode == "weight_shift":
        wave = 0.5 * (1.0 - math.cos(2.0 * math.pi * 0.20 * phase))
        if math.sin(2.0 * math.pi * 0.20 * phase) >= 0.0:
            return stand + wave * (shift_left - stand)
        return stand + wave * (shift_right - stand)

    if mode in {"lift_left", "lift_right"}:
        lift_side = "left" if mode == "lift_left" else "right"
        cycle = 0.5 * (1.0 - math.cos(2.0 * math.pi * 0.18 * phase))
        pose = shift_right.copy() if lift_side == "left" else shift_left.copy()

        if lift_side == "left":
            pose[2] += -0.03 * cycle
            pose[3] += 0.10 * cycle
            pose[4] += -0.05 * cycle
            pose[7] += 0.03 * cycle
            pose[8] -= 0.08 * cycle
            pose[9] += 0.05 * cycle
        else:
            pose[7] += -0.03 * cycle
            pose[8] += 0.10 * cycle
            pose[9] += -0.05 * cycle
            pose[2] += 0.03 * cycle
            pose[3] -= 0.08 * cycle
            pose[4] += 0.05 * cycle
        return pose

    swing = math.sin(2.0 * math.pi * 0.22 * phase)
    left = max(0.0, swing)
    right = max(0.0, -swing)

    pose = shift_right.copy() if left > 0.0 else shift_left.copy()

    # Swing leg: very small lift first, prioritize keeping support.
    pose[2] += -0.03 * left
    pose[3] += 0.10 * left
    pose[4] += -0.05 * left
    pose[7] += -0.03 * right
    pose[8] += 0.10 * right
    pose[9] += -0.05 * right

    pose[2] += 0.03 * right
    pose[3] -= 0.08 * right
    pose[4] += 0.05 * right
    pose[7] += 0.03 * left
    pose[8] -= 0.08 * left
    pose[9] += 0.05 * left
    return pose


def init_sim(cfg) -> PyBulletRobotSimulator:
    sim = PyBulletRobotSimulator(
        robot_description_name=cfg.robot_description_name or "h1_description",
        physics_hz=cfg.physics.physics_hz,
        render=True,
        gravity=cfg.physics.gravity,
        num_solver_iterations=cfg.physics.num_solver_iterations,
    )
    sim.connect()
    sim.reset(
        np.array([0.0, 0.0, cfg.spawn_z], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )

    sim.set_dynamics(cfg.feet_link_indices["left"], lateral_friction=cfg.foot_friction, restitution=0.0)
    sim.set_dynamics(cfg.feet_link_indices["right"], lateral_friction=cfg.foot_friction, restitution=0.0)
    sim.reset_joint_state(cfg.stand_pose)
    sim.disable_default_motors()
    sim.lock_joints(cfg.lock_joints)

    for j in range(sim.get_num_joints()):
        name = sim.get_joint_name(j)
        if name.endswith(("imu_link", "logo_link", "imager_link", "rgb_module_link", "mid360_link")):
            sim.set_dynamics(j, mass=0.0)

    p.resetDebugVisualizerCamera(
        cameraDistance=2.3,
        cameraYaw=35,
        cameraPitch=-12,
        cameraTargetPosition=[0.0, 0.0, 0.95],
        physicsClientId=sim.get_client_id(),
    )
    return sim


def print_status(sim: PyBulletRobotSimulator, cfg, target: np.ndarray, elapsed: float) -> None:
    pos = sim.get_base_position()
    euler = sim.get_base_orientation_euler()
    contacts = sim.get_foot_contact(cfg.feet_link_indices)
    joint_pos = sim.get_joint_positions(cfg.active_joints)
    print(
        f"t={elapsed:5.2f}s"
        f"  z={pos[2]:.3f}"
        f"  pitch={euler[1]:+.3f}"
        f"  roll={euler[0]:+.3f}"
        f"  lknee={joint_pos[3]:+.3f}"
        f"  rknee={joint_pos[8]:+.3f}"
        f"  l_contact={int(contacts['left'])}"
        f"  r_contact={int(contacts['right'])}"
        f"  target_lknee={target[3]:+.3f}"
        f"  target_rknee={target[8]:+.3f}"
    )


def main() -> None:
    args = build_parser().parse_args()
    cfg = ConfigManager().load_robot_config("h1")
    sim = init_sim(cfg)

    active = cfg.active_joints
    stand = np.array([cfg.stand_pose.get(j, 0.0) for j in active], dtype=np.float32)
    max_torque = np.array(cfg.max_torque, dtype=np.float32)
    kp = 220.0
    kd = 16.0

    start = time.time()
    last_print = -1.0

    try:
        while True:
            elapsed = time.time() - start
            if elapsed >= args.duration:
                break

            target = make_pose(args.mode, stand, elapsed, args.speed)
            for _ in range(cfg.physics.substeps):
                joint_pos = sim.get_joint_positions(active)
                joint_vel = sim.get_joint_velocities(active)
                torques = np.clip(kp * (target - joint_pos) - kd * joint_vel, -max_torque, max_torque)
                sim.set_joint_motor_control(active, torques)
                sim.step()

            if elapsed - last_print >= 0.25:
                last_print = elapsed
                print_status(sim, cfg, target, elapsed)

            time.sleep(1.0 / cfg.physics.policy_hz)
    finally:
        sim.disconnect()


if __name__ == "__main__":
    main()
