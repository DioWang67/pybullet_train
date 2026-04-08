"""
Scan H1 stand poses for balance and lateral-shift stability.

Usage:
    python h1_stand_pose_scan.py
"""

from __future__ import annotations

import itertools
from copy import deepcopy

import numpy as np

from config import ConfigManager
from envs.h1_env import H1Env
from simulators import PyBulletRobotSimulator


def make_cfg(hip_pitch: float, knee: float, ankle: float, hip_roll_bias: float):
    cfg = ConfigManager().load_robot_config("h1")
    cfg = deepcopy(cfg)
    pose = dict(cfg.stand_pose)
    pose[1] = hip_roll_bias
    pose[6] = hip_roll_bias
    pose[2] = hip_pitch
    pose[3] = knee
    pose[4] = ankle
    pose[7] = hip_pitch
    pose[8] = knee
    pose[9] = ankle
    cfg.stand_pose = pose
    return cfg


def balance_score(cfg) -> tuple[float, float, float, bool]:
    env = H1Env(robot_config=cfg)
    env.reset()
    survived = True
    for _ in range(int(cfg.physics.policy_hz)):
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, terminated, _, _ = env.step(zero_action)
        if terminated:
            survived = False
            break
    pos = env.robot_simulator.get_base_position()
    euler = env.robot_simulator.get_base_orientation_euler()
    env.close()
    return float(pos[2]), float(euler[1]), float(euler[0]), survived


def init_sim(cfg) -> PyBulletRobotSimulator:
    sim = PyBulletRobotSimulator(
        robot_description_name=cfg.robot_description_name or "h1_description",
        physics_hz=cfg.physics.physics_hz,
        render=False,
        gravity=cfg.physics.gravity,
        num_solver_iterations=cfg.physics.num_solver_iterations,
    )
    sim.connect()
    sim.reset(
        np.array([0.0, 0.0, cfg.spawn_z], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    sim.reset_joint_state(cfg.stand_pose)
    sim.disable_default_motors()
    sim.lock_joints(cfg.lock_joints)
    sim.set_dynamics(cfg.feet_link_indices["left"], lateral_friction=cfg.foot_friction, restitution=0.0)
    sim.set_dynamics(cfg.feet_link_indices["right"], lateral_friction=cfg.foot_friction, restitution=0.0)
    for j in range(sim.get_num_joints()):
        name = sim.get_joint_name(j)
        if name.endswith(("imu_link", "logo_link", "imager_link", "rgb_module_link", "mid360_link")):
            sim.set_dynamics(j, mass=0.0)
    return sim


def weight_shift_score(cfg) -> tuple[float, float, float, bool]:
    sim = init_sim(cfg)
    active = cfg.active_joints
    stand = np.array([cfg.stand_pose.get(j, 0.0) for j in active], dtype=np.float32)
    max_torque = np.array(cfg.max_torque, dtype=np.float32)

    shift = stand.copy()
    shift[1] += -0.04
    shift[6] += -0.04
    shift[4] += 0.08
    shift[9] += 0.08

    kp = 180.0
    kd = 12.0
    survived = True
    steps = int(2.0 * cfg.physics.physics_hz)
    for i in range(steps):
        blend = min(1.0, i / max(1, int(0.8 * cfg.physics.physics_hz)))
        target = stand + blend * (shift - stand)
        joint_pos = sim.get_joint_positions(active)
        joint_vel = sim.get_joint_velocities(active)
        torques = np.clip(kp * (target - joint_pos) - kd * joint_vel, -max_torque, max_torque)
        sim.set_joint_motor_control(active, torques)
        sim.step()
        pos = sim.get_base_position()
        euler = sim.get_base_orientation_euler()
        if pos[2] < cfg.torso_min_z or abs(float(euler[0])) > 0.5 or abs(float(euler[1])) > 0.5:
            survived = False
            break

    pos = sim.get_base_position()
    euler = sim.get_base_orientation_euler()
    sim.disconnect()
    return float(pos[2]), float(euler[1]), float(euler[0]), survived


def main() -> None:
    hip_pitch_vals = [0.24, 0.30, 0.36]
    knee_vals = [-0.35, -0.50, -0.65]
    ankle_vals = [0.10, 0.20, 0.30]
    hip_roll_bias_vals = [0.0, -0.04]

    results = []
    for hip_pitch, knee, ankle, hip_roll_bias in itertools.product(
        hip_pitch_vals, knee_vals, ankle_vals, hip_roll_bias_vals
    ):
        cfg = make_cfg(hip_pitch, knee, ankle, hip_roll_bias)
        bal_z, bal_pitch, bal_roll, bal_ok = balance_score(cfg)
        shift_z, shift_pitch, shift_roll, shift_ok = weight_shift_score(cfg)
        score = (
            (2.0 if bal_ok else 0.0)
            + (2.0 if shift_ok else 0.0)
            + bal_z
            + shift_z
            - abs(bal_pitch) - abs(bal_roll)
            - abs(shift_pitch) - abs(shift_roll)
        )
        results.append(
            {
                "hip_pitch": hip_pitch,
                "knee": knee,
                "ankle": ankle,
                "hip_roll_bias": hip_roll_bias,
                "bal_ok": bal_ok,
                "shift_ok": shift_ok,
                "bal_z": bal_z,
                "shift_z": shift_z,
                "bal_pitch": bal_pitch,
                "shift_pitch": shift_pitch,
                "score": score,
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    for r in results[:12]:
        print(
            f"score={r['score']:.3f}"
            f" hip_pitch={r['hip_pitch']:.2f}"
            f" knee={r['knee']:.2f}"
            f" ankle={r['ankle']:.2f}"
            f" hip_roll_bias={r['hip_roll_bias']:+.2f}"
            f" bal_ok={int(r['bal_ok'])}"
            f" shift_ok={int(r['shift_ok'])}"
            f" bal_z={r['bal_z']:.3f}"
            f" shift_z={r['shift_z']:.3f}"
            f" bal_pitch={r['bal_pitch']:+.3f}"
            f" shift_pitch={r['shift_pitch']:+.3f}"
        )


if __name__ == "__main__":
    main()
