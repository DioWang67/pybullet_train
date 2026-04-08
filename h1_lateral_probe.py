"""
Probe lateral balance directions for H1.

Usage:
    python h1_lateral_probe.py
"""

from __future__ import annotations

import itertools
import numpy as np

from config import ConfigManager
from simulators import PyBulletRobotSimulator


def run_probe(hip_roll_delta: float, ankle_delta: float) -> dict:
    cfg = ConfigManager().load_robot_config("h1")
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

    active = cfg.active_joints
    stand = np.array([cfg.stand_pose.get(j, 0.0) for j in active], dtype=np.float32)
    target = stand.copy()
    # left hip_roll idx=1, right hip_roll idx=6
    # left ankle idx=4, right ankle idx=9
    target[1] += hip_roll_delta
    target[6] += hip_roll_delta
    target[4] += ankle_delta
    target[9] += ankle_delta

    max_torque = np.array(cfg.max_torque, dtype=np.float32)
    kp = 180.0
    kd = 12.0
    terminated = False

    for _ in range(int(1.0 * cfg.physics.physics_hz)):
        joint_pos = sim.get_joint_positions(active)
        joint_vel = sim.get_joint_velocities(active)
        torques = np.clip(kp * (target - joint_pos) - kd * joint_vel, -max_torque, max_torque)
        sim.set_joint_motor_control(active, torques)
        sim.step()
        base_pos = sim.get_base_position()
        euler = sim.get_base_orientation_euler()
        if base_pos[2] < cfg.torso_min_z or abs(float(euler[0])) > 0.5 or abs(float(euler[1])) > 0.5:
            terminated = True
            break

    base_pos = sim.get_base_position()
    euler = sim.get_base_orientation_euler()
    contacts = sim.get_foot_contact(cfg.feet_link_indices)
    sim.disconnect()
    return {
        "hip_roll_delta": hip_roll_delta,
        "ankle_delta": ankle_delta,
        "base_y": float(base_pos[1]),
        "base_z": float(base_pos[2]),
        "roll": float(euler[0]),
        "pitch": float(euler[1]),
        "l_contact": int(contacts["left"]),
        "r_contact": int(contacts["right"]),
        "terminated": terminated,
    }


def main() -> None:
    deltas = [-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12]
    results = []
    for hip_roll_delta, ankle_delta in itertools.product(deltas, [-0.08, 0.0, 0.08]):
        results.append(run_probe(hip_roll_delta, ankle_delta))

    results.sort(key=lambda r: (r["terminated"], abs(r["roll"]), -r["base_z"]))
    for r in results:
        print(
            f"hip_roll={r['hip_roll_delta']:+.2f}"
            f" ankle={r['ankle_delta']:+.2f}"
            f" y={r['base_y']:+.3f}"
            f" z={r['base_z']:.3f}"
            f" roll={r['roll']:+.3f}"
            f" pitch={r['pitch']:+.3f}"
            f" l={r['l_contact']}"
            f" r={r['r_contact']}"
            f" terminated={int(r['terminated'])}"
        )


if __name__ == "__main__":
    main()
