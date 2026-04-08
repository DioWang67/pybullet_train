"""Shared stand PD + residual-torque balance controller.

This module is the single source of truth for the low-level controller used
by both the RL environment (`H1Env`, future biped envs) and the diagnostic
tooling (`h1_controller_tools`). Any tuning to the feedback distribution or
gain scheduling happens here, so sweep results always reflect real env
behavior.

The controller is morphology-agnostic: it uses `JointRoles` to locate
hip_pitch / ankle / hip_roll within `active_joints`. A robot missing a role
(e.g. a simplified biped with no independent ankle) simply leaves the field
`None` and that branch of the feedback is skipped.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config import BalanceControlConfig, JointRoles, RobotConfig
from simulators.robot_interface import RobotInterface


def compute_stand_target(cfg: RobotConfig) -> np.ndarray:
    """Return the stand-pose target vector aligned with active_joints order."""
    return np.array(
        [cfg.stand_pose.get(j, 0.0) for j in cfg.active_joints],
        dtype=np.float32,
    )


def max_torque_vector(cfg: RobotConfig) -> np.ndarray:
    return np.array(cfg.max_torque, dtype=np.float32)


def compute_balance_torques(
    sim: RobotInterface,
    cfg: RobotConfig,
    target_pose: np.ndarray,
    max_torque: np.ndarray,
    balance_cfg: Optional[BalanceControlConfig] = None,
    joint_roles: Optional[JointRoles] = None,
) -> np.ndarray:
    """Compute stand PD + body-pitch/roll feedback torques.

    Parameters
    ----------
    sim
        Connected robot simulator. Used for joint/base state queries only.
    cfg
        Robot configuration (for active_joints).
    target_pose
        Desired joint angles (shape = len(active_joints)). Pass a copy of
        the stand pose for pure standing, or a time-varying trajectory for
        tasks like weight-shift / single-leg lift.
    max_torque
        Per-joint torque limits (shape = len(active_joints)).
    balance_cfg
        PD gains and feedback distribution. Defaults to `cfg.balance_control`.
    joint_roles
        Semantic joint role mapping. Defaults to `cfg.joint_roles`.

    Returns
    -------
    np.ndarray
        Clipped torque vector in `[-max_torque, max_torque]`.

    Notes
    -----
    Sign convention (verified against H1 URDF via sanity_check.py):
      - Positive pitch → robot leans forward → positive torque added to both
        hip_pitch and ankle joints pulls the CoM back.
      - Positive roll  → robot leans right  → right hip_roll torque increases,
        left hip_roll torque decreases (differential abduction).
    If you port to a URDF with a different convention, audit sanity_check
    before changing signs here — the env and evaluator both use this path.
    """
    balance_cfg = balance_cfg or cfg.balance_control
    joint_roles = joint_roles or cfg.joint_roles

    active = cfg.active_joints
    pos = sim.get_joint_positions(active)
    vel = sim.get_joint_velocities(active)

    # Joint-space PD.
    torques = balance_cfg.stand_kp * (target_pose - pos) - balance_cfg.stand_kd * vel

    # Body attitude feedback.
    base_euler = sim.get_base_orientation_euler()
    base_ang_vel = sim.get_base_angular_velocity()
    pitch = float(base_euler[1])
    roll = float(base_euler[0])
    pitch_rate = float(base_ang_vel[1])
    roll_rate = float(base_ang_vel[0])

    pitch_correction = balance_cfg.pitch_kp * pitch + balance_cfg.pitch_kd * pitch_rate
    # Negated so that positive roll pushes the right hip_roll up (see note above).
    roll_correction = -(balance_cfg.roll_kp * roll + balance_cfg.roll_kd * roll_rate)

    # Distribute pitch correction across both legs (hip_pitch + ankle).
    for hip_idx in (joint_roles.left_hip_pitch, joint_roles.right_hip_pitch):
        if hip_idx is not None:
            torques[hip_idx] += balance_cfg.pitch_hip_gain * pitch_correction
    for ankle_idx in (joint_roles.left_ankle, joint_roles.right_ankle):
        if ankle_idx is not None:
            torques[ankle_idx] += balance_cfg.pitch_ankle_gain * pitch_correction

    # Differential roll correction across hip_roll joints.
    if joint_roles.left_hip_roll is not None:
        torques[joint_roles.left_hip_roll] += balance_cfg.roll_hip_gain * roll_correction
    if joint_roles.right_hip_roll is not None:
        torques[joint_roles.right_hip_roll] -= balance_cfg.roll_hip_gain * roll_correction

    return np.clip(torques, -max_torque, max_torque)


def apply_residual_action(
    base_torques: np.ndarray,
    action: np.ndarray,
    max_torque: np.ndarray,
    residual_scale: float,
) -> np.ndarray:
    """Combine stand PD torques with an RL residual action.

    `action` is assumed to be in [-1, 1]; it is scaled by
    `residual_scale * max_torque` before being added to `base_torques`.
    The final torque is clipped to `[-max_torque, max_torque]`.
    """
    residual = np.clip(action, -1.0, 1.0) * max_torque * residual_scale
    return np.clip(base_torques + residual, -max_torque, max_torque)
