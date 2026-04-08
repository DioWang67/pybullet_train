"""Fast H1 controller evaluation in DIRECT mode.

This tool drives the *exact same* low-level controller as the RL env (via
`envs.balance_controller.compute_balance_torques`), so sweep results
directly predict in-env behavior. The previous version copy-pasted the
controller body and silently diverged whenever someone tuned the env.

Tasks measured:
  - stand:        hold the stand pose for 2 s.
  - weight_shift: cycle CoM right ↔ left via differential hip_roll.
  - lift_left:    shift onto the right foot, then lift the left foot.

Each task gets a normalized score in [0, 1]; the composite total is a
weighted mean. This makes runs comparable even when one task's max raw
score changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import pybullet as p

from config import BalanceControlConfig, ConfigManager, JointRoles, RobotConfig
from envs.balance_controller import compute_balance_torques, max_torque_vector
from simulators import PyBulletRobotSimulator


# ── Pose / controller parameters ──────────────────────────────────────────


@dataclass(frozen=True)
class PoseParams:
    """Symmetric stand pose parameters (applied to both legs)."""
    hip_pitch: float
    knee: float
    ankle: float
    hip_roll_bias: float = 0.0


@dataclass(frozen=True)
class ControllerParams:
    """Subset of `BalanceControlConfig` exposed for sweeps.

    Kept as a thin proxy so legacy sweep scripts (`h1_controller_eval.py`,
    `h1_controller_sweep.py`) and their CLI flags keep working unchanged.
    """
    stand_kp: float = 120.0
    stand_kd: float = 8.0
    pitch_kp: float = 45.0
    pitch_kd: float = 8.0
    roll_kp: float = 10.0
    roll_kd: float = 2.0

    def to_balance_config(
        self, base: Optional[BalanceControlConfig] = None
    ) -> BalanceControlConfig:
        """Project this sweep point onto a full BalanceControlConfig.

        Distribution gains (pitch_hip_gain etc.) and residual_torque_scale
        are taken from `base` (defaults to `BalanceControlConfig()`), so the
        sweep only varies what the user actually requested on the CLI.
        """
        base = base or BalanceControlConfig()
        return BalanceControlConfig(
            stand_kp=self.stand_kp,
            stand_kd=self.stand_kd,
            residual_torque_scale=base.residual_torque_scale,
            pitch_kp=self.pitch_kp,
            pitch_kd=self.pitch_kd,
            pitch_hip_gain=base.pitch_hip_gain,
            pitch_ankle_gain=base.pitch_ankle_gain,
            roll_kp=self.roll_kp,
            roll_kd=self.roll_kd,
            roll_hip_gain=base.roll_hip_gain,
        )


# ── Metrics ────────────────────────────────────────────────────────────────


@dataclass
class TaskMetrics:
    survived: bool
    duration: float
    elapsed: float
    final_z: float
    max_abs_pitch: float
    max_abs_roll: float
    y_span: float
    swing_clearance: float
    single_support_ratio: float
    expected_support_ratio: float
    score: float = 0.0
    norm_score: float = 0.0


@dataclass
class EvalResult:
    pose: PoseParams
    controller: ControllerParams
    stand: TaskMetrics
    weight_shift: TaskMetrics
    lift_left: TaskMetrics
    total_score: float


# ── Config / sim setup ─────────────────────────────────────────────────────


def load_cfg() -> RobotConfig:
    return ConfigManager().load_robot_config("h1")


def default_pose_from_cfg(cfg: RobotConfig) -> PoseParams:
    """Read the symmetric pose values from the loaded YAML stand_pose.

    Uses joint_roles when available so this works across robots; falls back
    to the H1-historical indices (1, 2, 3, 4) when roles are missing.
    """
    roles = cfg.joint_roles
    active = cfg.active_joints

    def stand(role_idx_in_active: Optional[int], fallback: int) -> float:
        if role_idx_in_active is None:
            return float(cfg.stand_pose.get(fallback, 0.0))
        urdf_idx = active[role_idx_in_active]
        return float(cfg.stand_pose.get(urdf_idx, 0.0))

    return PoseParams(
        hip_pitch=stand(roles.left_hip_pitch, 2),
        knee=stand(roles.left_knee, 3),
        ankle=stand(roles.left_ankle, 4),
        hip_roll_bias=stand(roles.left_hip_roll, 1),
    )


def build_stand_pose_dict(cfg: RobotConfig, pose: PoseParams) -> dict[int, float]:
    """Apply symmetric pose to both legs via joint_roles.

    Returns a dict keyed by URDF joint index (the format `stand_pose` uses
    everywhere else in the codebase).
    """
    stand_pose = dict(cfg.stand_pose)
    roles = cfg.joint_roles
    active = cfg.active_joints

    def assign(role_idx: Optional[int], value: float) -> None:
        if role_idx is None:
            return
        stand_pose[active[role_idx]] = value

    assign(roles.left_hip_roll, pose.hip_roll_bias)
    assign(roles.right_hip_roll, pose.hip_roll_bias)
    assign(roles.left_hip_pitch, pose.hip_pitch)
    assign(roles.right_hip_pitch, pose.hip_pitch)
    assign(roles.left_knee, pose.knee)
    assign(roles.right_knee, pose.knee)
    assign(roles.left_ankle, pose.ankle)
    assign(roles.right_ankle, pose.ankle)
    return stand_pose


def stand_array(cfg: RobotConfig, stand_pose_dict: dict[int, float]) -> np.ndarray:
    return np.array(
        [stand_pose_dict.get(j, 0.0) for j in cfg.active_joints],
        dtype=np.float32,
    )


def init_sim(
    cfg: RobotConfig,
    stand_pose_dict: dict[int, float],
    render: bool = False,
) -> PyBulletRobotSimulator:
    sim = PyBulletRobotSimulator(
        robot_description_name=cfg.robot_description_name or "h1_description",
        physics_hz=cfg.physics.physics_hz,
        render=render,
        gravity=cfg.physics.gravity,
        num_solver_iterations=cfg.physics.num_solver_iterations,
    )
    sim.connect()
    sim.reset(
        np.array([0.0, 0.0, cfg.spawn_z], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    sim.reset_joint_state(stand_pose_dict)
    sim.disable_default_motors()
    sim.lock_joints(cfg.lock_joints)
    sim.set_dynamics(
        cfg.feet_link_indices["left"],
        lateral_friction=cfg.foot_friction,
        restitution=0.0,
    )
    sim.set_dynamics(
        cfg.feet_link_indices["right"],
        lateral_friction=cfg.foot_friction,
        restitution=0.0,
    )

    # Zero ghost mass on H1 sensor links (mirrors H1Env._init_robot).
    for joint_idx in range(sim.get_num_joints()):
        name = sim.get_joint_name(joint_idx)
        if name.endswith(
            ("imu_link", "logo_link", "imager_link", "rgb_module_link", "mid360_link")
        ):
            sim.set_dynamics(joint_idx, mass=0.0)
    return sim


def settle_sim(
    sim: PyBulletRobotSimulator,
    cfg: RobotConfig,
    stand: np.ndarray,
    balance_cfg: BalanceControlConfig,
    seconds: float = 0.8,
) -> None:
    """Run pure-PD settle so the robot is on the ground before the task."""
    n_steps = max(1, int(seconds * cfg.physics.physics_hz))
    max_torque = max_torque_vector(cfg)
    for _ in range(n_steps):
        torques = compute_balance_torques(
            sim, cfg, stand, max_torque, balance_cfg=balance_cfg
        )
        sim.set_joint_motor_control(cfg.active_joints, torques)
        sim.step()


# ── Task trajectories ──────────────────────────────────────────────────────
#
# Single source of truth for the symmetric weight-shift / lift kinematics.
# Indices come from joint_roles, so the same trajectory generator works for
# any biped that fills in roles.


@dataclass
class _LegIndices:
    hip_roll: int
    hip_pitch: int
    knee: int
    ankle: int


def _resolve_legs(roles: JointRoles) -> tuple[_LegIndices, _LegIndices]:
    required = (
        roles.left_hip_roll, roles.left_hip_pitch, roles.left_knee, roles.left_ankle,
        roles.right_hip_roll, roles.right_hip_pitch, roles.right_knee, roles.right_ankle,
    )
    if any(idx is None for idx in required):
        raise ValueError(
            "h1_controller_tools requires all left/right hip_roll/hip_pitch/"
            "knee/ankle joint_roles to be set."
        )
    left = _LegIndices(
        hip_roll=roles.left_hip_roll,    # type: ignore[arg-type]
        hip_pitch=roles.left_hip_pitch,  # type: ignore[arg-type]
        knee=roles.left_knee,            # type: ignore[arg-type]
        ankle=roles.left_ankle,          # type: ignore[arg-type]
    )
    right = _LegIndices(
        hip_roll=roles.right_hip_roll,    # type: ignore[arg-type]
        hip_pitch=roles.right_hip_pitch,  # type: ignore[arg-type]
        knee=roles.right_knee,            # type: ignore[arg-type]
        ankle=roles.right_ankle,          # type: ignore[arg-type]
    )
    return left, right


# Kinematic deltas for the weight-shift / lift trajectories.
# Differential hip_roll (positive on stance leg, negative on swing leg)
# actually moves CoM laterally; previous version moved both legs by the
# same amount which only tilts the body without transferring weight.
_HIP_ROLL_SHIFT = 0.10   # rad of differential hip_roll between stance/swing
_ANKLE_SHIFT = 0.05      # rad of inward ankle to support the lateral CoM
_LIFT_HIP_PITCH = 0.10   # swing-leg hip flexion during the lift
_LIFT_KNEE = 0.25        # swing-leg knee flexion during the lift
_LIFT_ANKLE = 0.10       # swing-leg ankle dorsiflexion during the lift


def _shifted_pose(
    stand: np.ndarray,
    legs: tuple[_LegIndices, _LegIndices],
    direction: int,
) -> np.ndarray:
    """Build a target pose loaded onto one foot.

    direction = +1 → load right foot (CoM moves right);
                -1 → load left foot.
    """
    left, right = legs
    out = stand.copy()
    # Differential hip_roll: stance leg abducts, swing leg adducts.
    out[right.hip_roll] += -direction * _HIP_ROLL_SHIFT
    out[left.hip_roll] += -direction * _HIP_ROLL_SHIFT
    # Symmetric ankle inversion holds the CoM over the stance foot.
    out[right.ankle] += direction * _ANKLE_SHIFT
    out[left.ankle] += direction * _ANKLE_SHIFT
    return out


def _make_target_factories(roles: JointRoles):
    legs = _resolve_legs(roles)
    left_leg, _ = legs

    def stand_target(stand: np.ndarray, _t: float) -> np.ndarray:
        return stand

    def weight_shift_target(stand: np.ndarray, t: float) -> np.ndarray:
        right_pose = _shifted_pose(stand, legs, direction=+1)
        left_pose = _shifted_pose(stand, legs, direction=-1)
        if t < 0.8:
            alpha = t / 0.8
            return stand + alpha * (right_pose - stand)
        if t < 1.5:
            return right_pose
        if t < 2.3:
            alpha = (t - 1.5) / 0.8
            return right_pose + alpha * (left_pose - right_pose)
        return left_pose

    def lift_left_target(stand: np.ndarray, t: float) -> np.ndarray:
        right_pose = _shifted_pose(stand, legs, direction=+1)
        if t < 0.9:
            alpha = t / 0.9
            return stand + alpha * (right_pose - stand)
        if t < 1.6:
            return right_pose
        # Apply lift deltas to the LEFT leg (swing) on top of right_pose.
        lift = right_pose.copy()
        if t < 2.4:
            alpha = (t - 1.6) / 0.8
        elif t < 3.0:
            alpha = 1.0
        else:
            alpha = max(0.0, 1.0 - (t - 3.0) / 0.8)
        lift[left_leg.hip_pitch] += _LIFT_HIP_PITCH * alpha
        lift[left_leg.knee] += _LIFT_KNEE * alpha
        lift[left_leg.ankle] += _LIFT_ANKLE * alpha
        return lift

    return {
        "stand": stand_target,
        "weight_shift": weight_shift_target,
        "lift_left": lift_left_target,
    }


# ── Scoring ────────────────────────────────────────────────────────────────


def termination_limits(cfg: RobotConfig) -> tuple[float, float]:
    pitch_limit = float(getattr(cfg.reward, "pitch_termination", 0.35))
    roll_limit = float(getattr(cfg.reward, "roll_termination", 0.30))
    if not np.isfinite(pitch_limit):
        pitch_limit = 0.35
    if not np.isfinite(roll_limit):
        roll_limit = 0.30
    return pitch_limit, roll_limit


def foot_height(sim: PyBulletRobotSimulator, foot_link_index: int) -> float:
    state = p.getLinkState(
        sim.get_robot_id(),
        foot_link_index,
        physicsClientId=sim.get_client_id(),
    )
    return float(state[0][2])


# Per-task max raw scores. Used to normalize total_score so changing one
# task's bonus weights does not silently rescale the composite metric.
_TASK_MAX = {"stand": 8.0, "weight_shift": 12.0, "lift_left": 13.0}


def task_score(task_name: str, metrics: TaskMetrics, cfg: RobotConfig) -> tuple[float, float]:
    """Return (raw_score, normalized_score in [0,1])."""
    pitch_limit, roll_limit = termination_limits(cfg)
    survival_ratio = (
        metrics.elapsed / metrics.duration if metrics.duration > 0.0 else 0.0
    )
    pitch_term = min(1.5, metrics.max_abs_pitch / max(1e-6, pitch_limit))
    roll_term = min(1.5, metrics.max_abs_roll / max(1e-6, roll_limit))
    z_bonus = float(np.clip((metrics.final_z - cfg.torso_min_z) / 0.25, 0.0, 1.0))

    score = 6.0 * survival_ratio + 2.0 * z_bonus - 2.0 * pitch_term - 2.0 * roll_term

    if task_name == "weight_shift":
        score += 2.0 * min(metrics.y_span / 0.06, 1.0)
        score += 2.0 * metrics.single_support_ratio
    elif task_name == "lift_left":
        score += 3.0 * min(metrics.swing_clearance / 0.04, 1.0)
        score += 2.0 * metrics.expected_support_ratio

    raw = float(score)
    norm = float(np.clip(raw / _TASK_MAX[task_name], -1.0, 1.0))
    return raw, norm


# ── Task runner ────────────────────────────────────────────────────────────


def run_task(
    cfg: RobotConfig,
    pose: PoseParams,
    controller: ControllerParams,
    task_name: str,
    duration: float,
    target_fn: Callable[[np.ndarray, float], np.ndarray],
    render: bool = False,
) -> TaskMetrics:
    stand_pose_dict = build_stand_pose_dict(cfg, pose)
    stand = stand_array(cfg, stand_pose_dict)
    balance_cfg = controller.to_balance_config(cfg.balance_control)

    sim = init_sim(cfg, stand_pose_dict, render=render)
    pitch_limit, roll_limit = termination_limits(cfg)

    swing_is_left = task_name == "lift_left"
    swing_foot = (
        cfg.feet_link_indices["left"]
        if swing_is_left
        else cfg.feet_link_indices["right"]
    )
    support_foot_name = "right" if swing_is_left else "left"
    swing_foot_name = "left" if swing_is_left else "right"

    y_values: list[float] = []
    pitch_values: list[float] = []
    roll_values: list[float] = []
    single_support_steps = 0
    expected_support_steps = 0
    elapsed = 0.0
    final_z = cfg.spawn_z
    swing_clearance = 0.0
    survived = True

    try:
        # Settle phase: get the robot onto the ground at the stand pose so
        # foot_baseline reflects the actual on-ground foot z, not the
        # mid-air spawn position.
        settle_sim(sim, cfg, stand, balance_cfg, seconds=0.8)
        foot_baseline = foot_height(sim, swing_foot)

        total_steps = max(1, int(duration * cfg.physics.physics_hz))
        for step in range(total_steps):
            elapsed = (step + 1) / cfg.physics.physics_hz
            target = target_fn(stand, step / cfg.physics.physics_hz)
            torques = compute_balance_torques(
                sim, cfg, target, max_torque_vector(cfg), balance_cfg=balance_cfg
            )
            sim.set_joint_motor_control(cfg.active_joints, torques)
            sim.step()

            pos = sim.get_base_position()
            euler = sim.get_base_orientation_euler()
            contacts = sim.get_foot_contact(cfg.feet_link_indices)

            final_z = float(pos[2])
            pitch = abs(float(euler[1]))
            roll = abs(float(euler[0]))
            y_values.append(float(pos[1]))
            pitch_values.append(pitch)
            roll_values.append(roll)

            if contacts["left"] != contacts["right"]:
                single_support_steps += 1
            if (
                contacts.get(support_foot_name, False)
                and not contacts.get(swing_foot_name, False)
            ):
                expected_support_steps += 1

            if task_name == "lift_left":
                swing_clearance = max(
                    swing_clearance, foot_height(sim, swing_foot) - foot_baseline
                )

            if final_z < cfg.torso_min_z or pitch > pitch_limit or roll > roll_limit:
                survived = False
                break
    finally:
        sim.disconnect()

    metrics = TaskMetrics(
        survived=survived,
        duration=duration,
        elapsed=elapsed,
        final_z=final_z,
        max_abs_pitch=max(pitch_values, default=0.0),
        max_abs_roll=max(roll_values, default=0.0),
        y_span=(max(y_values) - min(y_values)) if y_values else 0.0,
        swing_clearance=max(0.0, swing_clearance),
        single_support_ratio=single_support_steps / max(1, len(y_values)),
        expected_support_ratio=expected_support_steps / max(1, len(y_values)),
    )
    raw, norm = task_score(task_name, metrics, cfg)
    metrics.score = raw
    metrics.norm_score = norm
    return metrics


def evaluate_controller(
    pose: Optional[PoseParams] = None,
    controller: Optional[ControllerParams] = None,
    render: bool = False,
) -> EvalResult:
    cfg = load_cfg()
    pose = pose or default_pose_from_cfg(cfg)
    controller = controller or ControllerParams()

    factories = _make_target_factories(cfg.joint_roles)

    stand = run_task(
        cfg, pose, controller, "stand",
        duration=2.0, target_fn=factories["stand"], render=render,
    )
    weight_shift = run_task(
        cfg, pose, controller, "weight_shift",
        duration=3.0, target_fn=factories["weight_shift"], render=render,
    )
    lift_left = run_task(
        cfg, pose, controller, "lift_left",
        duration=3.8, target_fn=factories["lift_left"], render=render,
    )

    # Composite uses normalized per-task scores so changing the bonus
    # weights inside one task does not rescale the composite metric.
    total_norm = (
        0.35 * stand.norm_score
        + 0.40 * weight_shift.norm_score
        + 0.25 * lift_left.norm_score
    )

    return EvalResult(
        pose=pose,
        controller=controller,
        stand=stand,
        weight_shift=weight_shift,
        lift_left=lift_left,
        total_score=float(total_norm),
    )


def result_to_dict(result: EvalResult) -> dict:
    return {
        "pose": asdict(result.pose),
        "controller": asdict(result.controller),
        "stand": asdict(result.stand),
        "weight_shift": asdict(result.weight_shift),
        "lift_left": asdict(result.lift_left),
        "total_score": result.total_score,
    }
