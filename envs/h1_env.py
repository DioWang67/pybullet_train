"""Unitree H1 humanoid environment.

Thin adapter over `WalkingEnv`. All controller logic lives in
`envs.balance_controller` and all tuning lives in `configs/robots/h1.yaml`.
Morphology-specific pieces (URDF loader, sensor-link mass zeroing) stay
here.
"""

from typing import Optional

import numpy as np

from config import RobotConfig
from simulators import PyBulletRobotSimulator

from .balance_controller import (
    apply_residual_action,
    compute_balance_torques,
    compute_stand_target,
    max_torque_vector,
)
from .base_walking_env import WalkingEnv

# Sensor / cosmetic links with no URDF inertia → PyBullet assigns a 1 kg
# ghost mass that perturbs the CoM. Zero them out after load.
_H1_SENSOR_LINK_SUFFIXES = (
    "imu_link",
    "logo_link",
    "imager_link",
    "rgb_module_link",
    "mid360_link",
)

# Minimum acceptable torso z after `_settle()`. If the robot has already
# collapsed below this, training from step 0 is pointless — surface it loud.
_SETTLE_MIN_HEIGHT_MARGIN = 0.10


class H1Env(WalkingEnv):
    """Unitree H1 walking environment."""

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        render_mode: Optional[str] = None,
    ):
        if robot_config is None:
            from config import ConfigManager
            robot_config = ConfigManager().load_robot_config("h1")

        simulator = PyBulletRobotSimulator(
            robot_description_name=robot_config.robot_description_name
            or "h1_description",
            physics_hz=robot_config.physics.physics_hz,
            render=render_mode == "human",
            gravity=robot_config.physics.gravity,
            num_solver_iterations=robot_config.physics.num_solver_iterations,
        )

        super().__init__(robot_config, simulator, render_mode)

        n_active = len(robot_config.active_joints)
        self._last_torques = np.zeros(n_active, dtype=np.float32)
        self._stand_target = compute_stand_target(robot_config)
        self._max_torque = max_torque_vector(robot_config)

    # ── Base pose / orientation ────────────────────────────────────────────

    def _get_base_position(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.robot_config.spawn_z], dtype=np.float32)

    def _get_base_orientation(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # ── Initialization / settle ────────────────────────────────────────────

    def _init_robot(self) -> None:
        """Set foot friction, reset joint state, configure motors, zero sensor masses."""
        cfg = self.robot_config
        sim = self.robot_simulator
        feet = cfg.feet_link_indices
        sim.set_dynamics(
            feet["left"], lateral_friction=cfg.foot_friction, restitution=0.0
        )
        sim.set_dynamics(
            feet["right"], lateral_friction=cfg.foot_friction, restitution=0.0
        )

        if not isinstance(sim, PyBulletRobotSimulator):
            return

        sim.reset_joint_state(cfg.stand_pose)
        # Explicitly zero locked joints BEFORE locking them.
        # stand_pose only contains active leg joints (0-9). After a crash,
        # arm/torso joints (10-18) may be at arbitrary angles with nonzero
        # velocity. If we skip resetting them, the position controller in
        # lock_joints() fights them during settle and destabilises the whole
        # robot — causing settle to fail on episode N>1.
        if cfg.lock_joints:
            sim.reset_joint_state({j: 0.0 for j in cfg.lock_joints})
            sim.disable_default_motors()
            sim.lock_joints(cfg.lock_joints)
        else:
            sim.disable_default_motors()

        # Zero ghost mass on sensor links (no URDF inertia → PyBullet default 1 kg).
        for j in range(sim.get_num_joints()):
            if sim.get_joint_name(j).endswith(_H1_SENSOR_LINK_SUFFIXES):
                sim.set_dynamics(j, mass=0.0)

        # Reset residual-action state on every episode to avoid carryover.
        self._last_torques = np.zeros_like(self._last_torques)

    def _settle(self) -> None:
        """Run pure PD for ~0.8s so the robot is standing at episode start.

        Asserts the robot is still upright at the end — a silent collapse
        during settle means upstream config is broken and training will be
        pure noise.
        """
        settle_steps = int(0.8 * self.robot_config.physics.physics_hz)
        for _ in range(settle_steps):
            torques = compute_balance_torques(
                self.robot_simulator,
                self.robot_config,
                self._stand_target,
                self._max_torque,
            )
            if isinstance(self.robot_simulator, PyBulletRobotSimulator):
                self.robot_simulator.set_joint_motor_control(
                    self.robot_config.active_joints, torques
                )
            self.robot_simulator.step()

        # Post-settle sanity: the PD baseline must leave the robot standing.
        z = float(self.robot_simulator.get_base_position()[2])
        floor = self.robot_config.torso_min_z + _SETTLE_MIN_HEIGHT_MARGIN
        if z < floor:
            raise RuntimeError(
                f"Settle failed: torso z={z:.3f} < {floor:.3f}. "
                "Check stand_pose / balance_control / lock_joints in the "
                "robot YAML before training."
            )

    # ── Action / observation ───────────────────────────────────────────────

    def _apply_action(self, action: np.ndarray) -> None:
        base = compute_balance_torques(
            self.robot_simulator,
            self.robot_config,
            self._stand_target,
            self._max_torque,
        )
        torques = apply_residual_action(
            base_torques=base,
            action=action,
            max_torque=self._max_torque,
            residual_scale=self.robot_config.balance_control.residual_torque_scale,
        )
        self._last_torques = torques

        if isinstance(self.robot_simulator, PyBulletRobotSimulator):
            self.robot_simulator.set_joint_motor_control(
                self.robot_config.active_joints, torques
            )

    def _get_last_torques(self) -> np.ndarray:
        return self._last_torques

    def _observe(self) -> np.ndarray:
        """Observation: torso(9) + joint_pos(n) + joint_vel(n) + contacts(2).

        For H1 n=10 → total obs_dim = 31 (matches auto-computed obs_dim in
        RobotConfig.__post_init__).
        """
        sim = self.robot_simulator
        cfg = self.robot_config

        pos = sim.get_base_position()
        orn_euler = sim.get_base_orientation_euler()
        lin_vel = sim.get_base_linear_velocity()
        ang_vel = sim.get_base_angular_velocity()

        torso = np.concatenate([
            [pos[2]],
            orn_euler[[1, 0]],  # pitch, roll
            lin_vel,
            ang_vel,
        ])

        joint_pos = sim.get_joint_positions(cfg.active_joints)
        joint_vel = sim.get_joint_velocities(cfg.active_joints)

        foot_contact = sim.get_foot_contact(cfg.feet_link_indices)
        contacts = np.array([
            float(foot_contact.get("left", False)),
            float(foot_contact.get("right", False)),
        ])

        return np.concatenate([torso, joint_pos, joint_vel, contacts]).astype(
            np.float32
        )
