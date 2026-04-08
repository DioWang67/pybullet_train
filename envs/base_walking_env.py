"""
Walking environment base class.
"""

from abc import abstractmethod
from typing import Optional, Tuple, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import RobotConfig
from simulators.robot_interface import RobotInterface


class WalkingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        robot_config: RobotConfig,
        robot_simulator: RobotInterface,
        render_mode: Optional[str] = None,
    ):
        self.robot_config = robot_config
        self.robot_simulator = robot_simulator
        self.render_mode = render_mode
        self._step_count = 0

        # `obs_dim` / `action_dim` should already be auto-populated by
        # RobotConfig.__post_init__ from `active_joints`. Assert loudly so a
        # misconfigured YAML cannot silently produce a zero-dim gym space.
        if robot_config.action_dim <= 0:
            raise ValueError(
                f"robot_config.action_dim={robot_config.action_dim}; "
                "check active_joints in the robot YAML."
            )
        if robot_config.obs_dim <= 0:
            raise ValueError(
                f"robot_config.obs_dim={robot_config.obs_dim}; "
                "check active_joints in the robot YAML."
            )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(robot_config.action_dim,),
            dtype=np.float32,
        )

        obs_limit = np.full(robot_config.obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_limit, obs_limit, dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if not self._is_simulator_connected():
            self.robot_simulator.connect()

        base_pos = self._get_base_position()
        base_orn = self._get_base_orientation()
        self.robot_simulator.reset(base_pos, base_orn)
        self._init_robot()
        self._settle()

        if self.render_mode == "human":
            self.robot_simulator.enable_rendering(base_pos)

        self._step_count = 0
        obs = self._observe()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        for _ in range(self.robot_config.physics.substeps):
            self._apply_action(action)
            self.robot_simulator.step()

        self._step_count += 1
        obs = self._observe()
        reward, terminated, truncated = self._compute_reward_and_termination()

        if self.render_mode == "human":
            robot_pos = self.robot_simulator.get_base_position()
            self.robot_simulator.update_camera(robot_pos)

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        if self._is_simulator_connected():
            self.robot_simulator.disconnect()

    def _compute_reward_and_termination(self) -> Tuple[float, bool, bool]:
        pos = self.robot_simulator.get_base_position()
        orn = self.robot_simulator.get_base_orientation_euler()
        lin_vel = self.robot_simulator.get_base_linear_velocity()
        ang_vel = self.robot_simulator.get_base_angular_velocity()
        torques = self._get_last_torques()
        roll = float(orn[0])
        pitch = float(orn[1])
        r = self.robot_config.reward

        terminated = bool(
            pos[2] < self.robot_config.torso_min_z
            or abs(pitch) > r.pitch_termination
            or abs(roll) > r.roll_termination
        )
        truncated = bool(self._step_count >= self.robot_config.max_steps)

        alive_bonus = r.alive_bonus if not terminated else 0.0
        height_reward = (pos[2] - self.robot_config.torso_min_z) * r.height_reward_scale
        forward_reward = lin_vel[0] * r.forward_reward_scale
        smooth_penalty = r.smooth_penalty_scale * float(np.sum(torques ** 2))
        posture_penalty = r.posture_penalty_scale * (pitch ** 2 + roll ** 2)
        # Penalize downward z-velocity only. A small upward bounce during
        # recovery is fine and should not be punished.
        downward_vz = max(0.0, -float(lin_vel[2]))
        vertical_velocity_penalty = (
            r.vertical_velocity_penalty_scale * downward_vz ** 2
        )
        angular_velocity_penalty = (
            r.angular_velocity_penalty_scale * float(np.sum(ang_vel ** 2))
        )
        death_penalty = r.death_penalty if terminated else 0.0

        reward = (
            alive_bonus
            + height_reward
            + forward_reward
            - smooth_penalty
            - posture_penalty
            - vertical_velocity_penalty
            - angular_velocity_penalty
            + death_penalty
        )
        return float(reward), terminated, truncated

    @abstractmethod
    def _get_base_position(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_base_orientation(self) -> np.ndarray:
        pass

    @abstractmethod
    def _init_robot(self) -> None:
        pass

    @abstractmethod
    def _settle(self) -> None:
        pass

    @abstractmethod
    def _observe(self) -> np.ndarray:
        pass

    @abstractmethod
    def _apply_action(self, action: np.ndarray) -> None:
        pass

    @abstractmethod
    def _get_last_torques(self) -> np.ndarray:
        pass

    def _is_simulator_connected(self) -> bool:
        try:
            if hasattr(self.robot_simulator, "get_client_id"):
                return self.robot_simulator.get_client_id() >= 0
            return True
        except Exception:
            return False
