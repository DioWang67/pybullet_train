"""
H1Env — Unitree H1 人形機器人走路 Gymnasium 環境

機器人：Unitree H1（via robot_descriptions）
特點：全串聯關節，無四連桿，PyBullet 可完整模擬
總質量：56.6 kg

控制：只控制腿部 10 個關節（手臂鎖定在 0）
物理：240Hz；策略：60Hz（4 substeps）

動作空間 (10)：腿部 10 關節標準化扭矩 [-1, 1]
  順序：L Hip Yaw/Roll/Pitch, L Knee, L Ankle,
        R Hip Yaw/Roll/Pitch, R Knee, R Ankle

觀測空間 (31)：
  軀幹 (9)：高度, pitch, roll, 線速度(3), 角速度(3)
  關節角 (10)：腿部 10 關節
  關節速度 (10)：腿部 10 關節
  腳底接觸 (2)：左踝/右踝

獎勵：
  alive_bonus    = 2.0
  height_reward  = (z - MIN_Z) * 3.0
  forward_reward = vx * 1.5
  smooth_penalty = 0.001 * ||torque||²
  death_penalty  = -30.0
"""

import os
import math
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

# ── 常數 ─────────────────────────────────────────────────────────────────────

PHYSICS_HZ = 240
POLICY_HZ  = 60
SUBSTEPS   = PHYSICS_HZ // POLICY_HZ   # 4

MAX_STEPS   = 1000
TORSO_MIN_Z = 0.50   # H1 正常站立 z≈0.99，低於 0.5 視為倒下

SPAWN_Z = 1.05

# 腿部關節 index（全串聯，無被動關節）
LEG_JOINTS = [0, 1, 2, 3, 4,    # L: HipYaw, HipRoll, HipPitch, Knee, Ankle
              5, 6, 7, 8, 9]     # R: HipYaw, HipRoll, HipPitch, Knee, Ankle

# 手臂 + 軀幹關節（鎖在 0）
LOCK_JOINTS = [10, 11, 12, 13, 14, 15, 16, 17, 18]

# 腳底 link index（ankle joint 的 child link）
LINK_L_FOOT = 4
LINK_R_FOOT = 9

# 最大扭矩（Nm）— 對應 URDF effort
MAX_TORQUE = np.array([
    200., 200., 200., 300., 40.,   # L: HipYaw, HipRoll, HipPitch, Knee, Ankle
    200., 200., 200., 300., 40.,   # R: same
], dtype=np.float32)

# 站立姿勢（全部 0 即可，已驗證 z=0.99 穩定）
STAND_POSE = {j: 0.0 for j in range(19)}


class H1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": POLICY_HZ}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode  = render_mode
        self._client: int = -1
        self._robot: int  = -1
        self._plane: int  = -1
        self._step_count  = 0

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        obs_limit = np.full(31, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_limit, obs_limit, dtype=np.float32)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._client < 0:
            self._connect()

        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / PHYSICS_HZ, physicsClientId=self._client)
        p.setPhysicsEngineParameter(numSolverIterations=50,
                                    physicsClientId=self._client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._plane = p.loadURDF("plane.urdf", physicsClientId=self._client)
        p.changeDynamics(self._plane, -1,
                         lateralFriction=0.8, restitution=0.0,
                         physicsClientId=self._client)

        from robot_descriptions.loaders.pybullet import load_robot_description
        self._robot = load_robot_description(
            "h1_description",
            basePosition=[0.0, 0.0, SPAWN_Z],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self._client,
        )

        # 腳底高摩擦
        for link in (LINK_L_FOOT, LINK_R_FOOT):
            p.changeDynamics(self._robot, link,
                             lateralFriction=1.5, restitution=0.0,
                             physicsClientId=self._client)

        # 站立姿勢
        for j, angle in STAND_POSE.items():
            p.resetJointState(self._robot, j, angle,
                              physicsClientId=self._client)

        # 關閉預設馬達（切換為扭矩控制）
        for j in LEG_JOINTS:
            p.setJointMotorControl2(
                self._robot, j,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self._client,
            )

        # 手臂 + 軀幹鎖在 0
        for j in LOCK_JOINTS:
            p.setJointMotorControl2(
                self._robot, j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
                positionGain=1.0,
                velocityGain=0.1,
                force=200.0,
                physicsClientId=self._client,
            )

        self._step_count = 0

        # 沉降 0.3 秒
        for _ in range(int(0.3 * PHYSICS_HZ)):
            p.stepSimulation(physicsClientId=self._client)

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0, cameraYaw=30, cameraPitch=-15,
                cameraTargetPosition=[0.0, 0.0, 1.0],
                physicsClientId=self._client,
            )

        return self._observe(), {}

    def step(self, action: np.ndarray):
        torques = np.clip(action, -1.0, 1.0) * MAX_TORQUE

        for idx, j in enumerate(LEG_JOINTS):
            p.setJointMotorControl2(
                self._robot, j,
                controlMode=p.TORQUE_CONTROL,
                force=float(torques[idx]),
                physicsClientId=self._client,
            )

        for _ in range(SUBSTEPS):
            p.stepSimulation(physicsClientId=self._client)

        self._step_count += 1
        obs = self._observe()

        pos, _ = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._client)
        lin_vel, _ = p.getBaseVelocity(
            self._robot, physicsClientId=self._client)

        terminated = pos[2] < TORSO_MIN_Z
        truncated  = self._step_count >= MAX_STEPS

        alive_bonus    = 2.0 if not terminated else 0.0
        height_reward  = (pos[2] - TORSO_MIN_Z) * 3.0
        forward_reward = lin_vel[0] * 1.5
        smooth_penalty = 0.001 * float(np.sum(torques ** 2))
        death_penalty  = -30.0 if terminated else 0.0

        reward = alive_bonus + height_reward + forward_reward \
                 - smooth_penalty + death_penalty

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0, cameraYaw=30, cameraPitch=-15,
                cameraTargetPosition=[pos[0], pos[1], 1.0],
                physicsClientId=self._client,
            )

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        if self._client >= 0:
            p.disconnect(physicsClientId=self._client)
            self._client = -1

    # ── 內部 ──────────────────────────────────────────────────────────────────

    def _connect(self):
        if self.render_mode == "human":
            self._client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1,
                                       physicsClientId=self._client)
        else:
            self._client = p.connect(p.DIRECT)

    def _observe(self) -> np.ndarray:
        pos, orn = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._client)
        lin_vel, ang_vel = p.getBaseVelocity(
            self._robot, physicsClientId=self._client)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        torso = [
            pos[2], pitch, roll,
            lin_vel[0], lin_vel[1], lin_vel[2],
            ang_vel[0], ang_vel[1], ang_vel[2],
        ]  # 9

        joint_pos = [
            p.getJointState(self._robot, j,
                            physicsClientId=self._client)[0]
            for j in LEG_JOINTS
        ]  # 10

        joint_vel = [
            p.getJointState(self._robot, j,
                            physicsClientId=self._client)[1]
            for j in LEG_JOINTS
        ]  # 10

        l_contact = int(len(p.getContactPoints(
            self._robot, self._plane,
            linkIndexA=LINK_L_FOOT,
            physicsClientId=self._client,
        )) > 0)
        r_contact = int(len(p.getContactPoints(
            self._robot, self._plane,
            linkIndexA=LINK_R_FOOT,
            physicsClientId=self._client,
        )) > 0)  # 2

        return np.array(torso + joint_pos + joint_vel + [l_contact, r_contact],
                        dtype=np.float32)  # 31
