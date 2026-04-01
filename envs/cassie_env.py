"""
CassieEnv — Cassie 雙足機器人走路 Gymnasium 環境

機器人：Agility Robotics Cassie（via robot_descriptions）
物理：PyBullet 500Hz；策略：50Hz（10 substeps）

動作空間 (10)：10 個主動關節標準化扭矩 [-1, 1]
  順序：L/R HipRoll, HipYaw, HipPitch, Knee, Foot

觀測空間 (41)：
  軀幹 (9)：高度, pitch, roll, 線速度(3), 角速度(3)
  關節角 (16)：全部 16 個關節
  關節速度 (14)：10 主動 + 4 被動（Shin/Tarsus 各側）
  腳底接觸 (2)：左腳/右腳

獎勵：
  alive_bonus    = 2.0（站著就給）
  height_reward  = (z - MIN_Z) * 3.0（站越高越好）
  forward_reward = vx * 1.5（前進速度）
  smooth_penalty = 0.001 * ||torque||²（扭矩平滑）
  death_penalty  = -30.0（倒下）
"""

import os
import math
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

# ── 常數 ─────────────────────────────────────────────────────────────────────

PHYSICS_HZ  = 500
POLICY_HZ   = 50
SUBSTEPS    = PHYSICS_HZ // POLICY_HZ   # 10

MAX_STEPS   = 1000
TORSO_MIN_Z = 0.60   # 低於此視為倒下（Cassie 正常站立 z≈0.98）
SPAWN_Z     = 1.00

# 主動關節 index（effort > 0）
ACTIVE_JOINTS = [0, 1, 2, 3, 7, 8, 9, 10, 11, 15]  # L: Roll,Yaw,Pitch,Knee,Foot  R: 同上

# 腳部 link index（用於接觸偵測）
LINK_L_FOOT = 7
LINK_R_FOOT = 15

# 最大扭矩（Nm），對應 URDF effort，乘上安全係數
MAX_TORQUE = np.array([
    4.5, 4.5, 12.2, 12.2, 0.9,   # L: Roll,Yaw,Pitch,Knee,Foot
    4.5, 4.5, 12.2, 12.2, 0.9,   # R: Roll,Yaw,Pitch,Knee,Foot
], dtype=np.float32)

# Cassie 官方站立姿勢（rad），用於 reset
STAND_POSE = {
    0:  0.0045,   1:  0.0,   2:  0.4973,  3: -1.1997,
    4:  0.0,      5:  1.4267, 6:  0.0,    7: -1.5968,
    8: -0.0045,   9:  0.0,   10:  0.4973, 11: -1.1997,
   12:  0.0,     13:  1.4267, 14:  0.0,  15: -1.5968,
}


class CassieEnv(gym.Env):
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
        obs_limit = np.full(41, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_limit, obs_limit, dtype=np.float32)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._client < 0:
            self._connect()

        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / PHYSICS_HZ, physicsClientId=self._client)
        p.setPhysicsEngineParameter(
            numSolverIterations=50,
            physicsClientId=self._client,
        )
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._plane = p.loadURDF("plane.urdf", physicsClientId=self._client)
        p.changeDynamics(
            self._plane, -1,
            lateralFriction=0.8, restitution=0.0,
            physicsClientId=self._client,
        )

        # 從 robot_descriptions 載入 Cassie URDF
        from robot_descriptions.loaders.pybullet import load_robot_description
        self._robot = load_robot_description(
            "cassie_description",
            basePosition=[0.0, 0.0, SPAWN_Z],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self._client,
        )

        # 腳底摩擦力
        for link in (LINK_L_FOOT, LINK_R_FOOT):
            p.changeDynamics(
                self._robot, link,
                lateralFriction=1.2, restitution=0.0,
                physicsClientId=self._client,
            )

        # 重置為站立姿勢
        for j, angle in STAND_POSE.items():
            p.resetJointState(self._robot, j, angle,
                              physicsClientId=self._client)

        # 關閉預設馬達（扭矩控制）
        for j in range(16):
            p.setJointMotorControl2(
                self._robot, j,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self._client,
            )

        self._step_count = 0

        # 沉降 0.3 秒
        for _ in range(int(0.3 * PHYSICS_HZ)):
            p.stepSimulation(physicsClientId=self._client)

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0, cameraYaw=30, cameraPitch=-15,
                cameraTargetPosition=[0.0, 0.0, 0.8],
                physicsClientId=self._client,
            )

        return self._observe(), {}

    def step(self, action: np.ndarray):
        torques = np.clip(action, -1.0, 1.0) * MAX_TORQUE

        for idx, j in enumerate(ACTIVE_JOINTS):
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
            self._robot, physicsClientId=self._client
        )
        lin_vel, _ = p.getBaseVelocity(
            self._robot, physicsClientId=self._client
        )

        terminated = pos[2] < TORSO_MIN_Z
        truncated  = self._step_count >= MAX_STEPS

        alive_bonus    = 2.0 if not terminated else 0.0
        height_reward  = (pos[2] - TORSO_MIN_Z) * 3.0
        forward_reward = lin_vel[0] * 1.5
        smooth_penalty = 0.001 * float(np.sum(torques ** 2))
        death_penalty  = -30.0 if terminated else 0.0

        reward = alive_bonus + height_reward + forward_reward - smooth_penalty + death_penalty

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0, cameraYaw=30, cameraPitch=-15,
                cameraTargetPosition=[pos[0], pos[1], 0.8],
                physicsClientId=self._client,
            )

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        if self._client >= 0:
            p.disconnect(physicsClientId=self._client)
            self._client = -1

    # ── 內部方法 ──────────────────────────────────────────────────────────────

    def _connect(self):
        if self.render_mode == "human":
            self._client = p.connect(p.GUI)
            p.configureDebugVisualizer(
                p.COV_ENABLE_SHADOWS, 1,
                physicsClientId=self._client,
            )
        else:
            self._client = p.connect(p.DIRECT)

    def _observe(self) -> np.ndarray:
        pos, orn = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._client
        )
        lin_vel, ang_vel = p.getBaseVelocity(
            self._robot, physicsClientId=self._client
        )
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        # 軀幹狀態 (9)
        torso = [
            pos[2], pitch, roll,
            lin_vel[0], lin_vel[1], lin_vel[2],
            ang_vel[0], ang_vel[1], ang_vel[2],
        ]

        # 全部 16 個關節角度 (16)
        joint_pos = [
            p.getJointState(self._robot, j, physicsClientId=self._client)[0]
            for j in range(16)
        ]

        # 主動關節速度 (10) + 被動 Shin/Tarsus 速度 (4) = 14
        vel_joints = ACTIVE_JOINTS + [4, 5, 12, 13]  # passive: L/R Shin, L/R Tarsus
        joint_vel = [
            p.getJointState(self._robot, j, physicsClientId=self._client)[1]
            for j in vel_joints
        ]

        # 腳底接觸 (2)
        l_contact = int(len(p.getContactPoints(
            self._robot, self._plane,
            linkIndexA=LINK_L_FOOT,
            physicsClientId=self._client,
        )) > 0)
        r_contact = int(len(p.getContactPoints(
            self._robot, self._plane,
            linkIndexA=LINK_R_FOOT,
            physicsClientId=self._client,
        )) > 0)

        return np.array(
            torso + joint_pos + joint_vel + [l_contact, r_contact],
            dtype=np.float32,
        )  # 9 + 16 + 14 + 2 = 41
