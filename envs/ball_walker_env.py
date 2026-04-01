"""
BallWalkerEnv — 球體軀幹雙足機器人走路環境
- 物理：PyBullet，240Hz；策略：60Hz（每 policy step = 4 substeps）
- 觀測(23)：軀幹高度/pitch/roll/線速度(3)/角速度(3) + 6關節角 + 6關節速度 + 2腳底接觸
- 動作(6)：6個關節的標準化扭矩 [-1, 1]
  順序：left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle
- 獎勵：前進速度 + 存活加分 − 能量懲罰
"""

import os
import math
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


# ── 常數 ─────────────────────────────────────────────────────────────────────

URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "robots", "ball_walker.urdf")

PHYSICS_HZ   = 240
POLICY_HZ    = 60
SUBSTEPS     = PHYSICS_HZ // POLICY_HZ   # 4

MAX_STEPS    = 1000
TORSO_MIN_Z  = 0.30   # 低於此高度視為倒地

# 各關節最大輸出扭矩 (Nm)，與 URDF effort 對應
MAX_TORQUE = np.array([150., 100., 60., 150., 100., 60.], dtype=np.float32)

# 關節 index（按 URDF 順序：left_hip=0 … right_ankle=5）
IDX_L_HIP, IDX_L_KNEE, IDX_L_ANKLE = 0, 1, 2
IDX_R_HIP, IDX_R_KNEE, IDX_R_ANKLE = 3, 4, 5

# 對應 link index（PyBullet 中 child link index = joint index）
LINK_L_FOOT = 2   # left_ankle 的 child
LINK_R_FOOT = 5   # right_ankle 的 child

# 初始生成高度（球心）：球半徑 + 大腿 + 小腿 + 腳半高 = 0.25+0.15+0.15+0.04 = 0.59
# 略高讓機器人自然沉降
SPAWN_Z = 0.65


class BallWalkerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": POLICY_HZ}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode   = render_mode
        self._client: int  = -1
        self._robot: int   = -1
        self._plane: int   = -1
        self._step_count   = 0

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        obs_limit = np.full(23, np.inf, dtype=np.float32)
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
            numSolverIterations=50, physicsClientId=self._client
        )

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane = p.loadURDF("plane.urdf", physicsClientId=self._client)
        p.changeDynamics(
            self._plane, -1,
            lateralFriction=1.5, restitution=0.0,
            physicsClientId=self._client,
        )

        self._robot = p.loadURDF(
            URDF_PATH,
            basePosition=[0.0, 0.0, SPAWN_Z],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self._client,
        )
        p.changeDynamics(
            self._robot, LINK_L_FOOT, lateralFriction=1.5,
            physicsClientId=self._client,
        )
        p.changeDynamics(
            self._robot, LINK_R_FOOT, lateralFriction=1.5,
            physicsClientId=self._client,
        )

        # 關閉預設位置馬達，切換為扭矩控制
        for j in range(6):
            p.setJointMotorControl2(
                self._robot, j,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self._client,
            )

        self._step_count = 0

        # 沉降 0.2 秒，讓機器人站穩再開始
        for _ in range(int(0.2 * PHYSICS_HZ)):
            p.stepSimulation(physicsClientId=self._client)

        # render 模式：每次 reset 後把相機拉回機器人頭頂
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=2.5, cameraYaw=30, cameraPitch=-20,
                cameraTargetPosition=[0.0, 0.0, 0.5],
                physicsClientId=self._client,
            )

        return self._observe(), {}

    def step(self, action: np.ndarray):
        torques = np.clip(action, -1.0, 1.0) * MAX_TORQUE

        for j, tau in enumerate(torques):
            p.setJointMotorControl2(
                self._robot, j,
                controlMode=p.TORQUE_CONTROL,
                force=float(tau),
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

        alive_bonus    = 2.0 if not terminated else 0.0   # 活著就給獎勵
        height_reward  = (pos[2] - TORSO_MIN_Z) * 3.0    # 站越高越好
        forward_reward = lin_vel[0] * 1.0                 # 鼓勵前進
        energy_penalty = 0.0005 * float(np.sum(torques ** 2))  # 輕微懲罰耗能
        death_penalty  = -20.0 if terminated else 0.0     # 倒下重罰

        reward = alive_bonus + height_reward + forward_reward - energy_penalty + death_penalty

        if self.render_mode == "human":
            self._follow_camera(pos)

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass  # GUI 模式下 PyBullet 自動渲染

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
            p.resetDebugVisualizerCamera(
                cameraDistance=2.5, cameraYaw=30, cameraPitch=-20,
                cameraTargetPosition=[0, 0, 0.5],
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

        torso_obs = [
            pos[2],         # 高度
            pitch,          # pitch（前傾/後仰）
            roll,           # roll（左傾/右傾）
            lin_vel[0],     # 前進速度 x
            lin_vel[1],     # 側向速度 y
            lin_vel[2],     # 垂直速度 z
            ang_vel[0],     # 角速度 roll
            ang_vel[1],     # 角速度 pitch
            ang_vel[2],     # 角速度 yaw
        ]  # 9 values

        joint_pos = []
        joint_vel = []
        for j in range(6):
            jstate = p.getJointState(
                self._robot, j, physicsClientId=self._client
            )
            joint_pos.append(jstate[0])  # 角度
            joint_vel.append(jstate[1])  # 角速度
        # 12 values

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
        # 2 values — 共 23

        return np.array(
            torso_obs + joint_pos + joint_vel + [l_contact, r_contact],
            dtype=np.float32,
        )

    def _follow_camera(self, pos):
        p.resetDebugVisualizerCamera(
            cameraDistance=2.5, cameraYaw=30, cameraPitch=-20,
            cameraTargetPosition=[pos[0], pos[1], 0.5],
            physicsClientId=self._client,
        )
