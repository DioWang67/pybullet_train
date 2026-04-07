"""
載入 HuggingFace 上的預訓練模型跑 PyBullet 走路模擬

環境選項（任選一個填入 ENV_ID）：
  Walker2DBulletEnv-v0         - 2D 雙足行走，最穩定
  HumanoidBulletEnv-v0         - 全身人形，較複雜
  HopperBulletEnv-v0           - 單腳跳躍
  HalfCheetahBulletEnv-v0      - 4腳獵豹

HuggingFace 模型對應：
  sb3/sac-Walker2DBulletEnv-v0
  sb3/sac-HumanoidBulletEnv-v0
  sb3/sac-HopperBulletEnv-v0
  sb3/sac-HalfCheetahBulletEnv-v0
"""

import time
import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
import pybullet_envs_gymnasium  # 注册 Bullet 環境
from stable_baselines3 import SAC
from huggingface_sb3 import load_from_hub

# ── 設定 ──────────────────────────────────────────────────────────────────────

ENV_ID    = "Walker2DBulletEnv-v0"          # 改這裡換機器人
REPO_ID   = f"sb3/sac-{ENV_ID}"            # HuggingFace 模型路徑
FILENAME  = f"sac-{ENV_ID}.zip"
EPISODES  = 5                               # 要跑幾個回合

# ── 下載並載入模型 ────────────────────────────────────────────────────────────

print(f"Downloading model: {REPO_ID}")
model_path = load_from_hub(repo_id=REPO_ID, filename=FILENAME)
model      = SAC.load(model_path, custom_objects={
    "learning_rate": 3e-4,
    "lr_schedule": lambda _: 3e-4,
})
print("Model loaded.\n")

# ── 觀測維度補齊 Wrapper ────────────────────────────────────────────────────────
# 預訓練模型用舊版 gym（obs=23），pybullet_envs_gymnasium 版本為 22，補一個 0 對齊

class PadObservationTo23(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_low  = np.append(env.observation_space.low,  -np.inf).astype(np.float64)
        obs_high = np.append(env.observation_space.high,  np.inf).astype(np.float64)
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float64)

    def observation(self, obs):
        return np.append(obs, 0.0).astype(np.float32)


# ── 建立環境（render_mode="human" 開 GUI 視窗） ───────────────────────────────

env = PadObservationTo23(gym.make(ENV_ID, render_mode="human"))

# ── 執行 ──────────────────────────────────────────────────────────────────────

for ep in range(1, EPISODES + 1):
    obs, _  = env.reset()
    done    = False
    total_r = 0.0
    steps   = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_r += reward
        steps   += 1
        done     = terminated or truncated
        time.sleep(1 / 60)  # 限制 60fps，讓動畫看起來流暢

    print(f"Episode {ep:2d} | steps={steps:4d} | reward={total_r:.1f}")

input("\n所有回合結束，按 Enter 關閉視窗...")
env.close()
print("Done.")
