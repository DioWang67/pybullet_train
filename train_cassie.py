"""
Cassie 走路訓練腳本
使用 SAC (Soft Actor-Critic)

執行訓練：
  python train_cassie.py

看結果：
  python train_cassie.py --render

繼續訓練：
  python train_cassie.py --resume
"""

import sys
import os
import time
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)

from envs.cassie_env import CassieEnv

# ── 路徑 ──────────────────────────────────────────────────────────────────────
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "models", "cassie")
LOG_DIR      = os.path.join(os.path.dirname(__file__), "logs",   "cassie")
MODEL_PATH   = os.path.join(MODEL_DIR, "cassie_sac")
VECNORM_PATH = os.path.join(MODEL_DIR, "cassie_vecnorm.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)


# ── VecNormalize 同步 ─────────────────────────────────────────────────────────
class SyncVecNormCallback(BaseCallback):
    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize):
        super().__init__(verbose=0)
        self.train_env = train_env
        self.eval_env  = eval_env

    def _on_step(self) -> bool:
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms
        return True


# ── 訓練進度顯示 ──────────────────────────────────────────────────────────────
class TrainingProgressCallback(BaseCallback):
    PRINT_FREQ = 10_000

    def __init__(self, total_timesteps: int):
        super().__init__(verbose=0)
        self.total_timesteps  = total_timesteps
        self._ep_rewards: deque = deque(maxlen=20)
        self._ep_lengths: deque = deque(maxlen=20)
        self._best_reward = -np.inf
        self._n_episodes  = 0
        self._t_start     = 0.0
        self._last_print  = 0

    def _on_training_start(self) -> None:
        self._t_start = time.time()
        print(f"\n{'Steps':>12}  {'Eps':>6}  {'Mean-R(20)':>10}  "
              f"{'Best-R':>8}  {'MeanLen':>7}  {'FPS':>5}  {'ETA':>9}")
        print("─" * 72)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self._ep_rewards.append(ep["r"])
                self._ep_lengths.append(ep["l"])
                self._n_episodes += 1
                if ep["r"] > self._best_reward:
                    self._best_reward = ep["r"]

        if self.num_timesteps - self._last_print >= self.PRINT_FREQ:
            self._last_print = self.num_timesteps
            elapsed   = time.time() - self._t_start
            fps       = int(self.num_timesteps / elapsed) if elapsed > 0 else 0
            remaining = (self.total_timesteps - self.num_timesteps) / fps if fps > 0 else 0
            mean_r    = np.mean(self._ep_rewards) if self._ep_rewards else float("nan")
            mean_l    = np.mean(self._ep_lengths) if self._ep_lengths else float("nan")
            best_r    = self._best_reward if self._ep_rewards else float("nan")
            m, s = divmod(int(remaining), 60)
            h, m = divmod(m, 60)
            print(f"{self.num_timesteps:>12,}  {self._n_episodes:>6}  "
                  f"{mean_r:>10.1f}  {best_r:>8.1f}  "
                  f"{mean_l:>7.0f}  {fps:>5}  {h:02d}:{m:02d}:{s:02d}")
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self._t_start
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        print("─" * 72)
        print(f"訓練完成！時間={h:02d}:{m:02d}:{s:02d}  "
              f"最佳reward={self._best_reward:.1f}  "
              f"總回合={self._n_episodes}\n")


# ── 環境工廠 ──────────────────────────────────────────────────────────────────
def make_env():
    return Monitor(CassieEnv())


# ── 訓練 ──────────────────────────────────────────────────────────────────────
def train(resume: bool = False):
    print("Checking environment...")
    test_env = CassieEnv()
    check_env(test_env, warn=True)
    test_env.close()
    print("Environment OK.\n")

    n_envs  = 8
    vec_env = DummyVecEnv([make_env] * n_envs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_vec = DummyVecEnv([make_env])
    eval_vec = VecNormalize(
        eval_vec, norm_obs=True, norm_reward=False,
        clip_obs=10.0, training=False,
    )

    total_timesteps = 5_000_000

    if resume and os.path.exists(MODEL_PATH + ".zip") and os.path.exists(VECNORM_PATH):
        print(f"Resuming from {MODEL_PATH}.zip ...")
        vec_env = VecNormalize.load(VECNORM_PATH, vec_env.venv)
        vec_env.training = True
        model = SAC.load(
            MODEL_PATH, env=vec_env,
            custom_objects={"learning_rate": 3e-4, "lr_schedule": lambda _: 3e-4},
        )
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            # 較大的網路：Cassie 動作空間比球體機器人複雜
            policy_kwargs={"net_arch": [400, 300]},
            verbose=1,
        )

    sync_cb     = SyncVecNormCallback(train_env=vec_env, eval_env=eval_vec)
    progress_cb = TrainingProgressCallback(total_timesteps=total_timesteps)
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=MODEL_DIR,
        name_prefix="cassie",
    )
    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=50_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    print(f"Training {total_timesteps:,} steps with {n_envs} envs...\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[sync_cb, progress_cb, checkpoint_cb, eval_cb],
        progress_bar=True,
        reset_num_timesteps=not resume,
    )

    model.save(MODEL_PATH)
    vec_env.save(VECNORM_PATH)
    print(f"\nModel saved → {MODEL_PATH}.zip")
    print(f"VecNormalize → {VECNORM_PATH}")

    vec_env.close()
    eval_vec.close()


# ── Render ─────────────────────────────────────────────────────────────────────
def render_trained():
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"找不到模型：{MODEL_PATH}.zip，請先訓練。")
        return

    model = SAC.load(MODEL_PATH)

    vec_norm = None
    if os.path.exists(VECNORM_PATH):
        _dummy = DummyVecEnv([lambda: CassieEnv()])
        vec_norm = VecNormalize.load(VECNORM_PATH, _dummy)
        vec_norm.training    = False
        vec_norm.norm_reward = False
        print("VecNormalize loaded.\n")

    env = CassieEnv(render_mode="human")
    print("Running 5 episodes... (Ctrl+C to quit)\n")

    try:
        for ep in range(1, 6):
            obs, _ = env.reset()
            done    = False
            total_r = 0.0
            steps   = 0

            while not done:
                obs_in = vec_norm.normalize_obs(obs.reshape(1, -1))[0] \
                    if vec_norm else obs
                action, _ = model.predict(obs_in, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_r += reward
                steps   += 1
                done     = terminated or truncated
                time.sleep(1 / 60)

            print(f"Episode {ep:2d} | steps={steps:4d} | reward={total_r:.1f}")

    except KeyboardInterrupt:
        print("\n中斷。")
    finally:
        env.close()
        if vec_norm:
            vec_norm.venv.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--render" in sys.argv:
        render_trained()
    elif "--resume" in sys.argv:
        train(resume=True)
    else:
        train(resume=False)
