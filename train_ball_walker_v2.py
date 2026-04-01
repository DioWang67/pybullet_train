"""
訓練球體雙足機器人走路  v2
使用 SAC (Soft Actor-Critic)

執行：
  python train_ball_walker_v2.py

訓練完成後 render：
  python train_ball_walker_v2.py --render
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from envs.ball_walker_env import BallWalkerEnv

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
LOG_DIR   = os.path.join(os.path.dirname(__file__), "logs", "ball_walker")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

MODEL_PATH   = os.path.join(MODEL_DIR, "ball_walker_sac")
VECNORM_PATH = os.path.join(MODEL_DIR, "ball_walker_vecnorm.pkl")


# ── VecNormalize 同步 Callback ────────────────────────────────────────────────
class SyncVecNormCallback(BaseCallback):
    """
    每次 EvalCallback 執行 evaluate_policy 之前，
    將訓練環境的 obs_rms / ret_rms 同步到 eval_env，
    確保 agent 在 eval 時看到的 obs 分布與訓練一致。

    用法：
        eval_cb = EvalCallback(..., callback_before_eval=SyncVecNormCallback())
    """

    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize):
        super().__init__(verbose=0)
        self.train_env = train_env
        self.eval_env  = eval_env

    def _on_step(self) -> bool:
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms
        return True


# ── 訓練進度 Callback ─────────────────────────────────────────────────────────
class TrainingProgressCallback(BaseCallback):
    """
    每 PRINT_FREQ steps 印一行訓練狀態：
      步數進度 / 回合數 / 平均 reward(近20回合) / 最佳 reward / 平均步數 / FPS / 剩餘時間
    """
    PRINT_FREQ = 10_000

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self._ep_rewards: deque = deque(maxlen=20)
        self._ep_lengths: deque = deque(maxlen=20)
        self._best_reward = -np.inf
        self._n_episodes  = 0
        self._t_start     = 0.0
        self._last_print  = 0

    def _on_training_start(self) -> None:
        self._t_start = time.time()
        print(
            f"\n{'步數':>12}  {'回合':>6}  {'近20回均獎':>10}  "
            f"{'最佳獎勵':>9}  {'均步數':>7}  {'FPS':>5}  {'剩餘時間':>9}"
        )
        print("─" * 72)

    def _on_step(self) -> bool:
        # Monitor 在 episode 結束時會在 infos 裡放 "episode" key
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._ep_rewards.append(ep["r"])
                self._ep_lengths.append(ep["l"])
                self._n_episodes += 1
                if ep["r"] > self._best_reward:
                    self._best_reward = ep["r"]

        if self.num_timesteps - self._last_print >= self.PRINT_FREQ:
            self._last_print = self.num_timesteps
            elapsed  = time.time() - self._t_start
            fps      = int(self.num_timesteps / elapsed) if elapsed > 0 else 0
            remaining = (self.total_timesteps - self.num_timesteps) / fps if fps > 0 else 0
            mean_r   = np.mean(self._ep_rewards) if self._ep_rewards else float("nan")
            mean_l   = np.mean(self._ep_lengths) if self._ep_lengths else float("nan")
            best_r   = self._best_reward if self._ep_rewards else float("nan")

            mins, secs = divmod(int(remaining), 60)
            hrs,  mins = divmod(mins, 60)
            remain_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

            print(
                f"{self.num_timesteps:>12,}  {self._n_episodes:>6}  "
                f"{mean_r:>10.1f}  {best_r:>9.1f}  "
                f"{mean_l:>7.0f}  {fps:>5}  {remain_str:>9}"
            )
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self._t_start
        mins, secs = divmod(int(elapsed), 60)
        hrs,  mins = divmod(mins, 60)
        print("─" * 72)
        print(f"訓練完成！總時間 {hrs:02d}:{mins:02d}:{secs:02d}  "
              f"最佳 reward = {self._best_reward:.1f}  "
              f"總回合數 = {self._n_episodes}\n")


# ── 工廠函式 ──────────────────────────────────────────────────────────────────
def make_env():
    """訓練用：包 Monitor 以便 TrainingProgressCallback 讀取回合統計"""
    return Monitor(BallWalkerEnv())


def make_eval_env():
    """
    Eval 用：包 Monitor，讓 EvalCallback 能正確讀 episode reward/length。
    注意：Monitor 必須在 VecNormalize 之前包，否則 reward 會是 normalized 值。
    """
    return Monitor(BallWalkerEnv())


# ── 訓練主程式 ────────────────────────────────────────────────────────────────
def train():
    # ── 環境檢查 ──────────────────────────────────────────────────────────────
    print("Checking environment...")
    test_env = BallWalkerEnv()
    check_env(test_env, warn=True)
    test_env.close()
    print("Environment OK.\n")

    # ── 訓練環境（8 個並行 + normalize）─────────────────────────────────────
    n_envs  = 4   # 降低記憶體用量（原本 8 個 PyBullet 進程可能 OOM）
    vec_env = SubprocVecEnv([make_env] * n_envs)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    # ── Eval 環境（單一，obs normalize，reward 不 normalize）─────────────────
    # ⚠️  eval_env 的 VecNormalize 統計初始為空，
    #     必須透過 SyncVecNormCallback 在每次 eval 前從 vec_env 同步。
    eval_vec = DummyVecEnv([make_eval_env])
    eval_vec = VecNormalize(
        eval_vec,
        norm_obs=True,
        norm_reward=False,  # eval reward 要看原始值
        clip_obs=10.0,
        training=False,     # eval 期間不更新統計，避免污染
    )

    # ── SAC 模型 ──────────────────────────────────────────────────────────────
    # v2 URDF 多了 2 個 abduction joints，action/obs 空間自動由 env 決定
    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=1_500_000,   # 動作空間變大，buffer 稍微放大
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
    )

    # ── 訓練 ──────────────────────────────────────────────────────────────────
    total_timesteps = 5_000_000

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=MODEL_DIR,
        name_prefix="ball_walker",
    )

    sync_cb     = SyncVecNormCallback(train_env=vec_env, eval_env=eval_vec)
    progress_cb = TrainingProgressCallback(total_timesteps=total_timesteps)

    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    print(f"Training for {total_timesteps:,} timesteps with {n_envs} parallel envs (SubprocVecEnv)...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[sync_cb, progress_cb, checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # ── 儲存 ──────────────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    vec_env.save(VECNORM_PATH)
    print(f"\nModel saved      → {MODEL_PATH}.zip")
    print(f"VecNormalize saved → {VECNORM_PATH}")

    vec_env.close()
    eval_vec.close()


# ── Render 已訓練模型 ─────────────────────────────────────────────────────────
def render_trained():
    """載入訓練好的模型並用 GUI 播放（Ctrl+C 隨時退出）"""
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"找不到模型：{MODEL_PATH}.zip")
        print("請先執行訓練。")
        return

    model = SAC.load(MODEL_PATH)
    print("Model loaded.\n")

    # 用獨立 dummy env 載入 obs 正規化統計
    vec_norm = None
    if os.path.exists(VECNORM_PATH):
        _dummy = DummyVecEnv([lambda: BallWalkerEnv()])
        vec_norm = VecNormalize.load(VECNORM_PATH, _dummy)
        vec_norm.training = False
        vec_norm.norm_reward = False
        print(f"VecNormalize loaded from {VECNORM_PATH}\n")

    env = BallWalkerEnv(render_mode="human")

    print("Running 5 episodes... (Ctrl+C to quit)\n")
    try:
        for ep in range(1, 6):
            obs, _ = env.reset()
            done    = False
            total_r = 0.0
            steps   = 0

            while not done:
                # 正規化觀測（與訓練時一致）
                if vec_norm is not None:
                    obs_in = vec_norm.normalize_obs(obs.reshape(1, -1))[0]
                else:
                    obs_in = obs

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
        if vec_norm is not None:
            vec_norm.venv.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--render" in sys.argv:
        render_trained()
    else:
        train()
