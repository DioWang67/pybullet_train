"""
Unitree H1 人形機器人走路訓練
使用 SAC (Soft Actor-Critic)

d:/miniconda/envs/pybullet_env/python.exe train_h1.py   --render
執行訓練：  python train_h1.py
看結果：    python train_h1.py --render
繼續訓練：  python train_h1.py --resume
"""

import glob
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.h1_env import H1Env

# ── 路徑 ──────────────────────────────────────────────────────────────────────
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "models", "h1")
LOG_DIR      = os.path.join(os.path.dirname(__file__), "logs",   "h1")
MODEL_PATH   = os.path.join(MODEL_DIR, "h1_sac")       # 最終模型
VECNORM_PATH = os.path.join(MODEL_DIR, "h1_vecnorm.pkl")  # 最終 VecNormalize
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# ── 超參數 ────────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_500_000
TRAIN_FREQ      = 1
GRADIENT_STEPS  = 1
BUFFER_SIZE     = 300_000
BATCH_SIZE      = 512
LEARNING_RATE   = 3e-4
LEARNING_STARTS = 20_000
TAU             = 0.005
GAMMA           = 0.99
ENT_COEF        = "auto_0.05"
SAVE_FREQ       = 100_000   # 每 100K 步存一次 checkpoint


# ── Callbacks ─────────────────────────────────────────────────────────────────

class SyncVecNormCallback(BaseCallback):
    def __init__(self, train_env, eval_env):
        super().__init__(0)
        self.train_env = train_env
        self.eval_env  = eval_env

    def _on_step(self) -> bool:
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms
        return True


class SaveVecNormCallback(BaseCallback):
    """每次 checkpoint 同步儲存 VecNormalize 統計量"""

    def __init__(self, vec_env: VecNormalize, save_path: str, save_freq: int):
        super().__init__(0)
        self.vec_env   = vec_env
        self.save_path = save_path
        self.save_freq = save_freq
        self._last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.save_freq:
            self._last_save = self.num_timesteps
            path = os.path.join(
                self.save_path,
                f"h1_vecnorm_{self.num_timesteps}_steps.pkl",
            )
            self.vec_env.save(path)
        return True


class TrainingProgressCallback(BaseCallback):
    PRINT_FREQ = 10_000

    def __init__(self, total_timesteps: int):
        super().__init__(0)
        self.total_timesteps = total_timesteps
        self._ep_rewards: deque = deque(maxlen=20)
        self._best = -np.inf
        self._n    = 0
        self._t0   = 0.0
        self._last = 0

    def _on_training_start(self) -> None:
        self._t0 = time.time()
        print(f"\n{'Steps':>12}  {'Eps':>6}  {'MeanR(20)':>10}  "
              f"{'BestR':>8}  {'FPS':>5}  {'ETA':>9}")
        print("─" * 62)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self._ep_rewards.append(ep["r"])
                self._n += 1
                if ep["r"] > self._best:
                    self._best = ep["r"]
        if self.num_timesteps - self._last >= self.PRINT_FREQ:
            self._last  = self.num_timesteps
            elapsed     = time.time() - self._t0
            fps         = int(self.num_timesteps / elapsed) if elapsed > 0 else 0
            remaining   = (self.total_timesteps - self.num_timesteps) / fps if fps > 0 else 0
            mean_r      = np.mean(self._ep_rewards) if self._ep_rewards else float("nan")
            m, s = divmod(int(remaining), 60)
            h, m = divmod(m, 60)
            print(f"{self.num_timesteps:>12,}  {self._n:>6}  {mean_r:>10.1f}  "
                  f"{self._best:>8.1f}  {fps:>5}  {h:02d}:{m:02d}:{s:02d}")
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self._t0
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        print("─" * 62)
        print(f"完成！時間={h:02d}:{m:02d}:{s:02d}  "
              f"最佳={self._best:.1f}  回合={self._n}\n")


# ── Resume 工具 ───────────────────────────────────────────────────────────────

def find_latest_checkpoint() -> tuple[str | None, str | None]:
    """找最新的 checkpoint 和對應的 VecNormalize"""
    ckpts = sorted(glob.glob(os.path.join(MODEL_DIR, "h1_*_steps.zip")))
    if not ckpts:
        # 嘗試最終模型
        if os.path.exists(MODEL_PATH + ".zip"):
            return MODEL_PATH, VECNORM_PATH if os.path.exists(VECNORM_PATH) else None
        return None, None

    latest_ckpt = ckpts[-1]

    # 找對應的 VecNormalize（同步存的那個）
    # 格式: h1_100000_steps.zip -> h1_vecnorm_100000_steps.pkl
    step_str = os.path.basename(latest_ckpt).replace("h1_", "").replace("_steps.zip", "")
    vecnorm  = os.path.join(MODEL_DIR, f"h1_vecnorm_{step_str}_steps.pkl")
    if not os.path.exists(vecnorm):
        vecnorm = VECNORM_PATH if os.path.exists(VECNORM_PATH) else None

    return latest_ckpt.replace(".zip", ""), vecnorm


# ── 訓練 ──────────────────────────────────────────────────────────────────────

def make_env():
    return Monitor(H1Env())


def train(resume: bool = False):
    vec_env  = DummyVecEnv([make_env])
    vec_env  = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_vec = DummyVecEnv([make_env])
    eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    if resume:
        ckpt_path, vn_path = find_latest_checkpoint()
        if ckpt_path is None:
            print("找不到任何 checkpoint，從頭開始訓練。")
            resume = False
        else:
            print(f"從 {ckpt_path}.zip 恢復訓練...")
            if vn_path:
                print(f"載入 VecNormalize: {vn_path}")
                vec_env = VecNormalize.load(vn_path, vec_env.venv)
                vec_env.training = True
            model = SAC.load(
                ckpt_path, env=vec_env,
                custom_objects={
                    "learning_rate": LEARNING_RATE,
                    "lr_schedule": lambda _: LEARNING_RATE,
                    "train_freq": TRAIN_FREQ,
                    "gradient_steps": GRADIENT_STEPS,
                    "learning_starts": LEARNING_STARTS,
                    "tau": TAU,
                    "gamma": GAMMA,
                    "ent_coef": ENT_COEF,
                },
            )
            print(f"已完成步數: {model.num_timesteps:,}\n")

    if not resume:
        model = SAC(
            "MlpPolicy", vec_env,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            learning_starts=LEARNING_STARTS,
            tau=TAU,
            gamma=GAMMA,
            ent_coef=ENT_COEF,
            train_freq=TRAIN_FREQ,
            gradient_steps=GRADIENT_STEPS,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=1,
        )

    remaining = TOTAL_TIMESTEPS - model.num_timesteps
    print(f"剩餘訓練步數: {remaining:,} / {TOTAL_TIMESTEPS:,}\n")

    model.learn(
        total_timesteps=remaining,
        callback=[
            SyncVecNormCallback(vec_env, eval_vec),
            SaveVecNormCallback(vec_env, MODEL_DIR, SAVE_FREQ),
            TrainingProgressCallback(remaining),
            CheckpointCallback(
                save_freq=SAVE_FREQ, save_path=MODEL_DIR, name_prefix="h1",
            ),
            EvalCallback(
                eval_vec, best_model_save_path=MODEL_DIR, log_path=LOG_DIR,
                eval_freq=50_000, n_eval_episodes=5, deterministic=True, verbose=1,
            ),
        ],
        progress_bar=True,
        reset_num_timesteps=not resume,
    )

    model.save(MODEL_PATH)
    vec_env.save(VECNORM_PATH)
    print(f"模型已存至 {MODEL_PATH}.zip")
    vec_env.close()
    eval_vec.close()


# ── 渲染測試 ──────────────────────────────────────────────────────────────────

def render_trained():
    # 優先用 EvalCallback 存的最佳模型
    best = os.path.join(MODEL_DIR, "best_model.zip")
    if os.path.exists(best):
        ckpt_path = os.path.join(MODEL_DIR, "best_model")
        vn_path   = VECNORM_PATH if os.path.exists(VECNORM_PATH) else None
        print(f"載入最佳模型: {best}")
    else:
        ckpt_path, vn_path = find_latest_checkpoint()
        if ckpt_path is None:
            print("找不到模型，請先訓練。")
            return
        print(f"載入模型: {ckpt_path}.zip")
    model = SAC.load(ckpt_path)

    vec_norm = None
    if vn_path:
        _dummy   = DummyVecEnv([lambda: H1Env()])
        vec_norm = VecNormalize.load(vn_path, _dummy)
        vec_norm.training    = False
        vec_norm.norm_reward = False

    env = H1Env(render_mode="human")
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


if __name__ == "__main__":
    if "--render" in sys.argv:
        render_trained()
    elif "--resume" in sys.argv:
        train(resume=True)
    else:
        train()
