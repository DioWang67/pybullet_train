"""
通用訓練 Callback 庫

所有機器人訓練共用的穩定且可配置的回調
"""

import time
from collections import deque
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


# ═══════════════════════════════════════════════════════════════════════════════
# VecNormalize 同步
# ═══════════════════════════════════════════════════════════════════════════════

class SyncVecNormCallback(BaseCallback):
    """
    同步訓練和評估環境的 VecNormalize 統計量
    
    確保評估環境使用與訓練環境相同的觀測和獎勵規範化
    """

    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize):
        """
        Args:
            train_env: 訓練向量化環境 (已配置 VecNormalize)
            eval_env: 評估向量化環境 (已配置 VecNormalize)
        """
        super().__init__(verbose=0)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        """每一步都同步規範化統計量"""
        self.eval_env.obs_rms = self.train_env.obs_rms
        self.eval_env.ret_rms = self.train_env.ret_rms
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# 訓練進度顯示
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingProgressCallback(BaseCallback):
    """
    實時訓練進度顯示
    
    每隔固定步數列印：
    - 當前步數和 episode 數
    - 平均獎勵 (最近 N episode)
    - 最佳獎勵
    - 平均 episode 長度
    - 運行速度 (FPS)
    - 預計剩餘時間
    """

    PRINT_FREQ = 10_000  # 每 10k 步列印一次

    def __init__(self, total_timesteps: int):
        """
        Args:
            total_timesteps: 訓練總步數
        """
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self._ep_rewards = deque(maxlen=20)
        self._ep_lengths = deque(maxlen=20)
        self._best_reward = -np.inf
        self._n_episodes = 0
        self._t_start = 0.0
        self._last_print = 0

    def _on_training_start(self) -> None:
        """訓練開始前初始化"""
        self._t_start = time.time()
        print(f"\n{'Steps':>12}  {'Eps':>6}  {'Mean-R(20)':>10}  "
              f"{'Best-R':>8}  {'MeanLen':>7}  {'FPS':>5}  {'ETA':>9}")
        print("─" * 72)

    def _on_step(self) -> bool:
        """每一步檢查是否達到列印條件"""
        # 收集 episode 信息
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self._ep_rewards.append(ep["r"])
                self._ep_lengths.append(ep["l"])
                self._n_episodes += 1
                if ep["r"] > self._best_reward:
                    self._best_reward = ep["r"]

        # 按時間間隔列印進度
        if self.num_timesteps - self._last_print >= self.PRINT_FREQ:
            self._last_print = self.num_timesteps
            elapsed = time.time() - self._t_start
            fps = int(self.num_timesteps / elapsed) if elapsed > 0 else 0
            remaining = (self.total_timesteps - self.num_timesteps) / fps if fps > 0 else 0
            mean_r = np.mean(self._ep_rewards) if self._ep_rewards else float("nan")
            mean_l = np.mean(self._ep_lengths) if self._ep_lengths else float("nan")
            best_r = self._best_reward if self._ep_rewards else float("nan")
            m, s = divmod(int(remaining), 60)
            h, m = divmod(m, 60)
            print(f"{self.num_timesteps:>12,}  {self._n_episodes:>6}  "
                  f"{mean_r:>10.1f}  {best_r:>8.1f}  "
                  f"{mean_l:>7.0f}  {fps:>5}  {h:02d}:{m:02d}:{s:02d}")
        return True

    def _on_training_end(self) -> None:
        """訓練結束後列印摘要"""
        elapsed = time.time() - self._t_start
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        print("─" * 72)
        print(f"訓練完成！時間={h:02d}:{m:02d}:{s:02d}  "
              f"最佳reward={self._best_reward:.1f}  "
              f"總回合={self._n_episodes}\n")
