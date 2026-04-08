# 配置選項詳解 — CONFIG_GUIDE

本指南詳細解釋所有配置參數的用途、範圍和響應效果。

---

## 快速參考表

| 配置段 | 參數 | 類型 | 默認值 | 範圍 | 影響 |
|--------|------|------|--------|------|------|
| **physics** | `physics_hz` | int | 500 | 240-1000 | 模擬精度 ↑ 計算 |
| | `substeps` | int | 5 | 1-10 | 精度 (計算 ×N) |
| | `gravity` | float | -9.81 | -9.81 | 重力加速度 |
| **robot** | `mass` | float | 55 | 30-100 | 慣性/穩定性 |
| | `max_torques` | dict | ... | 50-250 | 運動能力上限 |
| **reward** | `alive_bonus` | float | 2.0 | 0.1-10 | 鼓勵運動時長 |
| | `forward_reward_scale` | float | 1.5 | 0.1-5 | 前進速度重要性 |
| | `height_reward_scale` | float | 3.0 | 1-10 | 保持姿態的重要性 |
| | `smooth_penalty_scale` | float | 0.0001 | 0.00001-0.01 | 動作平滑度制約 |
| **SAC** | `learning_rate` | float | 3e-4 | 1e-5-1e-2 | 學習速度 |
| | `buffer_size` | int | 1000000 | 100k-10M | 記憶容量 |
| | `batch_size` | int | 256 | 32-1024 | 批量大小 |
| | `tau` | float | 0.005 | 0.001-0.05 | 目標網絡軟更新 |
| **training** | `n_envs` | int | 4 | 1-128 | 並行環境數 |
| | `total_timesteps` | int | 1000000 | 10k-10M | 訓練長度 |
| | `eval_freq` | int | 10000 | 1000-100k | 評估頻率 |

---

## 詳細參數解釋

### Physics 配置

#### `physics_hz` — 物理模擬頻率

```yaml
physics:
  physics_hz: 500  # Hz
```

**含義**: 物理引擎每秒更新次數

| 值 | 精度 | 計算 | 適用 | 說明 |
|----|------|------|------|------|
| 240 | ⭐⭐ | 快 | 低速機器人 | 步態簡單，動作緩慢 |
| 500 | ⭐⭐⭐ | 中 | 標准機器人 | **推薦**，平衡精度/計算 |
| 1000 | ⭐⭐⭐⭐⭐ | 慢 | 高精度需求 | 計算成本高 4-5 倍 |

**效果**:
- ↑ hz → 模擬更精確 (接觸檢測精細)，但訓練變慢
- ↓ hz → 訓練快但可能不穩定

**案例**:
```yaml
# Cassie (實驗室機器人)
physics_hz: 500

# H1 (快速反應)
physics_hz: 240  # 計算友好

# 精密機器人
physics_hz: 1000  # 高精度
```

#### `substeps` — 子步數

```yaml
physics:
  physics_hz: 500
  substeps: 5  # 每個 policy step 内部迭代次數
```

**含義**: 每個強化學習動作執行的物理步數

- **1 substep**: policy Hz = physics Hz
- **5 substeps**: policy Hz = physics Hz / 5

**效果**:
```
policy_hz = physics_hz / substeps
          = 500 / 5
          = 100 Hz  (10 ms/step)
```

**選擇指南**:
| substeps | 政策頻率 (500 Hz) | 適用 |
|----------|------------------|------|
| 1 | 500 Hz | 快速反饋，高計算 |
| 4 | 125 Hz | 標准運動 |
| 5 | 100 Hz | **推薦**，人類感知尺度 |
| 10 | 50 Hz | 低頻動作 |

**案例**:
```yaml
# 快速反應機器人
physics_hz: 500
substeps: 2  # 250 Hz 政策

# 標准行走
physics_hz: 500
substeps: 5  # 100 Hz 政策 (推薦)

# 平衡特化
physics_hz: 1000
substeps: 10  # 100 Hz 政策
```

---

### Robot 配置

#### `mass` — 總質量

```yaml
robot:
  mass: 55.0  # kg
```

**含義**: 機器人結構質量 (不含致動器)

**影響**:
- ↑ mass → 更穩定但動作慢
- ↓ mass → 更敏捷但容易摔倒

**選擇**:
| 機器人 | 質量 |
|--------|------|
| Cassie | 33-40 kg |
| H1 | 35-45 kg |
| 人類 | 70-80 kg |

**配置示例**:
```yaml
# 精准值 (查 URDF 或規格書)
robot:
  mass: 33.5  # Cassie v3

  # 或保守估計
  mass: 40.0  # Cassie v3 + 附件
```

#### `max_torques` — 最大扭矩限制

```yaml
robot:
  max_torques:
    joint_0: 150.0  # Nm (牛·米)
    joint_1: 150.0
    joint_2: 50.0
    # ... 所有活躍關節
```

**含義**: 各關節的最大輸出扭矩 (動作縮放)

**影響**:
- 動作在 [-1, 1] 縮放到 [-max_torque, max_torque]
- 過小 → 力量不足，無法完成任務
- 過大 → 過度施力，能量浪費

**選擇**:
```python
# policy 返回 action ∈ [-1, 1]
# 實際扭矩 = action * max_torque

# 例: max_torque = 150 Nm
torque = 0.5 * 150 = 75 Nm  # 50% 力量
torque = -1.0 * 150 = -150 Nm  # 完全反向
```

**配置建議**:
```yaml
# 查找 robot 規格書或 URDF 中 <limit effort="..."/>

robot:
  max_torques:
    # 髖部 (大肌肉, 高扭矩)
    hip_joint: 150.0
    
    # 膝蓋 (中等肌肉)
    knee_joint: 120.0
    
    # 踝部 (小肌肉)
    ankle_joint: 50.0
```

#### `stand_pose` — 平衡姿態

```yaml
robot:
  stand_pose:
    joint_0: 0.0
    joint_1: 0.5     # rad (弧度)
    joint_2: -1.0
    # ... 所有關節
```

**含義**: 機器人站立平衡時的關節角度

**影響**:
- 用於環境重置時穩定機器人
- 不正確 → 環境初始化時摔倒

**測量步驟**:

```python
import pybullet as p

# 手動調整機器人到平衡姿態
robot_id = p.loadURDF("robot.urdf")

# 通過試錯調整關節
for i in range(num_joints):
    p.resetJointState(robot_id, i, 0.5)  # 嘗試 0.5 rad

# 読取當前姿態
joint_angles = []
for i in range(num_joints):
    state = p.getJointState(robot_id, i)
    joint_angles.append(state[0])
    
print(joint_angles)  # 複製到 stand_pose
```

**案例**:
```yaml
# Cassie 典型平衡姿態
stand_pose:
  hip_roll: 0.0      # 無側傾
  hip_pitch: 0.5     # 前傾 ~30°
  knee: -1.0         # 膝蓋曲 ~60°
  ankle_pitch: -0.5  # 踝部後傾
  ankle_roll: 0.0    # 無內外翻
```

---

### Reward 配置

#### `alive_bonus` — 存活獎勵

```yaml
reward:
  alive_bonus: 2.0  # 每步獎勵
```

**含義**: 每個時步存活時獲得的固定獎勵

**效果**:
- t=0: +2.0
- t=1: +2.0
- t=1000: +2.0 (累計 2000!)
- t=摔倒: 0 (終止)

**用途**: 鼓勵機器人保持運行，避免快速失敗策略

**選擇指南**:
| 值 | 策略傾向 |
|----|---------|
| 0.1 | 激進，快速摔倒可接受 |
| 1.0 | 平衡 |
| 2.0 | **推薦**，穩定優先 |
| 5.0 | 保守，過度延長終止 |

**計算獎勵比例**:
```
動作獎勵 (前進等) 典型值: ~10/步
存活獎勵: ~2/步

總獎勵 ~ 12/步 * 1000步 = 12,000 (1000步軌跡)
```

#### `forward_reward_scale` — 前進獎勵權重

```yaml
reward:
  forward_reward_scale: 1.5
```

**含義**: 前進速度獎勵的縮放係數

**計算**:
```
forward_reward = forward_reward_scale * forward_speed
               = 1.5 * v_x

例: 走得 1 m/s → reward = 1.5
   走得 0.5 m/s → reward = 0.75
```

**選擇**:
| 值 | 效果 |
|----|------|
| 0.1 | 忽視前進 (不推薦) |
| 0.5 | 溫和鼓勵前進 |
| 1.5 | **推薦**，平衡 |
| 3.0 | 激進追求速度 (可能跌倒) |

**調試**:
- 若機器人不走路 → ↑ forward_reward_scale
- 若機器人跌倒衝刺 → ↓ forward_reward_scale

#### `height_reward_scale` — 高度獎勵權重

```yaml
reward:
  height_reward_scale: 3.0
```

**含義**: 保持軀幹高度的獎勵

**計算**:
```
height_reward = height_reward_scale * (h - h_min) / (h_max - h_min)

例: 目標高度 0.8m，實際高度 0.75m
   height_reward ≈ 3.0 * 0.25 (衰減)
```

**選擇**:
| 值 | 效果 |
|----|------|
| 1.0 | 低優先級 (容易駝背) |
| 3.0 | **推薦**，保持姿態 |
| 5.0+ | 高優先級 (過度僵硬) |

**調試**:
- 若機器人駝背/蹲下 → ↑ height_reward_scale
- 若軀幹過度刻板 → ↓ height_reward_scale

#### `smooth_penalty_scale` — 能量懲罰

```yaml
reward:
  smooth_penalty_scale: 0.0001
```

**含義**: 動作變化劇烈時的懲罰

**計算**:
```
smooth_penalty = smooth_penalty_scale * action²

例: action = 1.0 → penalty = 0.0001 * 1.0 = 0.0001
   action = 0.1 → penalty = 0.0001 * 0.01 = 0.000001
```

**用途**: 促進平滑、能量高效的運動

**選擇**:
| 值 | 效果 |
|----|------|
| 0 | 无懲罰 (生硬動作) |
| 0.0001 | **推薦**，微弱正則化 |
| 0.001 | 強正則化 (可能過度平滑) |
| 0.01 | 非常強 (基本無法移動) |

**調試**:
- 若動作生硬/振盪 → ↑ smooth_penalty_scale
- 若運動過度緩慢 → ↓ smooth_penalty_scale

#### `death_penalty` — 死亡懲罰

```yaml
reward:
  death_penalty: -30.0
```

**含義**: 機器人摔倒/失敗時的一次性懲罰

**計算**:
```
episode_reward = sum(step_rewards) + death_penalty_when_failed

例: 100 步，平均 reward 12/步 = 1200
    摔倒: 1200 - 30 = 1170 (最終佳績)
```

**選擇**:
| 值 | 效果 |
|----|------|
| -10 | 輕微懲罰 (激進策略) |
| -30 | **推薦**，不鼓勵摔倒 |
| -100 | 重懲罰 (過度保守) |

---

### SAC 訓練配置

#### `learning_rate` — 學習率

```yaml
sac:
  learning_rate: 3e-4  # 0.0003
```

**含義**: 策略和值網絡的梯度更新步長

**影響**:
- ↑ lr → 學習快，但不穩定
- ↓ lr → 學習穩定，但緩慢

**範圍**:
| 值 | 收斂 | 穩定性 | 計算時間 |
|----|------|----------|---------|
| 1e-5 | 極慢 | 高 | 1 個月 |
| 1e-4 | 慢 | 高 | 1-2 周 |
| 3e-4 | **推薦** | 中 | 1 周 |
| 1e-3 | 快 | 低 | 3 天 |
| 1e-2 | 很快 | 很低 | 1 天 (可能發散) |

**調試**:
```python
# tensorboard 監視損失曲線
tensorboard --logdir logs/cassie/

# 損失應該穩步下降，不是發散
```

#### `buffer_size` — 回放緩衝區大小

```yaml
sac:
  buffer_size: 1000000  # 1M 轉換
```

**含義**: 存儲過去經驗的容量

**影響**:
- ↑ size → 更多樣本，但內存 ↑
- ↓ size → 內存↓，但樣本多樣性↓

**選擇**:
| 值 | 內存 | 樣本多樣性 | 適用 |
|----|------|-----------|------|
| 100k | 100 MB | 低 | 小規模實驗 |
| 1M | 1 GB | **推薦** | 標准訓練 |
| 10M | 10 GB | 高 | 大規模集群 |

**計算**:
```
內存 ≈ buffer_size * (obs_dim + action_dim + 10) * 4 bytes

例: obs=31, action=10, 1M buffer
   內存 ≈ 1M * 51 * 4 ≈ 200 MB
```

#### `batch_size` — 批量大小

```yaml
sac:
  batch_size: 256  # 每次梯度更新的樣本數
```

**含義**: 訓練時從緩衝區采樣多少轉換

**影響**:
- ↑ size → 梯度估計更穩定，但計算↑
- ↓ size → 快速更新，但噪聲↑

**選擇**:
| 值 | 梯度估計 | 計算 | 適用 |
|----|---------|------|------|
| 32 | 嘈雜 | 快 | 小規模 |
| 64 | ^↑ | ^ | 中等 |
| 256 | **推薦** | 中 | 標准 |
| 512 | 穩定 | 慢 | 高精度 |
| 1024 | 很穩定 | 很慢 | 離線強化學習 |

#### `tau` — 目標網絡軟更新

```yaml
sac:
  tau: 0.005  # 軟更新比例
```

**含義**: 將訓練網絡權重複製到目標網絡的比例

**計算**:
```
target_params = tau * training_params + (1 - tau) * target_params

例: tau = 0.005
   每步目標网络吸收 0.5% 訓練進度
```

**影響**:
- ↑ tau → 目標快更新，可能不穩定
- ↓ tau → 目標緩更新，可能過時

**選擇**:
| 值 | 更新速度 | 穩定性 |
|----|---------|--------|
| 0.001 | 慢 | 高 |
| 0.005 | **推薦** | 中 |
| 0.01 | 快 | 低 |

---

### Training 配置

#### `n_envs` — 並行環境數

```yaml
training:
  n_envs: 4  # 同時運行 4 個環境
```

**含義**: DummyVecEnv 中的並行環境副本數

**影響**:
- ↑ n_envs → 數據採集快 (4 倍)，但內存↑
- ↓ n_envs → 內存↓，但訓練慢

**選擇**(基於 GPU/CPU):
| 機器 | n_envs | 內存 |
|------|--------|------|
| CPU (8核) | 4-8 | 4-8 GB |
| RTX 3060 | 16-32 | 8-12 GB |
| RTX 4090 | 64-128 | 20-24 GB |

**計算**:
```
total_memory ≈ n_envs * obs_size * buffer_batches

4 envs: ~100 MB
16 envs: ~400 MB
```

**調試**:
```bash
# 監視內存 (Linux)
watch -n 1 'ps aux | grep python | head -1'

# Windows 任務管理員
# 記錄最大內存使用
```

#### `total_timesteps` — 訓練總長度

```yaml
training:
  total_timesteps: 1000000  # 1M 步
```

**含義**: 訓練時環境交互總數

**選擇**:
| 值 | 時間 (4 envs) | 效果 |
|----|---------|--------|
| 10k | 1 分鐘 | 快速原型 |
| 100k | 10 分鐘 | 快速測試 |
| 1M | 2-3 小時 | **推薦** (初始訓練) |
| 10M | 1-2 天 | 精細調優 |

**計算**:
```
訓練時間 ≈ total_timesteps / (n_envs * fps)

例: 1M 步, 4 envs, 100 fps
   時間 ≈ 1M / (4 * 100) = 2500 秒oken ≈ 42 分鐘
```

#### `eval_freq` — 評估頻率

```yaml
training:
  callbacks:
    eval_freq: 10000  # 每 10k 步評估一次
```

**含義**: 多久進行一次 off-policy 評估

**影響**:
- ↑ freq → 更多評估數據(曲線平滑)，但計算↑
- ↓ freq → 計算↓，但評估稀疏

**選擇**:
| 值 | 評估次數 (1M 步) | 計算開銷 |
|----|---------|--------|
| 1k | 1000 組 | 高 |
| 10k | 100 組 | **推薦** |
| 50k | 20 組 | 低 |
| 100k | 10 組 | 很低 |

---

## 完整配置示例

### 示例 1: 快速原型 (快速迭代)

```yaml
# configs/training/fast.yaml
sac:
  learning_rate: 1e-3         # 快速學習
  buffer_size: 100000         # 小緩衝
  batch_size: 64              # 小批量

training:
  n_envs: 8                   # 更多並行
  total_timesteps: 100000     # 短訓練
  callbacks:
    eval_freq: 5000           # 頻繁評估
```

**特點**: 運行快 (30 分鐘)，適合實驗

### 示例 2: 生產訓練 (穩定優先)

```yaml
# configs/training/production.yaml
sac:
  learning_rate: 3e-4         # 謹慎學習
  buffer_size: 1000000        # 大緩衝
  batch_size: 256             # 大批量
  tau: 0.005

training:
  n_envs: 16                  # 中等並行
  total_timesteps: 5000000    # 長訓練
  callbacks:
    eval_freq: 20000          # 定期評估
```

**特點**: 穩定收斂 (1-2 天)，適合部署

### 示例 3: 超參調優 (極限性能)

```yaml
# configs/training/tuning.yaml
sac:
  learning_rate: 5e-4         # 中等
  buffer_size: 2000000        # 很大
  batch_size: 512             # 大批量
  tau: 0.001                  # 保守更新

training:
  n_envs: 32                  # 大規模並行
  total_timesteps: 10000000   # 很長
  callbacks:
    eval_freq: 50000          # 稀疏評估
```

**特點**: 超級優化 (2-3 天)，最高性能

---

## 調試技巧

### 技巧 1: 監視獎勵曲線

```bash
tensorboard --logdir logs/cassie/

# 在瀏覽器打開 http://localhost:6006
# 觀察:
#   - 獎勵應逐步增加
#   - 無突兀躍升或下降 (表示參數錯誤)
#   - 最終穩定在較高值
```

### 技巧 2: 快速原型流程

```bash
# 1. 快速測試配置 (5 分鐘)
python train.py --robot cassie --total_timesteps 10000

# 2. 檢查基本行為 (無崩潰)
# 3. 觀察獎勵曲線 (正確方向?)

# 4. 調整嚴重問題
python train.py --robot cassie \
    --sac__learning_rate 1e-3 \
    --total_timesteps 100000 \
    --render

# 5. 最終生產訓練
python train.py --robot cassie --total_timesteps 5000000
```

### 技巧 3: 參數靈敏度分析

```bash
# 對比學習率效果
PYWALKING_SAC__LEARNING_RATE=1e-4 python train.py --robot cassie &
PYWALKING_SAC__LEARNING_RATE=3e-4 python train.py --robot cassie &
PYWALKING_SAC__LEARNING_RATE=1e-3 python train.py --robot cassie &

# tensorboard 中並排比較
tensorboard --logdir logs/
```

---

## 常見錯誤

### 錯誤 1: 獎勵不增加

```
可能原因:
1. learning_rate 太小 → ↑ 10 倍
2. alive_bonus 太小 → ↑ 2 倍
3. 環境問題 (總是失敗) → 檢查 physics 配置

解決:
python train.py --sac__learning_rate 1e-3
```

### 錯誤 2: 機器人陷入局部最優 (只會走一個固定步態)

```
可能原因:
1. reward 權重比例不通吸引
2. forward_reward_scale 太大 → ↓ 0.5 倍
3. smooth_penalty_scale 太大 → ↓ 10 倍

解決:
python train.py --reward__forward_reward_scale 0.75 \
                 --reward__smooth_penalty_scale 0.00001
```

### 錯誤 3: 內存不足

```
可能原因:
1. n_envs 太大 → ↓ 一半
2. buffer_size 太大 → ↓ 一半

解決:
python train.py --n_envs 4 --sac__buffer_size 500000
```

---

## 相關文檔

- [ARCHITECTURE.md](ARCHITECTURE.md) — 系統設計
- [ADDING_ROBOT.md](ADDING_ROBOT.md) — 添加新機器人
- [../train.py](../train.py) — 訓練腳本

---

**版本**: Phase 4 v1.0  
**最後更新**: 2026-04-01
