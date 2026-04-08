# 如何添加新機器人 — 完整指南

本指南展示如何添加一個新機器人到框架中，支持訓練和評估。

---

## 前置要求

✅ 已安裝依賴:
```bash
pip install PyYAML pybullet stable-baselines3 gymnasium
```

✅ 了解基本概念:
- Gymnasium 環境 API (`reset()`, `step()`, `render()`)
- PyBullet 基本操作 (加載 URDF, 獲取 JointState)
- 機器人的物理參數 (質量、摩擦、關節限制)

---

## 步驟 1: 準備機器人 URDF 文件

### 位置
```
robots/
└── your_robot.urdf    # 機器人定義 (或通過 robot_descriptions 包加載)
```

### 內容檢查清單

在 URDF 文件中確保有：

- ✅ 正確的質量和慣性張量
- ✅ 所有關節已定義 (`<joint>` 元素)
- ✅ 活躍關節 (可驅動的關節)
- ✅ 腳部接觸點 (末端執行器或接觸傳感器)
- ✅ 正確的軸向 (Z 軸向上)

**示例 URDF 片段**:
```xml
<?xml version="1.0"?>
<robot name="your_robot">
  <!-- 基座 -->
  <link name="torso">
    <mass value="30.0"/>
    <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
  </link>
  
  <!-- 左腿關節 -->
  <joint name="left_hip" type="revolute">
    <parent link="torso"/>
    <child link="left_hip_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="150" velocity="5"/>
  </joint>
  
  <!-- ... 更多關節 ... -->
  
  <!-- 腳接觸點 -->
  <link name="left_foot"/>
</robot>
```

---

## 步驟 2: 測量/提取機器人參數

運行 PyBullet 檢查腳本以獲取機器人信息：

```python
import pybullet as p
import pybullet_data

# 創建物理引擎
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 加載機器人
robot_id = p.loadURDF("robots/your_robot.urdf")

# 獲取所有關節
num_joints = p.getNumJoints(robot_id)
print(f"總關節數: {num_joints}")

for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]  # REVOLUTE=0, PRISMATIC=1, FIXED=4
    lower_limit = joint_info[8]
    upper_limit = joint_info[9]
    max_force = joint_info[10]
    max_velocity = joint_info[11]
    
    print(f"  關節 {i}: {joint_name}")
    print(f"    類型: {joint_type}, 限制: [{lower_limit}, {upper_limit}]")
    print(f"    最大扭矩: {max_force} Nm, 最大速度: {max_velocity} rad/s")

# 測試基座和腳位置
p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.5], [0, 0, 0, 1])
base_pos, base_quat = p.getBasePositionAndOrientation(robot_id)
print(f"\n基座位置: {base_pos}")

# 找到腳關節索引
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    if "foot" in joint_info[1].decode('utf-8').lower():
        print(f"  腳部找到: 關節 {i} = {joint_info[1].decode('utf-8')}")

p.disconnect()
```

**記錄以下參數**:
- [ ] 活躍關節索引: `[0, 1, 2, ...]`
- [ ] 腳部接觸點（左/右）: `{"left": idx, "right": idx}`
- [ ] 最大扭矩 (各關節): `[150, 120, 100, ...]` (Nm)
- [ ] 物理 Hz (通常 240 或 500): `240`
- [ ] 總關節數: `N`
- [ ] 觀測維度: `9 + N*2 + 2 = 9 + 2N + 2`

---

## 步驟 3: 創建機器人配置 YAML

在 `configs/robots/` 目錄中創建配置文件：

```bash
$ cat > configs/robots/your_robot.yaml << 'EOF'
# 機器人配置: Your Robot
# 版本: 1.0
# 維護者: Your Name

name: your_robot
urdf_description: "your_robot_description"  # robot_descriptions 包名稱，或本地路徑

physics:
  physics_hz: 240               # 物理模擬頻率 (Hz)
  substeps: 4                   # 每個策略步的子步數
  timestep: 0.00416             # 1 / physics_hz (自動計算)
  gravity: -9.81

robot:
  mass: 55.0                    # 總質量 (kg)
  active_joints: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 活躍關節索引
  
  # 最大扭矩限制 (Nm) — 用於動作縮放
  max_torques:
    joint_0: 150.0
    joint_1: 150.0
    joint_2: 50.0
    joint_3: 150.0
    joint_4: 150.0
    joint_5: 50.0
    joint_6: 150.0
    joint_7: 150.0
    joint_8: 50.0
    joint_9: 50.0
  
  # 相應的關節最大速度 (rad/s)
  max_velocities:
    joint_0: 5.0
    joint_1: 5.0
    # ... 等等
  
  # 初始/平衡姿態 (站立姿勢)
  stand_pose:
    joint_0: 0.0
    joint_1: 0.5
    joint_2: -1.0
    # ... 所有活躍關節
  
  # 腳接觸點鏈接索引
  feet:
    left_foot: 7   # 左腳鏈接編號
    right_foot: 14 # 右腳鏈接編號
  
  # 觀測配置
  obs_dim: 31  # 自動: 9 + 10*2 + 2
  action_dim: 10  # 活躍關節數

reward:
  # 生存獎勵
  alive_bonus: 2.0
  
  # 前進運動獎勵
  forward_reward_scale: 1.5
  forward_speed_target: 1.0  # 目標前進速度 (m/s)
  
  # 高度獎勵 (保持站立)
  height_reward_scale: 3.0
  target_height: 0.8  # 目標軀幹高度 (m)
  
  # 能量懲罰 (平滑運動)
  smooth_penalty_scale: 0.0001
  
  # 死亡懲罰 (摔倒)
  death_penalty: -30.0
  min_height: 0.3  # 低於此高度判定為失敗
  max_contact_force: 10000  # 接觸力過大判定為失敗

EOF
```

### 配置參數詳解

| 參數 | 說明 | 範例 |
|------|------|------|
| `physics_hz` | 物理引擎頻率 | 240-500 |
| `active_joints` | 可驅控的關節 | [0,1,2,...] |
| `max_torques` | 各關節最大扭矩 (Nm) | 150, 50 等 |
| `stand_pose` | 平衡/初始姿態 (rad) | 站立時的關節角 |
| `feet` | 腳部鏈接索引 | 用於接觸檢測 |
| `alive_bonus` | 每步存活獎勵 | 1-5 |
| `forward_reward_scale` | 前進速度權重 | 0.1-2.0 |
| `smooth_penalty_scale` | 動作平滑度懲罰 | 0.0001-0.001 |

---

## 步驟 4: 創建環境類 (可選)

如果機器人有**特定的觀測/動作邏輯**，創建子類：

### 情況 A: 標準雙足機器人 (推薦)

若機器人遵循標準模式 (軀幹 + 10 個腿關節 + 足接觸)，**無需自訂環境**，直接使用通用環境。

### 情況 B: 自訂觀測或動作

若需特殊邏輯，創建 `envs/your_robot_env.py`:

```python
from envs.base_walking_env import WalkingEnv
from config import RobotConfig

class YourRobotEnv(WalkingEnv):
    """Your Robot 環境 (自訂邏輯)"""
    
    def __init__(self, robot_config: RobotConfig, render_mode=None):
        super().__init__(robot_config, render_mode)
        
    def _init_robot(self):
        """初始化機器人（可選）"""
        # 加載機制臂或其他附件
        pass
        
    def _observe(self) -> np.ndarray:
        """觀測: 軀幹 + 關節 + 速度 + 接觸
        
        返回 obs_dim 維向量
        """
        base_pos = self._get_base_position()  # [x, y, z]
        base_vel = self.simulator.get_base_linear_velocity()  # [vx, vy, vz]
        base_ang_vel = self.simulator.get_base_angular_velocity()  # [wx, wy, wz]
        
        joint_pos = self.simulator.get_joint_positions()  # n 維
        joint_vel = self.simulator.get_joint_velocities()  # n 維
        
        foot_contact = self._get_foot_contact()  # [left, right]
        
        obs = np.concatenate([
            [base_pos[2]],  # 高度
            # base_ori,  # 可選: 俯仰/滾動
            base_vel,
            base_ang_vel,
            joint_pos,
            joint_vel,
            foot_contact,
        ])
        
        return obs.astype(np.float32)
        
    def _apply_action(self, action: np.ndarray):
        """動作: 標準化扭矩 [-1, 1] → 實際扭矩"""
        torques = action * self.robot_config.max_torques
        self.simulator.apply_action(torques)
        
    def _get_base_position(self) -> np.ndarray:
        return self.simulator.get_base_position()
        
    def _get_base_orientation(self) -> np.ndarray:
        return self.simulator.get_base_orientation_euler()
        
    def _settle(self, num_steps=100):
        """讓機器人穩定到平衡姿態"""
        # 施加負反饋以穩定關節
        for _ in range(num_steps):
            pose = self.robot_config.stand_pose
            joint_pos = self.simulator.get_joint_positions()
            error = pose - joint_pos
            torques = error * 10.0  # 簡單 P 控制器
            self.simulator.apply_action(torques)
            self.simulator.step()
            
    # 其他 hook 方法實現...
```

然後在 train.py 中註冊:

```python
# train.py 中的 make_env 函數
def make_env(robot_name: str, render: bool = False):
    mgr = ConfigManager()
    robot_config = mgr.load_robot_config(robot_name)
    
    if robot_name == "cassie":
        from envs.cassie_env import CassieEnv
        return CassieEnv(robot_config, render_mode="human" if render else None)
    elif robot_name == "h1":
        from envs.h1_env import H1Env
        return H1Env(robot_config, render_mode="human" if render else None)
    elif robot_name == "your_robot":  # ← 添加
        from envs.your_robot_env import YourRobotEnv
        return YourRobotEnv(robot_config, render_mode="human" if render else None)
    else:
        raise ValueError(f"Unknown robot: {robot_name}")
```

---

## 步驟 5: 訓練新機器人

現在可以訓練新機器人，**無需修改 train.py**：

```bash
# 基本訓練
python train.py --robot your_robot

# 指定配置
python train.py --robot your_robot --config default

# 調整超參數
python train.py --robot your_robot \
    --sac__learning_rate 3e-4 \
    --n_envs 16 \
    --total_timesteps 1000000

# 環境變數覆蓋
PYWALKING_FORWARD_REWARD_SCALE=2.0 python train.py --robot your_robot

# 帶渲染
python train.py --robot your_robot --render
```

---

## 步驟 6: 驗證和調試

### 6.1 檢查配置

```bash
python -c "
from config import ConfigManager
mgr = ConfigManager(verbose=True)
cfg = mgr.load_robot_config('your_robot')
print(f'✓ obs_dim={cfg.obs_dim}, action_dim={cfg.action_dim}')
print(f'✓ active_joints={cfg.active_joints}')
print(f'✓ max_torques={cfg.max_torques}')
"
```

### 6.2 測試環境

```python
import sys
sys.path.insert(0, '.')
from config import ConfigManager
from envs.your_robot_env import YourRobotEnv

mgr = ConfigManager()
cfg = mgr.load_robot_config('your_robot')
env = YourRobotEnv(cfg, render_mode=None)

# 重置
obs, info = env.reset()
print(f"✓ 重置成功，obs shape: {obs.shape}")

# 運行 10 步
for i in range(10):
    action = env.action_space.sample()  # 隨機動作
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  步 {i}: reward={reward:.4f}, done={terminated or truncated}")

env.close()
print("✓ 環境測試通過")
```

### 6.3 短訓練測試

```bash
# 5000 步測試訓練 (快速驗證)
python train.py --robot your_robot --total_timesteps 5000
```

檢查:
- ✅ 無崩潰
- ✅ 模型文件生成: `models/your_robot/your_robot_sac.zip`
- ✅ 日誌生成: `logs/your_robot/`

---

## 常見問題 (FAQ)

### Q1: 配置中 `obs_dim` 如何計算？

**A**: 
```
obs_dim = 9 + len(active_joints)*2 + 2
        = 基座(9) + 關節角度(n) + 關節速度(n) + 接觸點(2)
        
例: 10 個活躍關節
obs_dim = 9 + 10 + 10 + 2 = 31
```

### Q2: 如何驗證 `max_torques` 值？

**A**: 查詢機器人規格書或 URDF 中的 `<limit effort="..."/>`

```python
# PyBullet 查詢
import pybullet as p
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    max_force = info[10]  # 接觸力上限
    print(f"Joint {i}: max_force={max_force} N")
```

### Q3: 獎勵權重應該怎樣設置？

**A**: 從 Cassie 的默認值開始，逐步調整：

```yaml
reward:
  alive_bonus: 2.0           # [1-5] 較小的存活獎勵
  forward_reward_scale: 1.5  # [0.5-2.0] 主要激勵前進
  height_reward_scale: 3.0   # [2-5] 保持姿態
  smooth_penalty_scale: 0.0001  # 能量消耗
```

**調試步驟**:
1. 執行 100k 步訓練
2. 觀察獎勵曲線: `tensorboard --logdir logs/your_robot`
3. 若獎勵平坦 → 增大激勵
4. 若機器人摔倒 → 增大 `height_reward_scale`

### Q4: 物理模擬频率該設多少？

**A**: 通常 240-500 Hz
- **240 Hz**: 較慢機器人，計算便宜
- **500 Hz**: 快速反應，需更高計算
- **實際測試**: 試試都行，觀察訓練曲線差異

### Q5: 如何從頭開始訓練？

**A**: 確保模型路徑不存在或使用 `--no-resume`:

```bash
# 清除舊模型
rm models/your_robot/*.zip

# 重新訓練
python train.py --robot your_robot
```

---

## 檢查清單

新機器人добавка完成:

- [ ] URDF 文件位於 `robots/` 或通過 robot_descriptions 可訪問
- [ ] YAML 配置已創建: `configs/robots/your_robot.yaml`
- [ ] (可選) 環境類已創建: `envs/your_robot_env.py`
- [ ] (可選) train.py 中已註冊環境
- [ ] 配置驗證通過 (步驟 6.1)
- [ ] 環境測試通過 (步驟 6.2)
- [ ] 短訓練測試通過 (步驟 6.3)
- [ ] 完整訓練已啟動: `python train.py --robot your_robot`

---

## 下一步

- 📊 監視訓練進度: `tensorboard --logdir logs/your_robot/`
- 🎯 調整獎勵權重以優化學習：見 [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
- 🤖 為真實機器人集成做準備：見 [ARCHITECTURE.md](ARCHITECTURE.md#未來擴展點)

---

**相関文檔**: [ARCHITECTURE.md](ARCHITECTURE.md), [CONFIG_GUIDE.md](CONFIG_GUIDE.md)

**版本**: Phase 4 v1.0  
**最後更新**: 2026-04-01
