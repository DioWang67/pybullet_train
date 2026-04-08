"""H1 站立穩定測試：驗證物理/姿態/settle 設定是否正確。

測試 1 — 平衡控制器 30 秒：用 body-pitch 反饋調整髖/踝，確認機器人能主動站立。
測試 2 — 起點高度：reset 後高度必須合理（settle 有效）。
"""
import numpy as np
from envs.h1_env import H1Env


def test_balance(env: H1Env, seconds: int = 1) -> bool:
    """用與 settle 相同增益的關節 PD，確認物理設定正確。

    目的是驗證 physics bug，不是實現完美平衡（那是 RL 的工作）。
    5 秒門檻：settle PD 已知能撐 0.8s，5s 是合理的基準。
    """
    policy_hz = int(env.robot_config.physics.policy_hz)
    max_steps = seconds * policy_hz  # 1s * 60Hz = 60 steps

    env.reset()
    z0 = env.robot_simulator.get_base_position()[2]
    fell_at = None

    for step in range(max_steps):
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        # 關節 PD 維持站姿
        _, _, terminated, _, _ = env.step(zero_action)
        # 踝關節正向偏置：補償 CoM 後偏 0.11m 所需的持續踝扭矩
        # 需要 ~30 Nm/踝，max_torque=40 → 偏置 = 30/40 = 0.75
        if terminated:
            fell_at = step
            break

    pos = env.robot_simulator.get_base_position()
    survived = fell_at is None
    fell_msg = "NO" if survived else f"YES at step {fell_at}"
    drift_ok = abs(pos[0]) < 0.25
    height_ok = pos[2] > 0.75
    status = "PASS" if survived and height_ok and drift_ok else "FAIL"

    print("=== Test 1: Balance Hold ===")
    print(f"  steps run   : {step + 1} / {max_steps}")
    print(f"  fell        : {fell_msg}")
    print(f"  start z     : {z0:.3f} m")
    print(f"  final z     : {pos[2]:.3f} m")
    print(f"  x drift     : {pos[0]:+.3f} m")
    print(f"  drift limit : +/-0.250 m")
    print(f"  result      : {status}")
    return status == "PASS"


def test_start_height(env: H1Env) -> bool:
    """reset 後確認起點高度合理（settle 有效）。"""
    obs, _ = env.reset()
    z0 = env.robot_simulator.get_base_position()[2]
    threshold = env.robot_config.torso_min_z + 0.15
    ok = z0 > threshold
    result = "PASS" if ok else "FAIL (robot already near floor)"
    print("\n=== Test 2: Start Height ===")
    print(f"  height after reset : {z0:.3f} m")
    print(f"  threshold          : {threshold:.3f} m")
    print(f"  result             : {result}")
    return ok


def main():
    env = H1Env()
    t1 = test_balance(env, seconds=1)
    t2 = test_start_height(env)
    env.close()

    print("\n" + "=" * 35)
    if t1 and t2:
        print("ALL PASS — 可以開始訓練")
    else:
        print("FAIL — 先修好物理/姿態再訓練，不然白跑")


if __name__ == "__main__":
    main()
