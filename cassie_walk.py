"""
Cassie 正弦步態控制器
執行：python cassie_walk.py
Ctrl+C 退出
"""
import math
import time
import pybullet as p
import pybullet_data
from robot_descriptions.loaders.pybullet import load_robot_description

# ── 關節 index ────────────────────────────────────────────────────────────────
L_HIP_ROLL,  L_HIP_YAW,  L_HIP_PITCH  = 0, 1, 2
L_KNEE,      L_FOOT                    = 3, 7
R_HIP_ROLL,  R_HIP_YAW,  R_HIP_PITCH  = 8, 9, 10
R_KNEE,      R_FOOT                    = 11, 15

ACTIVE_JOINTS = [
    L_HIP_ROLL, L_HIP_YAW, L_HIP_PITCH, L_KNEE, L_FOOT,
    R_HIP_ROLL, R_HIP_YAW, R_HIP_PITCH, R_KNEE, R_FOOT,
]

# ── 步態參數 ──────────────────────────────────────────────────────────────────
FREQ      = 1.2    # 步頻 Hz
SPAWN_Z   = 1.05   # 生成高度（Cassie 站立高度約 1.0m）

# 各關節的靜止站立角度（rad）
STAND_POSE = {
    L_HIP_ROLL:  0.0,
    L_HIP_YAW:   0.0,
    L_HIP_PITCH: 0.5,    # 略微前傾
    L_KNEE:      -1.4,   # 在關節限制內：[-2.86, -0.95]
    L_FOOT:      -1.0,   # 在關節限制內：[-2.44, -0.52]
    R_HIP_ROLL:  0.0,
    R_HIP_YAW:   0.0,
    R_HIP_PITCH: 0.5,
    R_KNEE:      -1.4,
    R_FOOT:      -1.0,
}

# 各關節步態振幅（rad）
AMPLITUDE = {
    L_HIP_ROLL:  0.05,
    L_HIP_YAW:   0.0,
    L_HIP_PITCH: 0.35,   # 主要前後擺動
    L_KNEE:      0.30,
    L_FOOT:      0.20,
    R_HIP_ROLL:  0.05,
    R_HIP_YAW:   0.0,
    R_HIP_PITCH: 0.35,
    R_KNEE:      0.30,
    R_FOOT:      0.20,
}


def gait_target(t: float) -> dict:
    """t（秒）→ 各關節目標角度"""
    phase_l = 2 * math.pi * FREQ * t
    phase_r = phase_l + math.pi   # 右腿反相

    targets = {}
    for joint, phase in [
        (L_HIP_ROLL,  phase_l), (L_HIP_YAW,   phase_l),
        (L_HIP_PITCH, phase_l), (L_KNEE,       phase_l),
        (L_FOOT,      phase_l),
        (R_HIP_ROLL,  phase_r), (R_HIP_YAW,   phase_r),
        (R_HIP_PITCH, phase_r), (R_KNEE,       phase_r),
        (R_FOOT,      phase_r),
    ]:
        targets[joint] = STAND_POSE[joint] + AMPLITUDE[joint] * math.sin(phase)
    return targets


def main():
    # ── 初始化 ────────────────────────────────────────────────────────────────
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)
    p.setPhysicsEngineParameter(numSolverIterations=50)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

    ground = p.loadURDF("plane.urdf")
    p.changeDynamics(ground, -1, lateralFriction=1.5, restitution=0.0)

    robot = load_robot_description(
        "cassie_description",
        basePosition=[0, 0, SPAWN_Z],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    )

    # 腳底增加摩擦力
    for j in range(p.getNumJoints(robot)):
        name = p.getJointInfo(robot, j)[1].decode()
        if "Foot" in name:
            p.changeDynamics(robot, j, lateralFriction=1.5)

    # 關閉被動關節預設馬達
    for j in range(p.getNumJoints(robot)):
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)

    # 設定初始站立姿勢
    for joint, angle in STAND_POSE.items():
        p.resetJointState(robot, joint, angle)

    p.resetDebugVisualizerCamera(
        cameraDistance=3.0, cameraYaw=30, cameraPitch=-15,
        cameraTargetPosition=[0, 0, 0.8],
    )

    print("Cassie gait controller started.")
    print("Ctrl+C to quit\n")

    # ── 沉降 0.3 秒 ───────────────────────────────────────────────────────────
    print("Settling...")
    for _ in range(int(0.3 * 240)):
        p.stepSimulation()
    print("Walking!\n")

    # ── 主迴圈 ────────────────────────────────────────────────────────────────
    t_start = time.time()
    step    = 0

    try:
        while True:
            t = time.time() - t_start
            targets = gait_target(t)

            for joint, angle in targets.items():
                p.setJointMotorControl2(
                    robot, joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=angle,
                    positionGain=0.4,
                    velocityGain=0.05,
                    force=80,
                )

            p.stepSimulation()
            time.sleep(1 / 240)
            step += 1

            # 側向鎖定：強制在 XZ 平面行走，消除 roll/yaw 漂移
            pos, orn = p.getBasePositionAndOrientation(robot)
            vel, ang = p.getBaseVelocity(robot)
            _, pitch, _ = p.getEulerFromQuaternion(orn)
            p.resetBasePositionAndOrientation(
                robot,
                [pos[0], 0.0, pos[2]],
                p.getQuaternionFromEuler([0.0, pitch, 0.0]),
            )
            p.resetBaseVelocity(
                robot,
                [vel[0], 0.0, vel[2]],
                [0.0, ang[1], 0.0],
            )

            # 相機跟隨
            if step % 24 == 0:
                pos, _ = p.getBasePositionAndOrientation(robot)
                p.resetDebugVisualizerCamera(
                    cameraDistance=3.0, cameraYaw=30, cameraPitch=-15,
                    cameraTargetPosition=[pos[0], pos[1], 0.8],
                )

            # 每 3 秒印一次狀態
            if step % 720 == 0:
                pos, orn = p.getBasePositionAndOrientation(robot)
                vel, _   = p.getBaseVelocity(robot)
                pitch    = math.degrees(p.getEulerFromQuaternion(orn)[1])
                print(f"t={t:.1f}s  x={pos[0]:+.2f}m  z={pos[2]:.2f}m  "
                      f"vx={vel[0]:+.2f}m/s  pitch={pitch:+.1f}°")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
