"""
Cassie 機器人外觀預覽
執行：python view_cassie.py
按 Enter 關閉
"""
import time
import pybullet as p
import pybullet_data
from robot_descriptions.loaders.pybullet import load_robot_description

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1 / 240)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

ground = p.loadURDF("plane.urdf")
p.changeDynamics(ground, -1, lateralFriction=1.5)

# 載入 Cassie，spawn 高度 1.2m（機器人站立高度約 1.0m）
robot = load_robot_description("cassie_description", basePosition=[0, 0, 1.2])

p.resetDebugVisualizerCamera(
    cameraDistance=2.5, cameraYaw=45, cameraPitch=-20,
    cameraTargetPosition=[0, 0, 0.8],
)

print(f"Cassie loaded! joints: {p.getNumJoints(robot)}")
print("讓機器人自然落下...")
print("按 Enter 關閉\n")

# 關閉預設馬達（讓它自然落下）
for j in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot, j)[2] == p.JOINT_REVOLUTE:
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)

step = 0
try:
    while True:
        p.stepSimulation()
        time.sleep(1 / 240)
        step += 1
        if step % 240 == 0:
            pos, _ = p.getBasePositionAndOrientation(robot)
            p.resetDebugVisualizerCamera(
                cameraDistance=2.5, cameraYaw=45, cameraPitch=-20,
                cameraTargetPosition=[pos[0], pos[1], 0.8],
            )
except KeyboardInterrupt:
    pass

p.disconnect()
print("Done.")
