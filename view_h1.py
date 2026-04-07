"""
Unitree H1 站立預覽
執行：python view_h1.py
Ctrl+C 退出
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
p.changeDynamics(ground, -1, lateralFriction=1.0)

robot = load_robot_description(
    "h1_description",
    basePosition=[0, 0, 1.05],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
)

# 所有關節鎖在 0（站立姿勢）
for j in range(p.getNumJoints(robot)):
    if p.getJointInfo(robot, j)[2] == p.JOINT_REVOLUTE:
        p.setJointMotorControl2(
            robot, j, p.POSITION_CONTROL,
            targetPosition=0.0,
            positionGain=1.0,
            velocityGain=0.1,
            force=500,
        )

p.resetDebugVisualizerCamera(
    cameraDistance=3.0, cameraYaw=30, cameraPitch=-15,
    cameraTargetPosition=[0, 0, 1.0],
)

print("H1 standing. Ctrl+C to quit.")

try:
    while True:
        p.stepSimulation()
        time.sleep(1 / 240)
except KeyboardInterrupt:
    pass
finally:
    p.disconnect()
