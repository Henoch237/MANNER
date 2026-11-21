import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from dmp_assistant import paths
from fairino import Robot
import time
from dmp_assistant.skills.gripper import gripper_close, gripper_open

# ================================================================
# FILL IN YOUR TARGET JOINTS BEFORE RUNNING
# ================================================================
TARGET_JP = [87.0826416015625, -92.78965759277344, 99.05316925048828,
             -97.68281555175781, -92.49443817138672, 85.70880889892578]

TARGET_POSE = [72.56365966796875, -480.614013671875, 436.7640380859375,
               -177.40708923339844, 1.228800892829895, 91.3707504272461]
# ================================================================

def movej(robot, joint_pos, vel=30):
    robot.ResetAllError()
    time.sleep(0.3)
    error = robot.MoveJ(joint_pos, tool=0, user=0, vel=vel)
    print(f"MoveJ return: {error}")
    time.sleep(0.5)
    robot.ResetAllError()
    return error

def movel(robot, desc_pos, vel=30):
    robot.ResetAllError()
    time.sleep(0.3)
    error = robot.MoveL(desc_pos, tool=0, user=0, vel=vel)
    print(f"MoveL return: {error}")
    time.sleep(0.5)
    robot.ResetAllError()
    return error


# --- Main ---
robot = Robot.RPC('192.168.58.4')

# 1. Close gripper
#gripper_close(robot)

# 2. MoveJ to target
#movej(robot, TARGET_JP)
#movel(robot, TARGET_POSE)

# 3. Open gripper
gripper_open(robot)

print("Cycle complete.")