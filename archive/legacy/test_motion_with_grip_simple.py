import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from dmp_assistant import paths
from fairino import Robot
import time
from dmp_assistant.skills.gripper import gripper_close, gripper_open

# ================================================================
# FILL IN YOUR TARGET POSE BEFORE RUNNING
# [X, Y, Z, Rx, Ry, Rz]
# ================================================================
TARGET_POSE = [72.56365966796875, -480.614013671875, 436.7640380859375,
               -177.40708923339844, 1.228800892829895, 91.3707504272461]
TARGET2=[73.36058807373047, -464.8749694824219, 555.8865356445312, -176.35354614257812, 14.948424339294434, 91.51934814453125]
# ================================================================

robot = Robot.RPC('192.168.58.4')

# 1. Close gripper
gripper_close(robot)

# 2. Inline MoveL with sleeps (no wrapper)
time.sleep(0.5)
error = robot.MoveL(TARGET_POSE, tool=0, user=0, vel=30)
print(f"MoveL return: {error}")
time.sleep(0.5)

# 3. Open gripper
gripper_open(robot)

# 2. Inline MoveL with sleeps (no wrapper)
time.sleep(0.5)
error = robot.MoveL(TARGET2, tool=0, user=0, vel=30)
print(f"MoveL return: {error}")
time.sleep(0.5)

print("Cycle complete.")