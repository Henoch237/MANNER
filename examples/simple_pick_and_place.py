"""Example: scripted pick-and-place without the planner.
python examples/simple_pick_and_place.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

import time

from fairino import Robot
from dmp_assistant.skills.gripper import gripper_close, gripper_open

# ================================================================
# FILL IN YOUR POSES BEFORE RUNNING
# [X, Y, Z, Rx, Ry, Rz]
# ================================================================
INITIAL_POSE = [102.26055145263672, -383.2015075683594, 493.603759765625,
                179.2078399658203, 4.763824462890625, 91.51363372802734]

PICK_POSE    = [-21.24605369567871, -538.5603637695312, 221.66738891601562,
                176.8939971923828, 7.890963077545166, 87.8626937866211]

PLACE_POSE   = [271.9374084472656, -547.2503662109375, 260.673095703125,
                178.74368286132812, 1.8415530920028687, 96.92931365966797]
# ================================================================

# Approach pose = 35 mm above the pick pose
PICK_APPROACH = PICK_POSE.copy()
PICK_APPROACH[2] += 35.0

if __name__ == "__main__":
    robot = Robot.RPC('192.168.85.130')

    # 1. Go to initial pose
    print("Moving to initial pose...")
    robot.MoveL(INITIAL_POSE, tool=0, user=0, vel=30)
    time.sleep(0.5)

    # 2. Open gripper
    print("Opening gripper...")
    gripper_open(robot)
    time.sleep(2)

    # 3. Go to pick approach (35 mm above)
    print("Moving to pick approach...")
    robot.MoveL(PICK_APPROACH, tool=0, user=0, vel=30)
    time.sleep(0.5)

    # 4. Descend to actual pick pose
    print("Descending to grasp...")
    robot.MoveL(PICK_POSE, tool=0, user=0, vel=30)
    time.sleep(0.5)

    # 5. Close gripper
    print("Closing gripper...")
    gripper_close(robot)
    time.sleep(2)

    # 6. Retract to pick approach
    print("Retracting from pick...")
    robot.MoveL(PICK_APPROACH, tool=0, user=0, vel=30)
    time.sleep(0.5)

    # 7. Go to initial pose
    print("Moving to initial pose...")
    robot.MoveL(INITIAL_POSE, tool=0, user=0, vel=30)
    time.sleep(0.5)

    # 8. Go to place pose
    print("Moving to place pose...")
    robot.MoveL(PLACE_POSE, tool=0, user=0, vel=30)
    time.sleep(0.5)

    # 9. Open gripper
    print("Opening gripper...")
    gripper_open(robot)
    time.sleep(2)

    # 10. Return to initial pose
    print("Returning to initial pose...")
    robot.MoveL(INITIAL_POSE, tool=0, user=0, vel=30)

    print("Pick-and-place complete.")
