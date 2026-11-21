"""
tester_sequence.py - small integration test of the pour call.

Sequence: open gripper -> close gripper (bottle) -> pour -> open gripper.
The pour is run_pour.pour - the exact code validated standalone.

Run:  python tester_sequence.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from dmp_assistant import paths
import time
import numpy as np
from fairino import Robot
from dmp_assistant.skills.gripper import gripper_open, gripper_close
from dmp_assistant.skills.pour import pour

ROBOT_IP = "192.168.58.4"
TILT_DEG = None      # None = recorded demo (104); try 80 for a smaller pour
DWELL_S = None       # None = recorded (~3.4s)
SPEED = 1.0
REVS = 1

robot = Robot.RPC(ROBOT_IP)
robot.LoggerInit(output_model=0)
robot.SetLoggerLevel(1)
print("RobotEnable ->", robot.RobotEnable(1))
time.sleep(1.0)

print("\n>>> Step 1/4: OPEN gripper. Enter: ")
input()
gripper_open(robot)
time.sleep(1.5)

print(">>> Step 2/4: place the BOTTLE, then Enter to CLOSE: ")
input()
gripper_close(robot)
time.sleep(1.5)

print(f">>> Step 3/4: POUR (tilt {TILT_DEG or 104}, dwell "
      f"{DWELL_S or 3.4}, speed {SPEED}, revs {REVS}). Enter: ")
input()
dev = pour(robot, tilt_deg=TILT_DEG, dwell_s=DWELL_S, speed=SPEED,
           revs=REVS, confirm=False)
print(f"    pour done, deviation {dev} mm")

print(">>> Step 4/4: OPEN gripper (release). Enter: ")
input()
gripper_open(robot)
time.sleep(1.5)

print("\nSEQUENCE COMPLETE")