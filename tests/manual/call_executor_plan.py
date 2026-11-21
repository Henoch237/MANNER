"""Manual: run a hard-coded plan on the robot. python tests/manual/call_executor_plan.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

from dmp_assistant.execution.executor import Executor

# ================================================================
# SIMULATION ADDRESS
# ================================================================
SIM_IP = "192.168.85.130"

# ================================================================
# HARD-CODED TEST PLANS
# Uncomment ONE plan block at a time to test incrementally.
# ================================================================

#--- Plan A: Just move to poses (no gripper, no DMP) ---
#plan_a = [
#     {"action": "movel", "pose": "initial_pose.value"},
#     {"action": "movel", "pose": "red_block.approach_pose"},
#     {"action": "movel", "pose": "initial_pose.value"},
#]


# --- Plan B: Pick only ---
# plan_b = [
#     {"action": "pick", "object_id": "red_block"},
# ]

# --- Plan C: Full pick and place ---
plan_c = [
    {"action": "pick_and_place", "object_id": "red_block", "destination_id": "bin"},
]

# --- Plan D: Stir only (assumes stick is already held) ---
#plan_d = [
#     {"action": "stir", "cup_id": "coffee_cup", "omega_scale": 0.8, "revolutions": 2},
#]

# --- Plan E: Full sequence ---
#plan_e = [
#     {"action": "pick_and_place", "object_id": "red_block", "destination_id": "bin"},
#     {"action": "stir", "cup_id": "coffee_cup", "omega_scale": 0.8, "revolutions": 2},
#]

# ================================================================
# RUN
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("HARD-CODED PLAN TEST")
    print(f"Target: {SIM_IP}")
    print("=" * 60)

    exe = Executor(SIM_IP)

    # Pick the plan you want to run
    active_plan = plan_c

    print(f"\nExecuting {len(active_plan)} step(s)...")
    exe.run_plan(active_plan)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)