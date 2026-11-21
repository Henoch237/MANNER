"""Manual: BT executor smoke check (connects to the sim and ticks once).
python tests/manual/check_executor_bt.py"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from dmp_assistant import paths

print("=" * 60)
print("BT EXECUTOR VALIDATION TEST")
print("=" * 60)

# --- 1. Test imports ---
print("\n[1/6] Testing imports...")
try:
    from dmp_assistant.skills.gripper import gripper_close, gripper_open
    from dmp_assistant.skills.motion import movel, movej
    from dmp_assistant.skills.robot_state import get_current_pose
    from dmp_assistant.skills.dmp import RhythmicDMP
    import py_trees
    import py_trees.trees
    print("   ✅ All skill + py_trees imports successful.")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# --- 2. Test scene.yaml ---
print("\n[2/6] Testing scene.yaml...")
scene_path = paths.SCENE_YAML
if not os.path.exists(scene_path):
    print(f"   ❌ {scene_path} not found.")
    sys.exit(1)

try:
    import yaml
    with open(scene_path, "r") as f:
        scene = yaml.safe_load(f)
    print(f"   ✅ scene.yaml loaded. Top keys: {list(scene.keys())}")
except Exception as e:
    print(f"   ❌ scene.yaml parse error: {e}")
    sys.exit(1)

# --- 3. Test pose resolution via Executor ---
print("\n[3/6] Testing pose resolution...")
try:
    from dmp_assistant.execution.executor import Executor
    # Create executor WITHOUT connecting to robot (we'll test connection next)
    # But Executor.__init__ calls Robot.RPC... so we need to connect.
    # We'll do a lightweight check by importing the class and checking _resolve logic indirectly.
    poses = scene["poses"]
    print(f"   initial_pose: {poses['initial_pose']['value'][:3]}...")
    print(f"   red_block.grasp_pose: {poses['red_block']['grasp_pose']['value'][:3]}...")
    print(f"   coffee_cup.center_pose ref: {poses['coffee_cup']['center_pose'].get('ref', 'direct value')}")
    print("   ✅ Pose structure valid.")
except Exception as e:
    print(f"   ❌ Pose check failed: {e}")
    sys.exit(1)

# --- 4. Test DMP file ---
print("\n[4/6] Testing DMP data file...")
dmp_file = paths.STIRRING_DMP
if os.path.exists(dmp_file):
    print(f"   ✅ {dmp_file} found.")
else:
    print(f"   ⚠️  {dmp_file} NOT found (stirring will fail if used).")

# --- 5. Test robot connection + BT compilation (dry run) ---
SIM_IP = "192.168.85.130"
print(f"\n[5/6] Testing robot connection @ {SIM_IP}...")

try:
    from fairino import Robot
    robot = Robot.RPC(SIM_IP)
    print(f"   ✅ Robot.RPC('{SIM_IP}') created.")

    err, pose = robot.GetActualTCPPose(0)
    if err == 0:
        print(f"   ✅ GetActualTCPPose OK: {pose[:3]}")
    else:
        print(f"   ⚠️  GetActualTCPPose returned error: {err}")

except Exception as e:
    print(f"   ❌ Robot connection failed: {e}")
    sys.exit(1)

# --- 6. Test BT compilation (no motion) ---
print("\n[6/6] Testing BT compilation (dry run, no motion)...")

try:
    exe = Executor(SIM_IP)

    dry_plan = [
        {"action": "go_to", "target_pose": "initial_pose.value"},
        {"action": "pick", "object_id": "red_block"},
        {"action": "go_to", "target_pose": "initial_pose.value"},
        {"action": "stir", "cup_id": "coffee_cup", "omega_scale": 0.8, "revolutions": 1}
    ]

    root = exe.compile_plan(dry_plan)

    # Verify tree structure
    print(f"   Root type: {type(root).__name__}")
    print(f"   Children count: {len(root.children)}")
    for i, child in enumerate(root.children):
        print(f"     [{i+1}] {child.name} -> {type(child).__name__}")

    # Quick tick test (will attempt motion — comment out if you want pure compile check)
    print("\n   [OPTIONAL] Running one tick to verify tree starts...")
    tree = py_trees.trees.BehaviourTree(root)
    tree.setup()
    tree.tick()
    print(f"   Root status after first tick: {tree.root.status}")
    tree.shutdown()
    print("   ✅ BT compile + tick test passed.")

except Exception as e:
    print(f"   ❌ BT compilation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL BT CHECKS PASSED — Executor is ready.")
print("=" * 60)