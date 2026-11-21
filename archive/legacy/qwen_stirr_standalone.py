"""LEGACY standalone demo (superseded by dmp_assistant.launcher).
Qwen picks a DMP primitive + parameters, then executes it at the current
pose. Only stir_circle has a learned file; ellipse/offer are not recorded.

Run from the project root:  python archive/legacy/qwen_stirr_standalone.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fairino import Robot
from dmp_assistant import paths
from dmp_assistant.skills.dmp import RhythmicDMP
import numpy as np
import time
import json
import os
import dashscope

# ==================== CONFIG ====================
ROBOT_IP = "192.168.58.4"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # export DASHSCOPE_API_KEY=...

# DMP Library (embedded in prompt)
DMP_LIBRARY = {
    "primitives": [
        {
            "id": "stir_circle",
            "file": str(paths.DMP_DIR / "learned_stirring_dmp.npy"),
            "type": "rhythmic",
            "description": "Circular stirring in a plane",
            "parameters": {
                "scale": {"default": 1.0, "min": 0.3, "max": 3.0, "description": "Size multiplier"},
                "omega": {"default": 1.0, "min": 0.2, "max": 3.0, "description": "Speed multiplier"},
                "plane": {"default": "XY", "options": ["XY", "ZX", "YZ"]}
            }
        },
        {
            "id": "stir_ellipse",
            "file": str(paths.DMP_DIR / "learned_ellipse_dmp.npy"),
            "type": "rhythmic",
            "description": "Elliptical stirring for non-circular bowls",
            "parameters": {
                "scale": {"default": 1.0, "min": 0.3, "max": 3.0},
                "omega": {"default": 1.0, "min": 0.2, "max": 3.0},
                "plane": {"default": "XY", "options": ["XY", "ZX", "YZ"]}
            }
        },
        {
            "id": "offer_gesture",
            "file": str(paths.DMP_DIR / "learned_offer_dmp.npy"),
            "type": "point_to_point",
            "description": "Forward-present motion to hand object to the slower profile",
            "parameters": {
                "scale": {"default": 1.0, "min": 0.5, "max": 1.5}
            }
        }
    ]
}

# Execution settings
N_WAYPOINTS = 8
VEL = 30
ACC = 100
OVL = 50
BLEND_R = 15.0
BLEND_MODE = 0


# ================================================


# ==================== RHYTHMIC DMP ====================
# ==================== LLM PLANNER ====================
def call_qwen(user_request, scene_context=""):
    """Call Qwen API to get DMP parameters. Returns parsed JSON dict."""

    dashscope.api_key = DASHSCOPE_API_KEY

    prompt = f"""You are a robot motion planner for a caregiving FAIRINO F5 manipulator.
You have a library of learned DMP primitives recorded from human demonstration.

Library:
{json.dumps(DMP_LIBRARY, indent=2)}

Rules:
1. Choose the single best primitive for the task.
2. "scale" controls size (small cup=0.5, normal bowl=1.0, large bowl=1.5).
3. "omega" controls speed (gentle=0.5, normal=1.0, fast=1.5).
4. If the slower profile is elderly, anxious, or sleeping: reduce scale by 20% and omega by 30%.
5. Output ONLY valid JSON. No markdown, no explanation.

Output format:
{{
  "primitive_id": "stir_circle",
  "reason": "Stirring medicine in small cup for elderly slower profile",
  "parameters": {{
    "scale": 0.6,
    "omega": 0.7,
    "plane": "XY"
  }}
}}

User request: {user_request}
Scene: {scene_context}
"""

    try:
        response = dashscope.Generation.call(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            result_format="message"
        )

        if response.status_code != 200 or response.output is None:
            print(f"[LLM ERROR] Status: {response.status_code}")
            print(f"Message: {response.message}")
            return None

        content = response.output.choices[0].message.content
        plan = json.loads(content)

        # Validate required fields
        assert "primitive_id" in plan
        assert "parameters" in plan
        return plan

    except json.JSONDecodeError as e:
        print(f"[LLM ERROR] Invalid JSON: {e}")
        print(f"Raw output: {content}")
        return None
    except Exception as e:
        print(f"[LLM ERROR] {type(e).__name__}: {e}")
        return None


# ==================== ROBOT EXECUTOR ====================
def execute_dmp(robot, plan):
    """Execute a DMP plan on the robot."""

    pid = plan["primitive_id"]
    params = plan["parameters"]
    reason = plan.get("reason", "No reason provided")

    # Find primitive in library
    prim = None
    for p in DMP_LIBRARY["primitives"]:
        if p["id"] == pid:
            prim = p
            break
    if prim is None:
        print(f"[ERROR] Primitive '{pid}' not found in library")
        return False

    filepath = prim["file"]
    if not os.path.exists(filepath):
        print(f"[ERROR] DMP file not found: {filepath}")
        return False

    print(f"\n[EXECUTE] {pid}")
    print(f"  Reason: {reason}")
    print(f"  Parameters: {params}")

    # Load DMP
    dmp = RhythmicDMP()
    dmp.load(filepath)

    demo_radius_mm = np.max(dmp.r) * 1000.0
    period = 2 * np.pi / dmp.omega

    # Get current robot pose as center
    err, home = robot.GetActualTCPPose()
    if err != 0:
        print(f"[ERROR] GetActualTCPPose failed: {err}")
        return False
    home = np.array(home)
    center = home.copy()
    print(f"  Center: X={center[0]:.1f} Y={center[1]:.1f} Z={center[2]:.1f}")

    # Apply parameters
    scale = params.get("scale", 1.0)
    omega_scale = params.get("omega", 1.0)
    plane = params.get("plane", "XY")

    # Plane rotation
    plane_perm = {"XY": [0, 1, 2], "ZX": [1, 2, 0], "YZ": [2, 0, 1]}
    if plane in plane_perm:
        perm = plane_perm[plane]
        dmp.g = dmp.g[perm]
        dmp.r = dmp.r[perm]
        dmp.w = dmp.w[perm, :]
        print(f"  Plane: {plane}")

    # Generalize
    home_m = center[:3] / 1000.0
    orient_deg = center[3:6]
    goal_offset = home_m - dmp.g

    print(f"  Demo radius: {demo_radius_mm:.1f}mm")
    print(f"  Execute radius: {demo_radius_mm * scale:.1f}mm")
    print(f"  Scale: {scale:.2f}x | Omega: {omega_scale:.2f}x")

    # Generate dense trajectory, settle for 1 period, sample 8 from 2nd period
    n_dense = 1000
    dense_t = np.linspace(0, 2 * period, n_dense)
    dense_traj, _ = dmp.generate(
        dense_t, omega=dmp.omega * omega_scale, scale=scale,
        goal_offset=goal_offset, y0=home_m
    )

    second_half = dense_traj[n_dense // 2:]
    indices = np.linspace(0, len(second_half) - 1, N_WAYPOINTS, dtype=int)
    waypoints_m = second_half[indices]

    # Convert to mm + attach orientation
    waypoints = []
    for pt in waypoints_m:
        pos_mm = pt * 1000.0
        waypoints.append(np.concatenate([pos_mm, orient_deg]).tolist())

    # Print verification
    print(f"  Waypoints (first 3):")
    for i, wp in enumerate(waypoints[:3]):
        print(f"    {i + 1}: X={wp[0]:.1f} Y={wp[1]:.1f} Z={wp[2]:.1f}")

    # --- Revolution 1 ---
    print("\n  Revolution 1...")
    for pose in waypoints:
        robot.MoveL(
            desc_pos=pose, tool=0, user=0,
            vel=VEL, acc=ACC, ovl=OVL,
            blendR=BLEND_R, blendMode=BLEND_MODE
        )

    # --- Revolution 2 ---
    print("  Revolution 2...")
    for pose in waypoints:
        robot.MoveL(
            desc_pos=pose, tool=0, user=0,
            vel=VEL, acc=ACC, ovl=OVL,
            blendR=BLEND_R, blendMode=BLEND_MODE
        )

    # --- Closing circle ---
    print("  Closing circle...")
    robot.MoveL(
        desc_pos=waypoints[0], tool=0, user=0,
        vel=VEL, acc=ACC, ovl=OVL,
        blendR=BLEND_R, blendMode=BLEND_MODE
    )

    # --- Return to center ---
    mid = (np.array(waypoints[0]) + center) / 2.0
    print("  Approaching center...")
    robot.MoveL(
        desc_pos=mid.tolist(), tool=0, user=0,
        vel=VEL, acc=ACC, ovl=OVL,
        blendR=BLEND_R, blendMode=BLEND_MODE
    )

    print("  Landing at center...")
    robot.MoveL(
        desc_pos=center.tolist(), tool=0, user=0,
        vel=VEL, acc=30, ovl=OVL
    )

    print("  Done.")
    return True


# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("F5 DMP + Qwen LLM Executor")
    print(f"IP: {ROBOT_IP}")
    print("=" * 60)

    # Get user request
    user_request = input("\nEnter task (e.g., 'Stir medicine gently for anxious slower profile'): ")
    if not user_request.strip():
        user_request = "Stir the medicine gently for an anxious elderly slower profile"
        print(f"Using default: '{user_request}'")

    # Call LLM
    print("\n[LLM] Planning motion...")
    plan = call_qwen(user_request)
    if plan is None:
        print("[ABORT] LLM failed to return valid plan.")
        return

    print(f"\n[LLM] Plan received:")
    print(f"  Primitive: {plan['primitive_id']}")
    print(f"  Reason: {plan.get('reason', 'N/A')}")
    print(f"  Parameters: {plan['parameters']}")

    # Connect to robot
    print(f"\n[ROBOT] Connecting to {ROBOT_IP}...")
    robot = Robot.RPC(ROBOT_IP)
    robot.LoggerInit(output_model=0)
    robot.SetLoggerLevel(1)

    err = robot.RobotEnable(1)
    print(f"  RobotEnable: {err}")
    time.sleep(1.0)

    # Execute
    success = execute_dmp(robot, plan)

    # Cleanup logs
    if success:
        for f in os.listdir('.'):
            if f.startswith('fairino_') and f.endswith('.log'):
                try:
                    os.remove(f)
                except PermissionError:
                    pass

    print("\n" + "=" * 60)
    print("Session complete.")


if __name__ == "__main__":
    main()