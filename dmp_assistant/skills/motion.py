from fairino import Robot
import time

# ================================================================
# PLACEHOLDERS — Fill these in before running
# ================================================================
JP = [87.0824203491211, -84.28558349609375, 83.52220153808594, -97.68281555175781,
      -92.49574279785156, 85.70880889892578]  # Joint positions for MoveJ
EEP = [68.66001892089844, -557.06982421875, 497.0107727050781,
       -176.8755645751953, 8.235721588134766, 91.41429901123047]  # Cartesian pose [X,Y,Z,Rx,Ry,Rz] for IK / MoveL


# ================================================================


def _clear_all(robot):
    """Aggressively clear controller error log."""
    try:
        robot.ClearError()
    except Exception:
        pass
    time.sleep(0.1)
    robot.ResetAllError()
    time.sleep(0.2)
    robot.ResetAllError()
    time.sleep(0.1)


def movej(robot, joint_pos, vel=30, acc=100):
    """MoveJ: joint-space motion. vel/acc are percents (1-100)."""
    _clear_all(robot)
    time.sleep(0.2)
    error = robot.MoveJ(joint_pos, tool=0, user=0, vel=vel, acc=acc)
    print(f"MoveJ return: {error}")
    time.sleep(0.5)
    _clear_all(robot)
    return error


def movel(robot, cartesian_pose, vel=30, acc=100):
    """MoveL: linear cartesian motion. vel/acc are percents (1-100).

    vel/acc are the framework's speed knobs: the planner resolves them
    from the recipient's register (people.yaml, rule 18) and the executor
    passes them straight through. Single tuning point for all skills."""
    _clear_all(robot)
    time.sleep(0.2)
    error = robot.MoveL(cartesian_pose, tool=0, user=0, vel=vel, acc=acc)
    print(f"MoveL return: {error} (vel={vel}, acc={acc})")
    time.sleep(0.5)
    _clear_all(robot)
    return error


def calc_ik(robot, desc_pos, config=-1):
    """Calculate inverse kinematics for a given cartesian pose."""
    error, ik_sol = robot.GetInverseKin(type=0, desc_pos=desc_pos, config=config)
    if error == 0:
        print(f"IK solution: {ik_sol}")
    else:
        print(f"IK calculation failed, error: {error}")
    return error, ik_sol


# ================================================================
# Example usage
# ================================================================
if __name__ == "__main__":
    robot = Robot.RPC('192.168.85.130')

    #print("--- IK Test ---")
    #err, ik = calc_ik(robot, EEP)

    #print("\n--- MoveJ Test ---")
    #movej(robot, JP)

    print("\n--- MoveL Test ---")
    movel(robot, EEP)
