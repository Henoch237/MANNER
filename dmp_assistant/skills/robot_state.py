import time

from fairino import Robot


def wait_for_state(robot, timeout_s=5.0):
    """Block until the SDK has received its first robot state packet.

    Right after Robot.RPC() the SDK's robot_state_pkg is still the ctypes
    CLASS, not data, so every pose getter fails with "'_ctypes.CField'
    object is not subscriptable" until the first packet arrives."""
    t0 = time.time()
    while isinstance(robot.robot_state_pkg, type):
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"no robot state received within {timeout_s:.1f} s "
                               f"(robot on? network to the controller?)")
        time.sleep(0.05)


def get_current_joints(robot):
    error, joints = robot.GetActualJointPosDegree()
    if error == 0:
        print(f"Joints (deg): {joints}")
    else:
        print(f"GetActualJointPosDegree failed, error: {error}")
    return error, joints


def get_current_pose(robot):
    error, pose = robot.GetActualTCPPose(0)
    if error == 0:
        print(f"TCP Pose: {pose}")
    else:
        print(f"GetActualTCPPose failed, error: {error}")
    return error, pose


# --- Example usage ---
if __name__ == "__main__":
    robot = Robot.RPC('192.168.58.4')
    err_j, joints = get_current_joints(robot)
    err_p, pose = get_current_pose(robot)
    #INITIAL_POSE=[67.4990005493164, -317.0991516113281, 573.6724853515625,-161.8025360107422, 86.35399627685547, 108.64768981933594]
    #robot.MoveL(INITIAL_POSE, tool=0, user=0, vel=30)