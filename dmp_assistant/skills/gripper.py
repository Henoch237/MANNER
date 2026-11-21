from fairino import Robot
import time



# MoveGripper(index, pos, vel, force, maxtime, block, type, rotNum, rotVel, rotTorque)
# vel and force are percents; force=100 is the gripper's maximum.
def gripper_close(robot, force=100, speed=100, max_time=1000):
    error = robot.MoveGripper(1, 0, speed, force, max_time, 0, 0, 0, 0, 0)
    print("MoveGripper (close)", error)
    time.sleep(1)
    robot.ResetAllError()
    return error


def gripper_open(robot, force=100, speed=100, max_time=1000):
    error = robot.MoveGripper(1, 10, speed, force, max_time, 0, 0, 0, 0, 0)
    print("MoveGripper (open)", error)
    time.sleep(1)
    robot.ResetAllError()
    return error

