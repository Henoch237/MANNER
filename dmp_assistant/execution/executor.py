import time
import numpy as np
import yaml
import py_trees
import py_trees.trees

from fairino import Robot
from dmp_assistant import paths
from dmp_assistant.skills.gripper import gripper_close, gripper_open
from dmp_assistant.skills.motion import movel, movej
from dmp_assistant.skills.robot_state import get_current_pose, wait_for_state
from dmp_assistant.skills.dmp import RhythmicDMP
from dmp_assistant.skills.pour import pour as pour_skill
from dmp_assistant.skills.hand_over import hand_over as handover_skill

with open(paths.SCENE_YAML, "r", encoding="utf-8") as f:
    SCENE = yaml.safe_load(f)["poses"]

# Home / safe transit pose, read from the scene (was a hard-coded copy in
# SimplePickAndPlace.py that had to be kept in sync by hand).
INITIAL_POSE = SCENE["initial_pose"]["value"]


# ================================================================
# BEHAVIOUR LEAF NODES — one per action type
# Each wraps a skill call and reports SUCCESS / FAILURE
# ================================================================

class GoToPose(py_trees.behaviour.Behaviour):
    def __init__(self, name, executor, pose_ref, speed=30, accel=100):
        super().__init__(name)
        self.executor = executor
        self.pose_ref = pose_ref
        self.speed = speed
        self.accel = accel

    def update(self):
        try:
            pose = self.executor._resolve(self.pose_ref)
            movel(self.executor.robot, pose, vel=self.speed, acc=self.accel)
            self.executor.state["location"] = self.pose_ref
            self.executor._record_speed(self.speed, self.accel)
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            print(f"[BT ERROR] GoToPose failed: {e}")
            return py_trees.common.Status.FAILURE


class GripperOpen(py_trees.behaviour.Behaviour):
    def __init__(self, name, executor):
        super().__init__(name)
        self.executor = executor

    def update(self):
        gripper_open(self.executor.robot)
        time.sleep(2)
        self.executor.state["gripper"] = "open"
        return py_trees.common.Status.SUCCESS


class GripperClose(py_trees.behaviour.Behaviour):
    def __init__(self, name, executor):
        super().__init__(name)
        self.executor = executor

    def update(self):
        gripper_close(self.executor.robot)
        time.sleep(2)
        self.executor.state["gripper"] = "closed"
        return py_trees.common.Status.SUCCESS


class PickAction(py_trees.behaviour.Behaviour):
    def __init__(self, name, executor, object_id, speed=30, accel=100,
                 gripper_force=100):
        super().__init__(name)
        self.executor = executor
        self.object_id = object_id
        self.speed = speed
        self.accel = accel
        self.gripper_force = gripper_force

    def update(self):
        try:
            self.executor._pick(self.object_id, self.speed, self.accel,
                                self.gripper_force)
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            print(f"[BT ERROR] Pick failed: {e}")
            return py_trees.common.Status.FAILURE


class PlaceAction(py_trees.behaviour.Behaviour):
    def __init__(self, name, executor, location_id, speed=30, accel=100):
        super().__init__(name)
        self.executor = executor
        self.location_id = location_id
        self.speed = speed
        self.accel = accel

    def update(self):
        try:
            self.executor._place(self.location_id, self.speed, self.accel)
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            print(f"[BT ERROR] Place failed: {e}")
            return py_trees.common.Status.FAILURE


class PickAndPlaceAction(py_trees.behaviour.Behaviour):
    def __init__(self, name, executor, object_id, destination_id, speed=30,
                 accel=100, gripper_force=100):
        super().__init__(name)
        self.executor = executor
        self.object_id = object_id
        self.destination_id = destination_id
        self.speed = speed
        self.accel = accel
        self.gripper_force = gripper_force

    def update(self):
        try:
            self.executor._pick_and_place(self.object_id, self.destination_id,
                                          self.speed, self.accel,
                                          self.gripper_force)
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            print(f"[BT ERROR] PickAndPlace failed: {e}")
            return py_trees.common.Status.FAILURE


class StirAction(py_trees.behaviour.Behaviour):
    def __init__(self, name, executor, cup_id, omega_scale=1.0, revolutions=2,
                 speed=30, accel=100):
        super().__init__(name)
        self.executor = executor
        self.cup_id = cup_id
        self.omega_scale = omega_scale
        self.revolutions = revolutions
        self.speed = speed
        self.accel = accel

    def update(self):
        try:
            self.executor._stir(self.cup_id, self.omega_scale, self.revolutions,
                                self.speed, self.accel)
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            print(f"[BT ERROR] Stir failed: {e}")
            return py_trees.common.Status.FAILURE


class PourAction(py_trees.behaviour.Behaviour):
    def __init__(self, name, executor, omega_scale=1.0):
        super().__init__(name)
        self.executor = executor
        self.omega_scale = omega_scale

    def update(self):
        try:
            self.executor._pour(self.omega_scale)
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            print(f"[BT ERROR] Pour failed: {e}")
            return py_trees.common.Status.FAILURE


class HandOverAction(py_trees.behaviour.Behaviour):
    def __init__(self, name, executor, confirm_dwell_s=3.0, speed=30, accel=100):
        super().__init__(name)
        self.executor = executor
        self.confirm_dwell_s = confirm_dwell_s
        self.speed = speed
        self.accel = accel

    def update(self):
        try:
            self.executor._hand_over(self.confirm_dwell_s, self.speed, self.accel)
            return py_trees.common.Status.SUCCESS
        except Exception as e:
            print(f"[BT ERROR] HandOver failed: {e}")
            return py_trees.common.Status.FAILURE


# ================================================================
# EXECUTOR (BT-based)
# Same public interface as the old flat executor.
# ================================================================

class Executor:
    def __init__(self, ip="192.168.58.4", locator=None):
        self.robot = Robot.RPC(ip)
        wait_for_state(self.robot)
        self.state = {
            "location": "unknown",
            "holding": None,
            "gripper": "unknown",
            # Believed record of the last commanded motion profile.
            # Placeholder for future faster/slower adjustment factors.
            "last_speed": 30,
            "last_accel": 100
        }
        # Camera + YOLO. The web server passes one built on its live camera;
        # otherwise it is opened on the first pick.
        self._locator = locator
        self._sync_location()

    # ------------------------------------------------------------------
    # Helpers (unchanged from flat executor)
    # ------------------------------------------------------------------
    def _sync_location(self):
        err, pose = get_current_pose(self.robot)
        if err == 0 and self._near(pose, INITIAL_POSE):
            self.state["location"] = "initial_pose"

    def _near(self, a, b, tol=5.0):
        return all(abs(x - y) < tol for x, y in zip(a[:3], b[:3]))

    def _resolve(self, ref):
        if isinstance(ref, list):
            return ref
        parts = ref.split(".")
        node = SCENE
        for p in parts:
            node = node[p]
        while isinstance(node, dict) and "ref" in node:
            ref_path = node["ref"].split(".")
            node = SCENE
            for p in ref_path:
                node = node[p]
        # ADDED: computed approach poses — "offset_z_mm above derived_from".
        # Lets any object declare only its grasp_pose; the approach pose
        # (world-Z offset) is generated here, one convention, one place.
        if isinstance(node, dict) and "derived_from" in node and "offset_z_mm" in node:
            base = list(self._resolve(node["derived_from"]))
            base[2] += float(node["offset_z_mm"])
            return base
        if isinstance(node, dict) and "value" in node:
            return node["value"]
        if isinstance(node, list):
            return node
        raise ValueError(f"Cannot resolve pose reference: '{ref}' -> {node}")

    def _record_speed(self, speed, accel):
        # Believed record of the last commanded profile (rule 18 output).
        self.state["last_speed"] = speed
        self.state["last_accel"] = accel

    def _go(self, pose_ref, speed=30, accel=100):
        pose = self._resolve(pose_ref)
        movel(self.robot, pose, vel=speed, acc=accel)
        self.state["location"] = pose_ref
        self._record_speed(speed, accel)

    # ------------------------------------------------------------------
    # Skill methods (unchanged)
    # ------------------------------------------------------------------
    def _gripper_open(self):
        gripper_open(self.robot)
        time.sleep(2)
        self.state["gripper"] = "open"

    def _gripper_close(self, force=100):
        gripper_close(self.robot, force=force)
        time.sleep(2)
        self.state["gripper"] = "closed"

    # ------------------------------------------------------------------
    # Camera pick: objects declared in scene.yaml with `detect: <class>`
    # have no stored pose; the camera finds them at pick time.
    # ------------------------------------------------------------------
    def _is_detected(self, object_id):
        node = SCENE.get(object_id)
        return isinstance(node, dict) and "detect" in node

    @property
    def locator(self):
        if self._locator is None:
            from dmp_assistant.perception.locator import ObjectLocator
            self._locator = ObjectLocator()
        return self._locator

    def _flange_pose(self):
        err, pose = self.robot.GetActualToolFlangePose(0)
        if err != 0:
            raise RuntimeError(f"GetActualToolFlangePose failed, error {err}")
        return list(pose)

    def _movel_checked(self, pose, speed, accel, label):
        err = movel(self.robot, pose, vel=speed, acc=accel)
        if err != 0:
            raise RuntimeError(f"MoveL to {label} failed, error {err}")

    def _gripper_is_empty(self):
        err, fault, pos = self.robot.GetGripperCurPosition()
        if err != 0 or fault:
            print(f"[PICK] gripper position unreadable (err={err}, fault={fault})")
            return False
        empty = pos <= self.locator.cfg["grasp"]["empty_gripper_pos"]
        print(f"[PICK] gripper position after closing: {pos}"
              f"{' -> nothing grasped' if empty else ''}")
        return empty

    def _refine_at_hover(self, target):
        """From hover height, re-measure the object and correct XY. The
        top height and yaw stay from the scan (seen from farther away)."""
        from dmp_assistant.perception.locator import LocateError
        try:
            again = self.locator.locate(target.cls, self._flange_pose(), near=target.top)
        except LocateError as e:
            print(f"[PICK] hover re-check failed ({e}); keeping the scan estimate.")
            return target
        shift = float(np.linalg.norm(again.top[:2] - target.top[:2]))
        limit = self.locator.cfg["grasp"]["max_correction_mm"]
        if shift > limit:
            raise LocateError(f"hover correction {shift:.1f} mm > {limit} mm: "
                              f"inconsistent reading, not grasping.")
        print(f"[PICK] hover correction: {shift:.1f} mm")
        target.top = np.array([again.top[0], again.top[1], target.top[2]])
        return target

    def _pick_detected(self, object_id, speed=30, accel=100, gripper_force=100):
        cls = SCENE[object_id]["detect"]
        g = self.locator.cfg["grasp"]
        print(f"[PICK] camera pick of '{object_id}' (YOLO class '{cls}')")

        self._go(self.locator.cfg["locate"]["scan_pose"], speed, accel)
        self._gripper_open()
        scan = self._flange_pose()
        target = self.locator.locate(cls, scan)
        hover, grasp = self.locator.grasp_poses(target, scan)

        self._movel_checked(hover, speed, accel, f"{object_id} hover")
        self.state["location"] = f"{object_id}.hover"
        if g["refine_at_hover"]:
            target = self._refine_at_hover(target)
            new_hover, grasp = self.locator.grasp_poses(target, scan)
            if np.linalg.norm(np.subtract(new_hover[:2], hover[:2])) > 1.0:
                hover = new_hover
                self._movel_checked(hover, speed, accel, f"{object_id} hover (corrected)")

        print(f"[PICK] grasp flange pose {[round(v, 1) for v in grasp]}")
        self._movel_checked(grasp, speed, accel, f"{object_id} grasp")
        self._gripper_close(force=gripper_force)
        if g["verify_grasp"] and self._gripper_is_empty():
            self._gripper_open()
            self._movel_checked(hover, speed, accel, f"{object_id} hover")
            raise RuntimeError(f"'{object_id}' not grasped (gripper closed on nothing).")
        self.state["holding"] = object_id
        self._movel_checked(hover, speed, accel, f"{object_id} retract")
        self.state["location"] = f"{object_id}.hover"
        self._record_speed(speed, accel)

    def _pick(self, object_id, speed=30, accel=100, gripper_force=100):
        """Every pick goes through the camera: there is no stored-pose
        fallback, so the arm never heads for a pose the object left."""
        if not self._is_detected(object_id):
            detected = sorted(k for k in SCENE if self._is_detected(k))
            raise RuntimeError(f"'{object_id}' cannot be picked: picks use the camera and "
                               f"it has no YOLO class. Camera objects: {detected}")
        return self._pick_detected(object_id, speed, accel, gripper_force)

    def _place(self, location_id, speed=30, accel=100):
        place = self._resolve(f"{location_id}.place_pose")
        self._go("initial_pose.value", speed, accel)
        movel(self.robot, place, vel=speed, acc=accel)
        self._gripper_open()
        self.state["holding"] = None
        self._record_speed(speed, accel)

    def _pick_and_place(self, object_id, destination_id, speed=30, accel=100,
                        gripper_force=100):
        self._pick(object_id, speed, accel, gripper_force)
        self._place(destination_id, speed, accel)
        self._go("initial_pose.value", speed, accel)

    def _stir(self, cup_id, omega_scale=1.0, revolutions=2, speed=30, accel=100):
        # speed/accel: the transit move to the cup (register-resolved).
        # omega_scale: the stirring tempo itself (the DMP's knob).
        # The waypoint playback vel=30 below is a hard execution ceiling,
        # not an adaptation channel - the DMP sets the tempo.
        import numpy as np
        import os
        center_pose = self._resolve(f"{cup_id}.center_pose")
        self._gripper_close()
        movel(self.robot, center_pose, vel=speed, acc=accel)
        self._record_speed(speed, accel)
        dmp_file = str(paths.STIRRING_DMP)
        if not os.path.exists(dmp_file):
            print(f"[ERROR] {dmp_file} not found")
            return
        dmp = RhythmicDMP()
        dmp.load(dmp_file)
        period = 2 * np.pi / dmp.omega
        center = np.array(center_pose)
        home_m = center[:3] / 1000.0
        orient_deg = center[3:6]
        goal_offset = home_m - dmp.g
        n_dense = 1000
        dense_t = np.linspace(0, 2 * period, n_dense)
        dense_traj, _ = dmp.generate(
            dense_t, omega=dmp.omega * omega_scale,
            scale=1.0, goal_offset=goal_offset, y0=home_m
        )
        second_half = dense_traj[n_dense // 2:]
        indices = np.linspace(0, len(second_half) - 1, 8, dtype=int).tolist()
        waypoints_m = second_half[indices]
        for rev in range(revolutions):
            print(f"  Revolution {rev + 1}...")
            for pt in waypoints_m:
                wp = np.concatenate([pt * 1000.0, orient_deg]).tolist()
                self.robot.MoveL(desc_pos=wp, tool=0, user=0,
                                 vel=30, acc=100, ovl=50,
                                 blendR=15.0, blendMode=0)
        wp0 = np.concatenate([waypoints_m[0] * 1000.0, orient_deg]).tolist()
        self.robot.MoveL(desc_pos=wp0, tool=0, user=0,
                         vel=30, acc=100, ovl=50,
                         blendR=15.0, blendMode=0)
        movel(self.robot, center_pose, vel=speed, acc=accel)
        self.state["location"] = f"{cup_id}.center_pose"
        print("Stir complete.")

    def _pour(self, omega_scale=1.0):
        """In-place pour from the current pose via the learned pouring DMP.
        omega_scale time-scales the learned tilt cycle (register-resolved:
        1.0 = recorded tempo, 0.7 = gentle). pour() already waits for both
        the waypoint queue AND the return-to-start segment to settle, so it
        is safe for the next BT node to run immediately.

        pour() first checks the wrist (J4-6) against the recorded start
        configuration and realigns it if needed; if that is impossible it
        pours by rotating joint 6 instead. Both count as a pour; None means
        nothing was poured."""
        dev = pour_skill(self.robot, speed=omega_scale, confirm=False)
        if dev is None:
            raise RuntimeError("pour aborted")
        # pour returns to the pose it started from -> location label unchanged
        print(f"Pour complete. Position deviation {dev:.2f} mm.")

    def _hand_over(self, confirm_dwell_s=3.0, speed=30, accel=100):
        """Robot-to-human handover of the believed-held object.
        The skill internally: guards/freeze wrist orientation, verifies a
        full stop before the dwell, opens the gripper only after the queue
        has drained, and retreats to the observation pose."""
        if self.state["holding"] is None:
            print("[EXECUTOR WARN] hand_over requested with nothing believed "
                  "held - proceeding anyway.")
        ok = handover_skill(self.robot,
                            confirm_dwell_s=confirm_dwell_s,
                            speed=speed, accel=accel)
        if not ok:
            raise RuntimeError("hand_over failed")
        self.state["holding"] = None
        self.state["gripper"] = "open"
        self.state["location"] = "handover.observation_pose"
        print("HandOver complete.")

    # ------------------------------------------------------------------
    # BT COMPILATION
    # ------------------------------------------------------------------
    def compile_plan(self, plan):
        """
        Converts a JSON plan into a py_trees Behaviour Tree.
        Returns the root node.
        """
        # PLACEHOLDER: In the future, wrap this Sequence inside a Selector
        # so that if the whole plan fails, a recovery branch can run.
        # root = py_trees.composites.Selector(name="RootWithFallback", memory=False)
        # plan_seq = py_trees.composites.Sequence(name="PlanSequence", memory=True)
        # recovery = py_trees.composites.Sequence(name="Recovery", memory=True)
        # root.add_children([plan_seq, recovery])

        root = py_trees.composites.Sequence(name="PlanSequence", memory=True)

        for i, step in enumerate(plan):
            action = step["action"]
            node_name = f"{i + 1}_{action}"

            if action == "pick_and_place":
                root.add_child(PickAndPlaceAction(
                    node_name, self,
                    step["object_id"], step["destination_id"],
                    step.get("speed", 30), step.get("accel", 100),
                    step.get("gripper_force", 100)
                ))
            elif action == "pick":
                root.add_child(PickAction(node_name, self, step["object_id"],
                                          step.get("speed", 30),
                                          step.get("accel", 100),
                                          step.get("gripper_force", 100)))
            elif action == "place":
                root.add_child(PlaceAction(node_name, self, step["location_id"],
                                           step.get("speed", 30), step.get("accel", 100)))
            elif action == "stir":
                root.add_child(StirAction(
                    node_name, self,
                    step.get("cup_id", "coffee_cup"),
                    step.get("omega_scale", 1.0),
                    step.get("revolutions", 2),
                    step.get("speed", 30), step.get("accel", 100)
                ))
            elif action == "pour":
                root.add_child(PourAction(node_name, self,
                                          step.get("omega_scale", 1.0)))
            elif action == "hand_over":
                root.add_child(HandOverAction(
                    node_name, self,
                    step.get("confirm_dwell_s", 3.0),
                    step.get("speed", 30), step.get("accel", 100)
                ))
            elif action == "go_to":
                root.add_child(GoToPose(node_name, self, step["target_pose"],
                                        step.get("speed", 30), step.get("accel", 100)))
            elif action == "gripper_open":
                root.add_child(GripperOpen(node_name, self))
            elif action == "gripper_close":
                root.add_child(GripperClose(node_name, self))
            else:
                print(f"[WARNING] Unknown action in BT compile: {action}")

        return root

    # ------------------------------------------------------------------
    # PUBLIC API — unchanged from flat executor
    # ------------------------------------------------------------------
    def run_plan(self, plan):
        """
        Compile JSON plan into BT, tick until completion.
        Returns True when the whole tree succeeded.
        """
        print("\n[EXECUTOR BT] Compiling plan into Behaviour Tree...")
        root = self.compile_plan(plan)

        # PLACEHOLDER: Future — add a Parallel node here to monitor
        # e-stop or human proximity while the plan runs.
        # monitor = py_trees.composites.Parallel(name="Monitor", policy=py_trees.common.ParallelPolicy.SuccessOnAll())
        # monitor.add_children([root, SafetyMonitor("EStopWatcher")])
        # tree = py_trees.trees.BehaviourTree(monitor)

        tree = py_trees.trees.BehaviourTree(root)
        tree.setup()

        print("[EXECUTOR BT] Ticking tree...")
        try:
            while tree.root.status not in (
                    py_trees.common.Status.SUCCESS,
                    py_trees.common.Status.FAILURE
            ):
                tree.tick()
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n[EXECUTOR BT] Interrupted by user.")
            tree.interrupt()

        final = tree.root.status
        tree.shutdown()

        if final == py_trees.common.Status.SUCCESS:
            print("[EXECUTOR BT] Plan completed successfully.")
            return True
        print("[EXECUTOR BT] Plan failed.")
        return False

        return final == py_trees.common.Status.SUCCESS