"""One user's conversation with the robot, independent of the front-end.

The terminal launcher and the web app both drive a Session: text in,
Reply out; a returned plan waits in `pending_plan` until execute() or
discard(). Everything the planner/executor print stays on stdout.
"""
from dataclasses import dataclass, field

HELP = """Commands:
  state           show what the robot currently believes
  user <name>     switch the session owner (resets the drift)
  reset           new session (clears LLM history + drift)
  reset prefs     delete the current user's stored preference
  quit / exit     close the session
Anything else is a robot request - phrased any way you like. Manner
("softly", "easy does it"), speed wishes ("take it down a notch") and
persistence wishes ("from now on") are understood, not keyword-matched."""


@dataclass
class Reply:
    # "command": a session command ran; "plan": a plan awaits confirmation;
    # "no_plan": the planner refused / only updated session state;
    # "quit": the user asked to close the session.
    kind: str
    plan: list = field(default=None)


class Session:
    def __init__(self, planner, executor_factory, allow_user_switch=True):
        """executor_factory() -> Executor; called on the first execution so
        the robot connects lazily (and can be shared between sessions)."""
        self.planner = planner
        self._executor_factory = executor_factory
        self._executor = None
        self.allow_user_switch = allow_user_switch
        self.pending_plan = None

    @property
    def user(self):
        return self.planner.get_current_user()

    @property
    def executor(self):
        return self._executor

    def handle(self, text):
        text = text.strip()
        cmd = text.lower()
        # Any new input answers "no" to a plan still waiting for confirmation -
        # except a yes/no to the planner's "remember this speed?" question,
        # which comes together with that plan.
        if self.pending_plan is not None and not self._answers_store_question(cmd):
            self.discard()

        if cmd in ("quit", "exit"):
            return Reply("quit")
        if cmd.startswith("user "):
            if self.allow_user_switch:
                self.planner.set_current_user(cmd.split(None, 1)[1].strip())
            else:
                print("[SESSION] user switching is disabled here - the session "
                      "owner is the authenticated person.")
            return Reply("command")
        if cmd == "state":
            print(self.describe_state())
            return Reply("command")
        if cmd == "reset":
            self.planner.reset_session()
            print("[SESSION] new session - history and drift cleared.")
            return Reply("command")
        if cmd == "reset prefs":
            self.planner.reset_preferences()
            return Reply("command")

        # Plan (session-aware: history + drift persist between orders)
        plan = self.planner.plan(text)
        if plan is None:
            return Reply("no_plan")
        print("\n[PLANNER] Final plan:")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")
        self.pending_plan = plan
        return Reply("plan", plan)

    def _answers_store_question(self, cmd):
        p = self.planner
        return p.pending_store is not None and \
            cmd.strip(".!,") in (p._YES_WORDS | p._NO_WORDS)

    def execute(self):
        """Run the pending plan on the robot (blocking). -> True on success."""
        plan, self.pending_plan = self.pending_plan, None
        if plan is None:
            print("[SESSION] no plan to execute.")
            return False
        if self._executor is None:
            self._executor = self._executor_factory()
        ok = self._executor.run_plan(plan)
        print(self.describe_state())
        return ok

    def discard(self):
        if self.pending_plan is not None:
            self.pending_plan = None
            print("Skipped (stays in conversation memory).")

    def state(self):
        return None if self._executor is None else dict(self._executor.state)

    def describe_state(self):
        s = self.state()
        if s is None:
            return "[STATE] not connected to the robot yet (no execution this session)."
        return (f"[STATE] location: {s.get('location')} | holding: {s.get('holding')} "
                f"| gripper: {s.get('gripper')} | last: {s.get('last_speed')}/{s.get('last_accel')}")
