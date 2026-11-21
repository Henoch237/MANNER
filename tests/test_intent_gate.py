"""Offline checks for the intent-header architecture: store confirmation
gate (R4/R5), intent.recipient merge, adjustment drift, manner mapping.
dashscope is STUBBED - no network. Canned LLM envelopes are queued per
scenario; the deterministic planner logic is what is under test."""
import json
import sys
import types

# ---- stub dashscope BEFORE importing planner ----
fake = types.ModuleType("dashscope")
fake.api_key = None


class _Resp:
    def __init__(self, content):
        self.status_code = 200
        self.output = types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=content))])


class _Gen:
    queue = []

    @staticmethod
    def call(**kw):
        return _Resp(_Gen.queue.pop(0))


fake.Generation = _Gen
sys.modules["dashscope"] = fake

import os
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root
from dmp_assistant.planning.planner import Planner

SCRATCH = "/tmp/scratch_prefs_gate.yaml"
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, cond, detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


def envelope(manner="none", adjustment="none", store=False,
             recipient="none", plan=None):
    return json.dumps({
        "intent": {"manner": manner, "adjustment": adjustment,
                   "store": store, "recipient": recipient},
        "plan": plan or []})


def fresh_planner():
    with open(SCRATCH, "w") as f:
        f.write("stored: {}\n")
    return Planner(api_key="fake", preferences_path=SCRATCH)


def stored_mult(name):
    with open(SCRATCH) as f:
        return (yaml.safe_load(f).get("stored", {})
                .get(name, {}).get("mult", 1.0))


# 0. schema: recipient enum is the roster + none
p = fresh_planner()
rec_enum = p.schema["properties"]["intent"]["properties"]["recipient"]["enum"]
check("schema recipient enum = roster + none",
      rec_enum == sorted(p.people["people"]) + ["none"], str(rec_enum))

# 1. adjustment-only: drift moves, empty plan is a valid session command
p = fresh_planner()
_Gen.queue = [envelope(adjustment="down")]
r = p.plan("slower")
check("adjustment-only -> None return, drift 0.8",
      r is None and abs(p.session_drift - 0.8) < 1e-9,
      f"drift={p.session_drift}")

# 2. store gate: opens with the captured drift, yes writes
_Gen.queue = [envelope(store=True, plan=[{"action": "pour"}])]
r = p.plan("pour, and remember this speed")
check("store intent -> plan still returned (pour, omega 0.7x0.8=0.56)",
      r is not None and r[0]["action"] == "pour"
      and abs(r[0]["omega_scale"] - 0.56) < 1e-9, str(r))
check("gate opened with would_be 0.8",
      p.pending_store is not None
      and abs(p.pending_store["would_be"] - 0.8) < 1e-9,
      str(p.pending_store))
r = p.plan("yes")
check("yes -> written 0.8, no API call needed",
      r is None and abs(stored_mult("john") - 0.8) < 1e-9,
      f"stored={stored_mult('john')}")

# 3. store -> no -> nothing written
p = fresh_planner()
_Gen.queue = [envelope(adjustment="up"),
              envelope(store=True, plan=[{"action": "pour"}])]
p.plan("faster")
p.plan("pour, keep it like this")
p.plan("no")
check("no -> nothing written", abs(stored_mult("john") - 1.0) < 1e-9)

# 4. store -> unrelated request: discarded, request processed normally
p = fresh_planner()
_Gen.queue = [envelope(adjustment="up"),
              envelope(store=True, plan=[{"action": "pour"}]),
              envelope(plan=[{"action": "stir", "cup_id": "coffee_cup"}])]
p.plan("faster")
p.plan("pour, from now on")
r = p.plan("stir the coffee")
check("unrelated request -> discarded + processed",
      r is not None and r[0]["action"] == "stir"
      and abs(stored_mult("john") - 1.0) < 1e-9, str(r))

# 5. store at drift 1.0 -> no gate
p = fresh_planner()
_Gen.queue = [envelope(store=True, plan=[{"action": "pour"}])]
p.plan("pour, remember that")
check("drift 1.0 -> no gate opened", p.pending_store is None)

# 6. R5: store naming another person -> hard fail, no gate
p = fresh_planner()
_Gen.queue = [envelope(adjustment="up"),
              envelope(store=True, recipient="sarah",
                       plan=[{"action": "pour"}])]
p.plan("faster")
p.plan("pour, store that for sarah")
check("R5 -> no gate, nothing written",
      p.pending_store is None and abs(stored_mult("john") - 1.0) < 1e-9
      and abs(stored_mult("sarah") - 1.0) < 1e-9)

# 7. owner switch discards a pending store
p = fresh_planner()
_Gen.queue = [envelope(adjustment="up"),
              envelope(store=True, plan=[{"action": "pour"}])]
p.plan("faster")
p.plan("pour, remember that")
p.set_current_user("sarah")
check("owner switch -> pending discarded", p.pending_store is None)
check("nothing written after switch", abs(stored_mult("john") - 1.0) < 1e-9)

# 8. intent.recipient only (name not in text) -> recipient's register
p = fresh_planner()
_Gen.queue = [envelope(recipient="sarah",
                       plan=[{"action": "stir", "cup_id": "coffee_cup"}])]
r = p.plan("stir")
check("intent recipient sarah -> 45/45, omega 1.0",
      r[0]["speed"] == 45 and r[0]["accel"] == 45
      and r[0]["omega_scale"] == 1.0, str(r))

# 9. text scan and intent agree -> same result
p = fresh_planner()
_Gen.queue = [envelope(recipient="sarah",
                       plan=[{"action": "stir", "cup_id": "coffee_cup"}])]
r = p.plan("stir for sarah")
check("text+intent agree on sarah -> 45/45",
      r[0]["speed"] == 45 and r[0]["accel"] == 45, str(r))

# 10. text/intent conflict -> ask, no plan
p = fresh_planner()
_Gen.queue = [envelope(recipient="mary",
                       plan=[{"action": "stir", "cup_id": "coffee_cup"}])]
r = p.plan("stir for sarah")
check("conflict sarah-vs-mary -> refusal (ask)", r is None, str(r))

# 11. manner gentle via header -> slower profile values
p = fresh_planner()
_Gen.queue = [envelope(manner="gentle",
                       plan=[{"action": "stir", "cup_id": "coffee_cup"}])]
r = p.plan("stir softly")
check("manner gentle -> 25/25, omega 0.7",
      r[0]["speed"] == 25 and r[0]["accel"] == 25
      and r[0]["omega_scale"] == 0.7, str(r))

# 12. manner gentle vs recipient sarah -> slower profile wins
p = fresh_planner()
_Gen.queue = [envelope(manner="gentle", recipient="sarah",
                       plan=[{"action": "stir", "cup_id": "coffee_cup"}])]
r = p.plan("stir for sarah, gently")
check("gentle + sarah -> slower register wins (25/25, 0.7)",
      r[0]["speed"] == 25 and r[0]["omega_scale"] == 0.7, str(r))

# 13. bad header (invented recipient) -> validation failure
p = fresh_planner()
_Gen.queue = [envelope(recipient="gertrude",
                       plan=[{"action": "stir", "cup_id": "coffee_cup"}])]
r = p.plan("stir")
check("invented recipient -> validation refuses", r is None, str(r))

# 14. unknown 'for X' still pre-API (no API call spent)
p = fresh_planner()
_Gen.queue = []   # would crash if an API call were attempted
r = p.plan("stir for gertrude")
check("unknown person -> pre-API refusal", r is None)

# 15. manner "well" (thoroughness, not speed) -> revolutions 4, register untouched
p = fresh_planner()
_Gen.queue = [envelope(manner="well",
                       plan=[{"action": "stir", "cup_id": "coffee_cup"}])]
r = p.plan("stir well")
check("manner well -> revolutions 4, slower profile kept",
      r is not None and r[0]["revolutions"] == 4 and r[0]["speed"] == 40
      and r[0]["omega_scale"] == 0.7, str(r))

# 16. explicit count is taken as stated (an emitted value IS the user's wish)
p = fresh_planner()
_Gen.queue = [envelope(plan=[{"action": "stir", "cup_id": "coffee_cup",
                              "revolutions": 5}])]
r = p.plan("stir 5 times")
check("explicit count 5 -> revolutions 5",
      r is not None and r[0]["revolutions"] == 5, str(r))

# 17. explicit count is clamped to the declared bounds
p = fresh_planner()
_Gen.queue = [envelope(plan=[{"action": "stir", "cup_id": "coffee_cup",
                              "revolutions": 10}])]
r = p.plan("stir 10 times")
check("count 10 -> clamped to 6",
      r is not None and r[0]["revolutions"] == 6, str(r))

# 18. explicit count wins over manner "well" (presence-based resolution)
p = fresh_planner()
_Gen.queue = [envelope(manner="well",
                       plan=[{"action": "stir", "cup_id": "coffee_cup",
                              "revolutions": 2}])]
r = p.plan("stir 2 times, and stir it well")
check("explicit 2 + manner well -> 2 (explicit count wins)",
      r is not None and r[0]["revolutions"] == 2, str(r))

# 19. plain stir (field omitted) -> default 2
p = fresh_planner()
_Gen.queue = [envelope(plan=[{"action": "stir", "cup_id": "coffee_cup"}])]
r = p.plan("stir the coffee")
check("no thoroughness said -> revolutions 2",
      r is not None and r[0]["revolutions"] == 2, str(r))

# 20. gripper force: camera object without handling_class -> standard ceiling
p = fresh_planner()
_Gen.queue = [envelope(plan=[{"action": "pick", "object_id": "bottle"}])]
r = p.plan("pick the bottle")
check("pick bottle (standard) -> gripper_force 100",
      r is not None and r[0]["gripper_force"] == 100, str(r))

# 21. gentle pick steps force down within the class band
p = fresh_planner()
_Gen.queue = [envelope(manner="gentle",
                       plan=[{"action": "pick", "object_id": "bottle"}])]
r = p.plan("gently pick the bottle")
check("gentle pick bottle -> gripper_force 80 (standard floor)",
      r is not None and r[0]["gripper_force"] == 80, str(r))

# 22. pour is untouched by the extent fills
p = fresh_planner()
_Gen.queue = [envelope(manner="gentle", plan=[{"action": "pour"}])]
r = p.plan("pour gently")
check("pour gently -> omega only, no force/revolutions keys",
      r is not None and "gripper_force" not in r[0]
      and "revolutions" not in r[0] and r[0]["omega_scale"] == 0.7, str(r))

# 23/24. resolver units: fragile band + fail-firm default for unknown objects
p = fresh_planner()
check("fragile class: ceiling 80, gentle -> 60 (class floor)",
      p._resolve_gripper_force("coffee_cup", "none") == 80
      and p._resolve_gripper_force("coffee_cup", "gentle") == 60)
check("unknown object -> standard band (fail firm, never loose)",
      p._resolve_gripper_force("ghost", "none") == 100
      and p._resolve_gripper_force("ghost", "gentle") == 80)

print("=" * 60)
failed = [c for c in CHECKS if not c[1]]
print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks green")
sys.exit(1 if failed else 0)
