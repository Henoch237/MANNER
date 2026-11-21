"""test_lexicon_battery.py - seeded random battery for the intent-header
architecture, with persistent logging (paper-ready).

Two modes:
  --stub   offline self-check: a fake LLM answers with the EXPECTED
           envelope. Verifies the harness + expectation model are coherent
           (expect ~100% pass; LLM-accuracy metrics are meaningless here).
  (default) real mode: live Qwen calls. Reveals the actual strengths and
           weaknesses: per-category intent-routing accuracy, plan accuracy,
           and the LLM-side vs planner-side fault split.

Logs:
  results/battery/battery_log.jsonl                  one JSON line per turn + a run summary
  results/battery/results_lexicon_battery_<run>.json full detail (raw responses included)

Usage (from the project root):
  python -m dmp_assistant.evaluation.lexicon_battery --stub --n 100
  python -m dmp_assistant.evaluation.lexicon_battery --n 100 --seed 42   # needs DASHSCOPE_API_KEY
"""
import argparse
import json
import os
import random
import sys
import time
import types
from pathlib import Path

if __name__ == "__main__":      # direct run: make the project root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# paths is pure pathlib: importing it does not pull in dashscope, so the
# stub below can still be injected before the planner is imported.
from dmp_assistant import paths

API_KEY = os.getenv("DASHSCOPE_API_KEY")

SCRATCH_PREFS = paths.RESULTS_DIR / "battery_scratch_prefs.yaml"   # never touches preferences.yaml
LOG_JSONL = paths.RESULTS_DIR / "battery_log.jsonl"

# ------------------------------------------------------------------
# Phrasing pools - the free lexicon under test
# ------------------------------------------------------------------
GENTLE = ["softly", "gently", "with a light touch", "easy does it",
          "like you're handling eggs", "slow and careful", "nice and slow",
          "carefully please", "with great care"]
BRISK = ["quickly", "snappy", "in a hurry", "make it fast", "briskly",
         "hurry up", "be quick about it", "at full pace"]
NEGATION_GENTLE = ["not so fast", "no need to rush", "don't rush it",
                   "don't be so quick"]
ADJUST = {
    "down_strong": ["much slower", "way slower", "a lot slower"],
    "down": ["slower", "slow it down", "take it down a notch"],
    "down_slight": ["a touch slower", "a bit slower", "slightly slower"],
    "up_slight": ["a touch faster", "a bit faster", "slightly faster"],
    "up": ["faster", "speed it up", "pick up the pace"],
    "up_strong": ["much faster", "way faster", "a lot faster"],
}
STORE = ["remember this speed", "from now on", "keep it like this",
         "that's my default", "store that"]
STORE_TRAPS = ["remember to stir the coffee", "remember this recipe",
               "don't forget to pour"]
UNKNOWN_NAMES = ["gertrude", "bob", "charlie", "winifred"]

# action phrase -> (expected action list, stub plan)
ACTION_TEMPLATES = {
    "stir": ("stir the coffee", ["stir"],
             [{"action": "stir", "cup_id": "coffee_cup"}]),
    "pour": ("pour", ["pour"], [{"action": "pour"}]),
    "pick": ("pick the bottle", ["pick"],
             [{"action": "pick", "object_id": "bottle"}]),
    "hand_over": ("hand it over", ["hand_over"], [{"action": "hand_over"}]),
}


# ------------------------------------------------------------------
# Independent expectation model of R1-R8 (mirrors, never imports, the
# planner's arithmetic). Tracks drift and the scratch prefs file.
# ------------------------------------------------------------------
class Model:
    def __init__(self, people, prefs_path):
        self.registers = people["registers"]
        self.roster = people["people"]
        self.bounds = people["bounds"]
        self.adjustments = people["adjustments"]
        self.owner = people.get("current_user", "john")
        self.drift = 1.0
        self.prefs_path = prefs_path

    def clamp(self, v, lo, hi):
        return min(hi, max(lo, v))

    def stored(self, name):
        try:
            import yaml
            with open(self.prefs_path) as f:
                return (yaml.safe_load(f) or {}).get("stored", {}) \
                    .get(name, {}).get("mult", 1.0)
        except FileNotFoundError:
            return 1.0

    def apply_adjustment(self, adj):
        if adj != "none":
            f = float(self.adjustments[adj])
            lo, hi = self.bounds["drift"]
            self.drift = self.clamp(self.drift * f, lo, hi)

    def register_of(self, person, manner, named):
        """Mirrors planner._resolve_person: the person's own register is a
        slower-wins candidate only when the person was NAMED (text or
        intent recipient). Manner alone selects its register outright.
        Neither named nor mannered -> the person's register (default)."""
        cands = []
        if named and person in self.roster:
            cands.append(self.roster[person]["register"])
        m = {"gentle": "slower profile", "brisk": "faster profile"}.get(manner)
        if m:
            cands.append(m)
        if cands:
            return min(cands, key=lambda r: self.registers[r]["speed"])
        if person in self.roster:
            return self.roster[person]["register"]
        return None

    def eff(self, person):
        lo, hi = self.bounds["drift"]
        return self.clamp(self.stored(person) * self.drift, lo, hi)

    def step_values(self, action, person, manner, named):
        """Expected planner-filled values for one step, or None."""
        reg = self.register_of(person, manner, named)
        if reg is None:
            return None
        vals = self.registers[reg]
        eff = self.eff(person)
        slo, shi = self.bounds["speed"]
        olo, ohi = self.bounds["omega_scale"]

        def sc(v):
            return int(self.clamp(round(v * eff), slo, shi))

        out = {}
        if action in ("stir", "pour"):
            out["omega_scale"] = round(self.clamp(
                vals.get("omega_scale", 1.0) * eff, olo, ohi), 3)
        if action != "pour" and action in (
                "stir", "hand_over", "go_to", "pick", "place",
                "pick_and_place"):
            out["speed"] = sc(vals.get("speed", 30))
            out["accel"] = sc(vals.get("accel", 100))
        if action == "hand_over" and "confirm_dwell_s" in vals:
            d = vals["confirm_dwell_s"]
            if abs(eff - 1.0) > 1e-6:
                half = 1.0 + (eff - 1.0) * 0.5
                d = max(self.bounds["dwell_floor_s"], d / half)
            out["confirm_dwell_s"] = round(d, 2)
        return out

    def gate_would_open(self):
        old = self.stored(self.owner)
        lo, hi = self.bounds["drift"]
        would = self.clamp(old * self.drift, lo, hi)
        return abs(would - old) >= 1e-9

    def confirm_store(self, stored_before):
        """Expected stored value after a gate 'yes'. The PLANNER is the
        only writer - the model only predicts. (An earlier version wrote
        here too, which double-applied the drift to the scratch file and
        corrupted the stored-multiplier trajectory of the whole run.)"""
        lo, hi = self.bounds["drift"]
        return round(self.clamp(stored_before * self.drift, lo, hi), 3)


# ------------------------------------------------------------------
# Scenario generation. A scenario is a list of turns; each turn:
#   request, category, expect_intent (dict or "skip"),
#   expect_actions (list, or None = expect refusal/None return),
#   values_for = (person, manner) for numeric checks,
#   gate = None | "yes" | "no" | "expire"
# ------------------------------------------------------------------
def _turn(request, category, intent, actions, person, manner, gate=None,
          named=False):
    return {"request": request, "category": category,
            "expect_intent": intent, "expect_actions": actions,
            "values_for": (person, manner, named), "gate": gate}


def build_scenarios(rng, model, n):
    scenarios = []
    names = list(model.roster)
    cats = (["manner_gentle"] * 14 + ["manner_brisk"] * 8
            + ["manner_negation"] * 5 + ["adjustment_only"] * 14
            + ["store"] * 14 + ["store_trap"] * 5
            + ["recipient"] * 12 + ["recipient_unknown"] * 5
            + ["baseline"] * 13 + ["compound"] * 10)

    for i in range(n):
        cat = cats[i % len(cats)]
        owner = model.owner

        if cat == "manner_gentle":
            act = rng.choice(list(ACTION_TEMPLATES))
            phrase, actions, stub = ACTION_TEMPLATES[act]
            p = rng.choice(GENTLE)
            scenarios.append([_turn(f"{phrase}, {p}", cat,
                {"manner": "gentle", "adjustment": "none", "store": False,
                 "recipient": "none"}, actions, owner, "gentle")])

        elif cat == "manner_brisk":
            act = rng.choice(list(ACTION_TEMPLATES))
            phrase, actions, stub = ACTION_TEMPLATES[act]
            p = rng.choice(BRISK)
            scenarios.append([_turn(f"{phrase}, {p}", cat,
                {"manner": "brisk", "adjustment": "none", "store": False,
                 "recipient": "none"}, actions, owner, "brisk")])

        elif cat == "manner_negation":
            p = rng.choice(NEGATION_GENTLE)
            scenarios.append([_turn(f"stir the coffee, {p}", cat,
                {"manner": "gentle", "adjustment": "none", "store": False,
                 "recipient": "none"}, ["stir"], owner, "gentle")])

        elif cat == "adjustment_only":
            label = rng.choice(list(ADJUST))
            p = rng.choice(ADJUST[label])
            scenarios.append([_turn(p, cat,
                {"manner": "none", "adjustment": label, "store": False,
                 "recipient": "none"}, None, owner, "none")])

        elif cat == "store":
            # turn 1: adjustment (guarantees drift != 1.0 so the gate opens)
            label = rng.choice(list(ADJUST))
            p1 = rng.choice(ADJUST[label])
            # turn 2: store wish + action
            act = rng.choice(list(ACTION_TEMPLATES))
            phrase, actions, stub = ACTION_TEMPLATES[act]
            p2 = rng.choice(STORE)
            # turn 3: gate answer
            answer = rng.choices(["yes", "no", "expire"], weights=[5, 2, 3])[0]
            t3_req = {"yes": "yes", "no": "no",
                      "expire": "stir the coffee"}[answer]
            scenarios.append([
                _turn(p1, cat + ":adj",
                      {"manner": "none", "adjustment": label, "store": False,
                       "recipient": "none"}, None, owner, "none"),
                _turn(f"{phrase}, {p2}", cat,
                      {"manner": "none", "adjustment": "none", "store": True,
                       "recipient": "none"}, actions, owner, "none"),
                _turn(t3_req, cat + ":gate", "skip",
                      ["stir"] if answer == "expire" else None,
                      owner, "none", gate=answer)])

        elif cat == "store_trap":
            p = rng.choice(STORE_TRAPS)
            act = "stir" if "stir" in p else "pour"
            actions = [act]
            scenarios.append([_turn(p, cat,
                {"manner": "none", "adjustment": "none", "store": False,
                 "recipient": "none"}, actions, owner, "none")])

        elif cat == "recipient":
            n_ = rng.choice(names)
            style = rng.choice(["for", "possessive", "give"])
            if style == "for":
                req = f"stir the coffee for {n_}"
                actions = ["stir"]
            elif style == "possessive":
                req = f"{n_}'s coffee needs a stir"
                actions = ["stir"]
            else:
                req = f"give it to {n_}"
                actions = ["hand_over"]
            scenarios.append([_turn(req, cat,
                {"manner": "none", "adjustment": "none", "store": False,
                 "recipient": n_}, actions, n_, "none", named=True)])

        elif cat == "recipient_unknown":
            x = rng.choice(UNKNOWN_NAMES)
            scenarios.append([_turn(f"stir the coffee for {x}", cat,
                "skip", None, owner, "none")])

        elif cat == "baseline":
            req, actions = rng.choice([
                ("stir the coffee", ["stir"]),
                ("pour", ["pour"]),
                ("pick the bottle", ["pick"]),
                ("pick the bottle and put it in the bin",
                 ["pick_and_place"]),
                ("hand it over", ["hand_over"]),
                ("go to the bin", ["go_to"]),
            ])
            scenarios.append([_turn(req, cat,
                {"manner": "none", "adjustment": "none", "store": False,
                 "recipient": "none"}, actions, owner, "none")])

        elif cat == "compound":
            n_ = rng.choice(names)
            req, manner, reg_person, actions = rng.choice([
                ("stir the coffee gently and remember that", "gentle",
                 owner, ["stir"]),
                (f"pour for {n_}, softly", "gentle", n_, ["pour"]),
                ("stir the coffee with a light touch", "gentle", owner,
                 ["stir"]),
            ])
            store_flag = "remember" in req
            named = f"for {n_}" in req
            scenarios.append([_turn(req, cat,
                {"manner": manner, "adjustment": "none", "store": store_flag,
                 "recipient": n_ if named else "none"},
                actions, reg_person, manner, named=named)])

    rng.shuffle(scenarios)
    return scenarios


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------
STUB_PLANS = {
    "stir": [{"action": "stir", "cup_id": "coffee_cup"}],
    "pour": [{"action": "pour"}],
    "pick": [{"action": "pick", "object_id": "bottle"}],
    "pick_and_place": [{"action": "pick_and_place", "object_id": "bottle",
                        "destination_id": "bin"}],
    "hand_over": [{"action": "hand_over"}],
    "go_to": [{"action": "go_to", "target_pose": "bin.approach_pose"}],
}


def _stub_envelope(turn):
    intent = turn["expect_intent"]
    if intent == "skip":
        intent = {"manner": "none", "adjustment": "none", "store": False,
                  "recipient": "none"}
    plan = []
    if turn["expect_actions"]:
        for a in turn["expect_actions"]:
            plan.extend(STUB_PLANS[a])
    return json.dumps({"intent": intent, "plan": plan})


def install_stub():
    """Fake dashscope: answers with the EXPECTED envelope of the turn
    currently being run. Raises if the planner makes an unexpected call."""
    fake = types.ModuleType("dashscope")
    fake.api_key = None

    class _Resp:
        def __init__(self, content):
            self.status_code = 200
            self.output = types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=content))])

    class _Gen:
        current = None

        @staticmethod
        def call(**kw):
            if _Gen.current is None:
                raise RuntimeError("unexpected API call (stub queue empty)")
            return _Resp(_Gen.current)

    fake.Generation = _Gen
    sys.modules["dashscope"] = fake
    return _Gen


def check_numbers(plan, model, person, manner, named):
    """Given the intent acted on, were the planner's numbers right?
    Uses the planner's ACTUAL drift and the live scratch file, so this
    isolates planner-side arithmetic from LLM routing."""
    if not plan:
        return True, ""
    for step in plan:
        exp = model.step_values(step.get("action"), person, manner,
                                named)
        if not exp:
            continue
        for k, v in exp.items():
            got = step.get(k)
            if got is None or abs(float(got) - float(v)) > 1e-6:
                return False, f"{step.get('action')}.{k}: got {got}, want {v}"
    return True, ""


def run_battery(n, seed, stub_mode):
    rng = random.Random(seed)
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_s{seed}"

    if not stub_mode and not API_KEY:
        sys.exit("DASHSCOPE_API_KEY is not set (needed for live mode; use --stub offline).")
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCRATCH_PREFS, "w") as f:
        f.write("stored: {}\n")

    stub = install_stub() if stub_mode else None
    from dmp_assistant.planning.planner import Planner   # after stub injection
    p = Planner(api_key=API_KEY or "stub", preferences_path=SCRATCH_PREFS)
    model = Model(p.people, SCRATCH_PREFS)

    scenarios = build_scenarios(rng, model, n)
    results = []

    def log(entry):
        results.append(entry)
        with open(LOG_JSONL, "a") as f:
            f.write(json.dumps({"run": run_id, **entry}) + "\n")

    for idx, scenario in enumerate(scenarios):
        # housekeeping: never carry a pending gate across scenarios
        if p.pending_store is not None:
            p.plan("no")

        for turn in scenario:
            cat = turn["category"]
            req = turn["request"]
            ei = turn["expect_intent"]

            # guarantee the store gate can actually open
            if cat == "store" and not model.gate_would_open():
                for _ in range(4):
                    prep = "faster" if model.drift <= 1.0 else "slower"
                    label = "up" if model.drift <= 1.0 else "down"
                    if stub_mode:
                        stub.current = json.dumps({"intent": {
                            "manner": "none", "adjustment": label,
                            "store": False, "recipient": "none"},
                            "plan": []})
                    p.plan(prep)
                    model.apply_adjustment(label)
                    if model.gate_would_open():
                        break

            drift_before = model.drift
            stored_before = model.stored(model.owner)
            if stub_mode:
                stub.current = _stub_envelope(turn)

            t0 = time.time()
            try:
                r = p.plan(req)
                err = None
            except Exception as e:          # API failure, anything
                r, err = None, str(e)
            latency = time.time() - t0

            # ---- checks ----
            intent_ok = True
            if ei != "skip" and p.last_intent is not None:
                for k, v in ei.items():
                    if p.last_intent.get(k) != v:
                        intent_ok = False

            if turn["expect_actions"] is None:
                plan_ok = r is None
                actions_got = None
            else:
                actions_got = [s.get("action") for s in r] if r else None
                plan_ok = actions_got == turn["expect_actions"]

            person, manner, named = turn["values_for"]
            num_ok, num_detail = True, ""
            if r and turn["expect_actions"]:
                num_ok, num_detail = check_numbers(r, model, person,
                                                   manner, named)

            drift_ok = True
            if ei != "skip" and isinstance(ei, dict):
                want = drift_before
                if ei.get("adjustment", "none") != "none" and intent_ok:
                    lo, hi = model.bounds["drift"]
                    want = model.clamp(drift_before * float(
                        model.adjustments[ei["adjustment"]]), lo, hi)
                drift_ok = abs(p.session_drift - want) < 1e-9

            stored_now = model.stored(model.owner)
            stored_ok = True
            if turn["gate"] == "yes":
                # exact-value check: the file must hold the predicted
                # product - stronger than "did it change"
                stored_ok = abs(stored_now
                                - model.confirm_store(stored_before)) < 1e-9
            elif turn["gate"] in ("no", "expire"):
                stored_ok = abs(stored_now - stored_before) < 1e-9

            # ---- fault attribution ----
            if err:
                fault = "error"
            elif not intent_ok or not plan_ok:
                fault = "llm"          # misrouted intent or misplanned
            elif not num_ok or not drift_ok or not stored_ok:
                fault = "planner"      # right intent, wrong arithmetic
            else:
                fault = None

            log({"idx": idx, "category": cat, "request": req,
                 "expected_intent": ei, "actual_intent": p.last_intent,
                 "expected_actions": turn["expect_actions"],
                 "actual_actions": actions_got,
                 "returned_plan": r, "gate": turn["gate"],
                 "intent_ok": intent_ok, "plan_ok": plan_ok,
                 "numbers_ok": num_ok, "numbers_detail": num_detail,
                 "drift_ok": drift_ok, "stored_ok": stored_ok,
                 "fault": fault, "latency_s": round(latency, 3),
                 "raw": p.last_raw, "error": err})

            # ---- advance the model to reality ----
            if ei != "skip" and isinstance(ei, dict) and intent_ok:
                model.apply_adjustment(ei["adjustment"])
            else:
                model.drift = p.session_drift   # resync after LLM faults
            model.drift = p.session_drift

    # ---- summary ----
    cats = {}
    for e in results:
        c = cats.setdefault(e["category"], {"n": 0, "intent_ok": 0,
            "plan_ok": 0, "numbers_ok": 0, "pass": 0,
            "llm_fault": 0, "planner_fault": 0, "error": 0})
        c["n"] += 1
        c["intent_ok"] += e["intent_ok"]
        c["plan_ok"] += e["plan_ok"]
        c["numbers_ok"] += e["numbers_ok"]
        c["pass"] += e["fault"] is None
        if e["fault"]:
            c[e["fault"] + "_fault" if e["fault"] != "error" else "error"] += 1

    lat = [e["latency_s"] for e in results]
    total = len(results)
    passed = sum(c["pass"] for c in cats.values())
    summary = {"run": run_id, "seed": seed, "n_requested": n,
               "turns": total, "passed": passed,
               "pass_rate": round(passed / total, 4) if total else 0,
               "latency_mean_s": round(sum(lat) / len(lat), 3),
               "latency_max_s": max(lat),
               "by_category": cats}
    with open(LOG_JSONL, "a") as f:
        f.write(json.dumps({"run": run_id, "summary": summary}) + "\n")
    out = paths.RESULTS_DIR / f"results_lexicon_battery_{run_id}.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "turns": results}, f, indent=2)

    print("\n" + "=" * 72)
    print(f"RUN {run_id}  ({'STUB self-check' if stub_mode else 'LIVE Qwen'})")
    print(f"turns: {total}   pass: {passed} ({summary['pass_rate']:.1%})   "
          f"latency mean {summary['latency_mean_s']}s")
    print(f"{'category':<22}{'n':>4}{'intent':>8}{'plan':>6}{'nums':>6}"
          f"{'llm':>5}{'planner':>8}")
    for c, s in sorted(cats.items()):
        print(f"{c:<22}{s['n']:>4}{s['intent_ok']:>8}{s['plan_ok']:>6}"
              f"{s['numbers_ok']:>6}{s['llm_fault']:>5}{s['planner_fault']:>8}")
    print(f"\nfull detail: {out}   history: {LOG_JSONL}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stub", action="store_true",
                    help="offline self-check with a scripted fake LLM")
    a = ap.parse_args()
    run_battery(a.n, a.seed, a.stub)
