import json
import os
import re
import yaml
import dashscope

from dmp_assistant import paths
from dmp_assistant.planning.schema_builder import (
    build_schema, detected_object_ids, resolvable_pose_refs, INTENT_MANNERS,
    INTENT_ADJUSTMENTS)


class Planner:
    # ------------------------------------------------------------------
    # Lexicon-freedom architecture: the LLM classifies manner/adjustment/
    # store into the CLOSED intent labels (schema_builder.INTENT_*);
    # Python maps labels to the declared values in people.yaml. No word
    # lists are used for classification anywhere in this file.
    # ------------------------------------------------------------------
    MOTION_ACTIONS = {"go_to", "pick", "place", "pick_and_place"}

    # Intent label -> register. The ONLY manner mapping left; it is
    # word-free (labels come from the LLM's intent header).
    _MANNER_TO_REGISTER = {"gentle": "slower profile", "brisk": "faster profile"}

    # --- Extent parameters: integer, label -> absolute value, clamped. -----
    # Revolutions (stir): nothing said -> default; "well" -> 4.
    _REVOLUTIONS_MAP = {"well": 4}
    _REVOLUTIONS_DEFAULT = 2
    _REVOLUTIONS_BOUNDS = (1, 6)

    # Gripper force (percent of rated closing force). The OBJECT's handling
    # class (scene.yaml) sets the safe band: _FORCE_MAP is the ceiling (the
    # default grip) and _FORCE_FLOOR the lowest hold the class tolerates.
    # The INSTRUCTION can only soften within that band: manner "gentle"
    # steps force down once, never below the class floor, never above the
    # class ceiling. "brisk" never raises force - squeezing harder is not
    # what brisk means.
    _FORCE_MAP = {"robust": 100, "standard": 100, "fragile": 80,
                  "deformable": 80}
    _FORCE_FLOOR = {"robust": 80, "standard": 80, "fragile": 60,
                    "deformable": 60}
    _FORCE_GENTLE_STEP = 20
    _FORCE_DEFAULT = 100
    _FORCE_BOUNDS = (20, 100)

    _FOR_STOPWORDS = {"me", "my", "the", "a", "it"}
    # Tokens that may follow "for" without being a person reference. NOT a
    # classifier - manner itself comes from the intent header; this only
    # prevents false unknown-person alarms ("pour for the elderly").
    _FOR_X_NONPERSONS = {"elderly", "asleep", "nurse", "doctor"}

    # Store-confirmation gate (R4): closed answer sets, one intent per
    # confirmation turn. Deliberately dumb - this is the write path.
    _YES_WORDS = {"yes", "y", "yeah", "yep", "confirm"}
    _NO_WORDS = {"no", "n", "nope", "cancel"}

    def __init__(self, api_key=None, registry_path=paths.REGISTRY_YAML,
                 scene_path=paths.SCENE_YAML, people_path=paths.PEOPLE_YAML,
                 preferences_path=paths.PREFERENCES_YAML):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.registry = self._load_yaml(registry_path)
        self.scene = self._load_yaml(scene_path)
        self.people = self._load_yaml(people_path) if os.path.exists(people_path) else {}
        self.preferences_path = preferences_path
        # Session state (R1/R3): in-memory only, gone at process exit.
        self.session_drift = 1.0
        self.current_user_override = None
        # Pending store confirmation (R4): None or {"owner", "drift",
        # "would_be"}. Volatile; lives exactly one user turn.
        self.pending_store = None
        self.schema = build_schema(registry_path, scene_path,
                                   people_path)  # build once at init
        # resolvable_pose_refs (schema_builder) includes computed approach
        # poses (derived_from + offset_z_mm), so no extra work here.
        self.pose_refs = set(resolvable_pose_refs(scene_path))
        # Session memory (continuous conversation): message history for the
        # LLM. Volatile - cleared by reset_session() or process exit.
        self.history = []          # [{role, content}, ...] after the first turn
        self.first_prompt_sent = False

    def _load_yaml(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_prompt(self, user_request, current_state=None):
        actions = []
        for p in self.registry.get("primitives", []):
            if p.get("type") == "composite":
                params = []
                for param in p.get("parameters", []):
                    params.append(f"{param['name']} ({param.get('type', 'any')})")
                actions.append({
                    "name": p["id"],
                    "description": p["description"],
                    "parameters": params
                })

        objects = list(self.scene.get("poses", {}).keys())
        detected = {k: self.scene["poses"][k].get("description", "")
                    for k in detected_object_ids(self.scene)}
        pose_refs = sorted(self.pose_refs)
        people = self.people.get("people", {})
        registers = self.people.get("registers", {})
        current_user = self.get_current_user() or "unknown"

        state_str = ""
        if current_state:
            state_str = f"\nCurrent robot state:\n{json.dumps(current_state, indent=2)}"

        # Examples kept outside the f-string to avoid brace conflicts.
        # Every answer has the same envelope: an intent header (closed
        # labels) plus the plan array.
        intent_none = '"intent": {"manner": "none", "adjustment": "none", "store": false, "recipient": "none"}'
        example_a = '{\n  ' + intent_none + ',\n  "plan": [\n    {"action": "pick_and_place", "object_id": "bottle", "destination_id": "bin"}\n  ]\n}'
        example_b = '{\n  "intent": {"manner": "none", "adjustment": "none", "store": false, "recipient": "sarah"},\n  "plan": [\n    {"action": "pick", "object_id": "bottle", "recipient": "sarah"},\n    {"action": "go_to", "target_pose": "initial_pose.value", "recipient": "sarah"}\n  ]\n}'
        example_c = '{\n  ' + intent_none + ',\n  "plan": [\n    {"action": "stir", "cup_id": "coffee_cup", "omega_scale": 0.8}\n  ]\n}'
        example_d = '{\n  ' + intent_none + ',\n  "plan": [\n    {"action": "pour"}\n  ]\n}'
        example_e = '{\n  ' + intent_none + ',\n  "plan": [\n    {"action": "go_to", "target_pose": "red_block.approach_pose"},\n    {"action": "gripper_open"},\n    {"action": "go_to", "target_pose": "red_block.grasp_pose"},\n    {"action": "gripper_close"},\n    {"action": "go_to", "target_pose": "red_block.approach_pose"}\n  ]\n}'
        example_f = '{\n  ' + intent_none + ',\n  "plan": [\n    {"action": "go_to", "target_pose": "bin.approach_pose"},\n    {"action": "gripper_open"},\n    {"action": "go_to", "target_pose": "bin.place_pose"},\n    {"action": "gripper_close"},\n    {"action": "go_to", "target_pose": "bin.approach_pose"}\n  ]\n}'
        example_g = '{\n  "intent": {"manner": "none", "adjustment": "down", "store": false, "recipient": "none"},\n  "plan": []\n}'
        example_h = '{\n  "intent": {"manner": "gentle", "adjustment": "none", "store": false, "recipient": "none"},\n  "plan": [\n    {"action": "stir", "cup_id": "coffee_cup"}\n  ]\n}'
        example_i = '{\n  "intent": {"manner": "well", "adjustment": "none", "store": false, "recipient": "none"},\n  "plan": [\n    {"action": "stir", "cup_id": "coffee_cup", "revolutions": 4}\n  ]\n}'

        prompt = f"""You are a robot task planner for a FAIRINO F5 manipulator.
Convert the user's natural language request into a structured JSON object:
an "intent" header (how the request should FEEL and who it is FOR, in
closed labels) plus a "plan" array of action steps.

AVAILABLE ACTIONS:
{json.dumps(actions, indent=2)}

AVAILABLE OBJECTS:
{json.dumps(objects, indent=2)}

CAMERA-DETECTED OBJECTS (no fixed pose - the camera finds them when they are picked):
{json.dumps(detected, indent=2)}

RESOLVABLE POSES (the ONLY valid target_pose values):
{json.dumps(pose_refs, indent=2)}

AVAILABLE PEOPLE (name -> register):
{json.dumps(people, indent=2)}

SPEED REGISTERS (register -> motion parameters; the planner applies these - you never emit them):
{json.dumps(registers, indent=2)}

CURRENT USER (default recipient when the request names no one):
{current_user!r}{state_str}

RULES:
1. Output ONLY a JSON object with two fields: "intent" and "plan". No markdown, no explanation, no code blocks.
2. "intent.manner": classify the manner of the request from ANY wording. "gentle" for any phrasing of slow/soft/careful/delicate ("softly", "with a light touch", "easy does it", "like you're handling eggs"); "brisk" for any phrasing of fast/quick/snappy ("quickly", "snappy", "in a hurry"); "well" for any phrasing of thoroughness or quality about HOW WELL the action is done rather than how fast or soft ("stir well", "mix thoroughly", "do it properly", "give it a good stir"); "none" when the request says nothing about manner. Negation counts: "not so fast" -> "gentle"; "no need to rush" -> "gentle".
3. "intent.adjustment": classify any speed-change wish by direction and degree: "down_strong", "down", "down_slight", "up_slight", "up", "up_strong" - or "none" when there is no speed-change wish. "a touch slower" -> "down_slight"; "way too slow" -> "up_strong". The planner applies adjustments as session state - NEVER translate them into speed/omega/parameter values, and never model them as plan steps.
4. "intent.store": true ONLY when the user asks to make the current speed their lasting default ("from now on", "keep it like this", "remember this speed", "that's my default"). "Remember to do X" is a task, not a store -> false. A store or adjustment wish NEVER cancels the action verbs in the same sentence - plan them normally AND set the intent.
5. "intent.recipient": the person the action is FOR - one of AVAILABLE PEOPLE, or "none". Any phrasing counts ("for sarah", "sarah's coffee", "hand it to sarah"). When no one is named, the action is for the current user -> "none". Never invent a name outside AVAILABLE PEOPLE.
6. "plan" is an array of action steps. Each element must have an "action" field matching one of the available action names.
7. If the request is ONLY an adjustment and/or a store wish with no action verb, return "plan": [] with the intent fields set - an empty plan WITH intent is a valid answer. An empty plan with an all-"none" intent means refusal.
8. Use only object IDs listed in AVAILABLE OBJECTS.
9. If the user says "coffee", "cup", or "stir": use cup_id "coffee_cup".
10. "pick_and_place" is ONLY for when the user explicitly says BOTH "pick X" AND "place in Y" / "put in Y" in the same sentence. If no destination is given, do NOT invent one.
11. If the user only says "pick" or "grab" an object without saying where to put it, use action "pick" with object_id.
12. If the user says "return to", "go to", or "move to" a location, use action "go_to" with target_pose. This is always a SEPARATE step, never bundled inside pick_and_place.
13. "revolutions" (stir only) is the stir thoroughness knob, range 1-6. The planner clamps to the declared bounds. EMIT IT ONLY WHEN THE USER EXPRESSES A COUNT OR A THOROUGHNESS DEGREE - an emitted value IS the user's wish and is taken as stated (clamped). If the user states a NUMBER of times ("stir 5 times"), emit exactly that number. You MAY also emit revolutions (1 to 6) from ANY phrasing of how thorough or energetic the mixing should be ("stir well" -> 4, "stir vigorously" -> 5, "mix it properly" -> 4, "stir briefly" -> 1, "give it a quick swirl" -> 1). When the request says nothing about thoroughness, OMIT the field entirely - never invent a count.
14. If the user says "pour", use action "pour" (optionally with "recipient" per rule 22 - the planner fills the tempo). Pour happens in place from the current pose - if the user mentions a specific position, emit "go_to" first, then "pour".
15. "place pose", "place position", "placing position", "drop-off", or "the bin" all mean target_pose "bin.place_pose". target_pose MUST be one of the RESOLVABLE POSES listed above - never invent or shorten pose paths (e.g. "bin" alone is NOT valid).
16. "go to" an OBJECT by name only (block, cup, ...) means its approach_pose (above the object) - never a contact pose (grasp_pose, center_pose). But if the user names a SPECIFIC pose ("grasp pose", "cup center", a full pose path), use exactly that pose - an explicit pose request always overrides this default.
17. "open gripper" -> action "gripper_open"; "close gripper" -> action "gripper_close". These are standalone actions, never bundled into go_to.
18. If the user says "give", "hand", "pass", or "hand over": use action "hand_over". It hands over whatever the robot currently holds - do NOT add an object_id, destination, or target_pose to it. Do NOT set confirm_dwell_s - the planner fills it (and the speeds) from the recipient's register (rule 22).
19. VERB PRESERVATION: every action verb in the request ("pick", "grab", "place", "pour", "stir", "hand", "give", "pass") must appear in the plan as its corresponding action. Never answer a "pick ..." request with navigation only - a bare "go_to" is NOT a pick. If the verb's target cannot be resolved to an object or pose, return an error asking what to pick - do not downgrade the request to movement.
20. "pick from <pose>" where <pose> is a NAMED POSE (not an object): do NOT use action "pick" (pick only accepts objects). First resolve the description to its full pose path (e.g. "the pose where the bin was placed" -> "bin.place_pose"). Then compose the pick from small primitives in this order - approach, open, descend, close, retract:
    a. If <pose> belongs to an object X (form "X.<anything>_pose") and "X.approach_pose" exists: go_to "X.approach_pose", then gripper_open, then go_to <pose>, then gripper_close, then go_to "X.approach_pose".
    b. Otherwise (a standalone pose): gripper_open first, then go_to <pose>, then gripper_close.
    The named pose MUST be one of the RESOLVABLE POSES - never invent or shorten pose paths.
21. If no composite action fits the request, you may compose a SHORT sequence from ONLY these small actions: "go_to", "gripper_open", "gripper_close". Every go_to target MUST be one of the RESOLVABLE POSES. Never output "linear_move" or "approach_z" - those are executor-internal. When a composite action fits, prefer it over composing.
22. RECIPIENT, NOT NUMBERS: for go_to, pick, place, pick_and_place, pour and hand_over, you MAY set "recipient" (a name from AVAILABLE PEOPLE or a register) - the intent header's "recipient" (rule 5) is the primary signal, so this per-step field is optional. NEVER emit "speed", "accel" or "omega_scale" - the planner fills them from the recipient's register and the intent header's "manner" (gentle -> the slower profile, brisk -> the faster profile; a conflict resolves to the SLOWER profile).
    - "for <name>" -> that name. "for me" / "my" / no recipient -> "none" in the intent header (the current user's register is used).
    - A person name not in AVAILABLE PEOPLE -> return an error asking who that is; never guess.
23. If the request names a person, that person is the recipient of the action (their register applies to that action only); the session owner does not change.
24. DMP skills use "omega_scale" as their tempo knob (planner-filled from the register: slower profile/gentle -> 0.7, faster profile/brisk -> 1.0): stir for the circles, pour for the whole tilt cycle. speed/accel on a stir step apply ONLY to the transit move to the cup - never to the motion itself. hand_over uses register speed/accel (presentation vel = speed x 0.6 inside the skill) and confirm_dwell_s, planner-filled (slower profile -> 4.0, faster profile -> 3.0).
25. ONLY CAMERA-DETECTED OBJECTS can be picked ("pick" / "pick_and_place" object_id) - the executor locates them with the camera. Map the user's words onto them ("the bottle" -> "bottle", "the hand gel" -> "sanitizer", "the cup"/"the container" when picking -> "hollow"). If the user asks to pick anything else (e.g. "the red block"), refuse (empty plan, all-"none" intent, rule 7) - never pick a non-detected object. Camera objects have NO poses: never use them in go_to, never compose their pick from go_to/gripper steps (rule 20 does not apply to them).

OUTPUT FORMAT EXAMPLES:

Example A — explicit pick AND place:
{example_a}

Example B — pick only, then go somewhere else (for sarah):
{example_b}

Example C — stir only:
{example_c}

Example D — pour in place (no parameters):
{example_d}

Example E — "pick from" a named pose (composed from small primitives):
{example_e}

Example F — "pick from" described indirectly ("the pose where the bin was placed" -> "bin.place_pose"):
{example_f}

Example G — adjustment only, no action ("slower"):
{example_g}

Example H — free-phrased manner ("stir the coffee softly"):
{example_h}

Example I — quality manner ("stir well" -> manner "well", revolutions emitted for the requested thoroughness):
{example_i}

User request: "{user_request}"
"""
        return prompt

    def reset_session(self):
        """New session: forget conversation history, session drift and any
        pending store confirmation. Believed robot state is NOT touched -
        that belongs to the executor."""
        self.history = []
        self.first_prompt_sent = False
        self.session_drift = 1.0
        self.pending_store = None

    def get_current_user(self):
        return self.current_user_override or self.people.get("current_user")

    def set_current_user(self, name):
        """Switch the session owner (launcher command: user <name>)."""
        if name not in self.people.get("people", {}):
            print(f"[SESSION ERROR] unknown person '{name}'")
            return False
        if self.pending_store is not None:
            self.pending_store = None
            print("[SESSION] pending store discarded (owner switch).")
        self.current_user_override = name
        self.session_drift = 1.0   # fresh drift for the new owner
        print(f"[SESSION] current_user -> {name} (drift reset to 1.0)")
        return True

    def _build_messages(self, user_request, current_state=None):
        """First turn sends the full prompt; later turns ride the history."""
        if not self.first_prompt_sent:
            return [{"role": "user", "content": self._build_prompt(user_request, current_state)}]
        return self.history + [{"role": "user", "content": user_request}]

    def _remember_exchange(self, user_request, assistant_content):
        """Append this turn to history, keeping the first prompt + last N."""
        user_msg = (self._build_prompt(user_request) if not self.first_prompt_sent
                    else user_request)
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_content})
        self.first_prompt_sent = True
        self.history = self.history[:1] + self.history[-16:]  # cap ~8 exchanges

    # ------------------------------------------------------------------
    # Session state helpers. Drift/store INTENT now arrives via the LLM's
    # intent header; the arithmetic and the writes stay deterministic here.
    # ------------------------------------------------------------------

    @staticmethod
    def _words(text):
        """Punctuation-immune word set: 'give it to sarah, faster' contains
        'sarah' (a trailing comma must not hide a name)."""
        return set(re.findall(r"[a-z]+", text.lower()))

    def _load_preferences(self):
        if os.path.exists(self.preferences_path):
            return self._load_yaml(self.preferences_path)
        return {"stored": {}}

    def _other_person_named(self, request):
        owner = self.get_current_user()
        words = self._words(request)
        for name in self.people.get("people", {}):
            if name != owner and name in words:
                return name
        return None

    def _open_store_gate(self, owner, user_request):
        """R4/R5: the LLM recognized a persistence wish; Python decides if
        it may be written. R5 (another person named) -> hard fail. Drift at
        default -> nothing to store. Otherwise open a one-turn
        echo-confirmation keyed to the owner at request time."""
        other = self._other_person_named(user_request)
        if other is not None:
            print(f"[PLANNER ERROR] R5: '{other}' is not the session owner "
                  f"('{owner}'). Cannot store to another person's "
                  f"preferences - nothing was written.")
            return
        prefs = self._load_preferences()
        old = prefs.get("stored", {}).get(owner, {}).get("mult", 1.0)
        lo, hi = self.people.get("bounds", {}).get("drift", [0.3, 1.5])
        would_be = min(hi, max(lo, old * self.session_drift))
        if abs(would_be - old) < 1e-9:
            print(f"[SESSION] nothing to store for '{owner}': drift is at "
                  f"default (stored stays {old:.2f})")
            return
        self.pending_store = {"owner": owner, "drift": self.session_drift,
                              "would_be": would_be}
        print(f"[SESSION] Remember this speed as your default? "
              f"(stored multiplier for '{owner}' would become "
              f"{would_be:.2f}) - yes/no")

    def _write_stored(self, owner, drift=None):
        """R4 (accumulate, never replace): stored[owner] *= drift.
        A drift of 1.0 is a no-op - storing at default speed can never
        erase a preference. Result clamped to the declared drift bounds.
        Called ONLY by the confirmation gate, with the drift captured at
        request time."""
        drift = self.session_drift if drift is None else drift
        prefs = self._load_preferences()
        stored = prefs.setdefault("stored", {})
        old = stored.get(owner, {}).get("mult", 1.0)
        lo, hi = self.people.get("bounds", {}).get("drift", [0.3, 1.5])
        new = min(hi, max(lo, old * drift))
        if abs(new - old) < 1e-9:
            print(f"[SESSION] nothing to store for '{owner}': drift is at "
                  f"default (stored stays {old:.2f})")
            return
        stored.setdefault(owner, {})["mult"] = round(new, 3)
        with open(self.preferences_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(prefs, f, allow_unicode=True, sort_keys=False)
        print(f"[SESSION] stored '{owner}': {old:.2f} x drift "
              f"{drift:.2f} -> {new:.2f}  ({self.preferences_path})")

    def reset_preferences(self, owner=None):
        """Explicit deletion of a stored preference - the ONLY way a
        preference is ever removed (never via store)."""
        owner = owner or self.get_current_user()
        prefs = self._load_preferences()
        stored = prefs.get("stored", {})
        if owner in stored:
            del stored[owner]
            with open(self.preferences_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(prefs, f, allow_unicode=True, sort_keys=False)
            print(f"[SESSION] reset stored prefs for '{owner}'")
        else:
            print(f"[SESSION] '{owner}' has no stored prefs")

    def plan(self, user_request, current_state=None):
        if not self.api_key:
            raise ValueError("No DashScope API key. Set DASHSCOPE_API_KEY env var or pass api_key=...")

        owner = self.get_current_user()
        # Observability for test harnesses: the last parsed intent header
        # and raw LLM response (None when no API call was made).
        self.last_intent = None
        self.last_raw = None

        # ---- Pending store confirmation (R4): one turn, yes/no only,
        # resolved deterministically BEFORE any API call. Anything else
        # discards the pending store and is processed normally. ----
        if self.pending_store is not None:
            pend = self.pending_store
            self.pending_store = None      # always consumed after one turn
            cmd = user_request.lower().strip().strip(".!,")
            if cmd in self._YES_WORDS:
                self._write_stored(pend["owner"], pend["drift"])
                return None
            if cmd in self._NO_WORDS:
                print("[SESSION] store declined - nothing was written.")
                return None
            print("[SESSION] store confirmation expired (no yes/no) - "
                  "nothing was written; processing your request normally.")
            # fall through and plan this request

        # Rule 21 backstop, before spending an API call: unknown "for X" ->
        # ask, never guess.
        unknown = self._unknown_person_in_request(user_request)
        if unknown is not None:
            print(f"[PLANNER ERROR] Unknown person '{unknown}' - not in people.yaml. "
                  f"Asking instead of guessing (rule 21).")
            return None

        dashscope.api_key = self.api_key
        messages = self._build_messages(user_request, current_state)

        # ------------------------------------------------------------------
        # Constrained decoding with fallback
        # ------------------------------------------------------------------
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "robot_plan",
                "strict": True,
                "schema": self.schema
            }
        }

        print("[PLANNER] Asking Qwen...")
        try:
            response = dashscope.Generation.call(
                model="qwen-plus",
                messages=messages,
                result_format="message",
                response_format=response_format
            )
        except TypeError:
            print("[PLANNER] Schema constraint not supported by API, falling back...")
            response = dashscope.Generation.call(
                model="qwen-plus",
                messages=messages,
                result_format="message"
            )
        # ------------------------------------------------------------------

        if response.status_code != 200 or response.output is None:
            print(f"[PLANNER ERROR] API status: {response.status_code}")
            return None

        content = response.output.choices[0].message.content
        print(f"[PLANNER] Raw response:\n{content}\n")

        parsed = self._extract_json(content)
        if parsed is None:
            return None
        intent, plan = parsed
        self.last_intent = intent
        self.last_raw = content

        if not self._validate(plan, intent):
            print("[PLANNER] Validation failed.")
            return None

        # ---- Adjustment intent -> session drift (R3). Same arithmetic as
        # before, applied BEFORE register fill so "stir faster" speeds up
        # that very stir. The factor comes from people.yaml, keyed by the
        # intent label - never from the LLM. ----
        adj = intent["adjustment"]
        if adj != "none":
            factor = float(self.people.get("adjustments", {}).get(adj, 1.0))
            lo, hi = self.people.get("bounds", {}).get("drift", [0.3, 1.5])
            self.session_drift = min(hi, max(lo, self.session_drift * factor))
            print(f"[SESSION] drift x{factor:g} -> {self.session_drift:.2f} "
                  f"(volatile, session '{owner}')")

        # ---- Store intent (R4/R5): open the echo-confirmation gate.
        # Nothing is written until a deterministic yes next turn. ----
        if intent["store"]:
            self._open_store_gate(owner, user_request)

        # Deterministic register fill (rule 21) + duplicate-step filter.
        # Manner and recipient come from the intent header, not word lists.
        plan = self._apply_registers(plan, user_request, intent["manner"],
                                     intent.get("recipient"))
        if plan is None:
            return None
        if not plan:
            if adj != "none" or intent["store"]:
                return None      # valid session command, handled above
            print("[PLANNER ERROR] Empty plan - treating as refusal.")
            return None

        print("[PLANNER] Plan validated.")
        try:
            self._remember_exchange(user_request, content)
        except Exception as e:
            print(f"[PLANNER WARN] could not store history: {e}")
        return plan

    # ------------------------------------------------------------------
    # Recipient resolution - DETERMINISTIC, done HERE in Python. Person
    # names are scanned from the request text against the roster; manner
    # arrives as the LLM's intent-header label. The LLM may set a step's
    # "recipient" field, but the fallback (and default) resolution never
    # depends on it. All numbers come from people.yaml - never from the LLM.
    # ------------------------------------------------------------------
    def _unknown_person_in_request(self, user_request):
        """Rule 21: 'for X' where X is not a known person, register, or
        declared non-person token -> return X so the planner can ask
        instead of guess. Returns None when everything mentioned is known."""
        req = " " + user_request.lower() + " "
        people = self.people.get("people", {})
        registers = self.people.get("registers", {})
        for m in re.finditer(r"\bfor\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)?)", req):
            tok = m.group(1)
            if any(tok.startswith(r) for r in registers):
                continue                      # "for slower profile", ...
            tok = tok.split()[0]              # the first word decides otherwise
            if tok in self._FOR_STOPWORDS or tok in people or tok in registers \
                    or tok in self._FOR_X_NONPERSONS:
                continue
            return tok
        return None

    def _resolve_person(self, user_request, manner="none",
                        intent_recipient=None):
        """-> (person_name, register_name, problem). The person is the
        recipient, else the session owner. Two recipient signals exist:
        the deterministic roster scan of the request text, and the LLM's
        intent-header recipient (enum-bound to the roster). If BOTH fire
        and disagree -> ask, never reroute. The manner register comes from
        the intent header (gentle -> slower profile, brisk -> faster profile).
        Register conflicts resolve to the SLOWER profile (R8)."""
        req = " " + user_request.lower() + " "
        words = self._words(user_request)
        people = self.people.get("people", {})
        registers = self.people.get("registers", {})
        person = None
        found = []
        text_person = None
        for name, info in people.items():
            if name in words and info.get("register"):
                text_person = name
                found.append(info["register"])
        if intent_recipient and intent_recipient != "none" \
                and intent_recipient in people:
            if text_person is not None and text_person != intent_recipient:
                return None, None, ("conflict", text_person, intent_recipient)
            person = intent_recipient
            found.append(people[intent_recipient]["register"])
        elif text_person is not None:
            person = text_person
        manner_reg = self._MANNER_TO_REGISTER.get(manner)
        if manner_reg:
            found.append(manner_reg)
        for m in re.finditer(r"\bfor\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)?)", req):
            tok = m.group(1)
            if any(tok.startswith(r) for r in registers):
                continue                      # "for slower profile", ...
            tok = tok.split()[0]              # the first word decides otherwise
            if tok in self._FOR_STOPWORDS or tok in people or tok in registers \
                    or tok in self._FOR_X_NONPERSONS:
                continue
            return None, None, tok
        register = None
        if found:
            register = min(found, key=lambda r: registers.get(r, {}).get("speed", 100))
        cur = self.get_current_user()
        if person is None and cur in people:
            person = cur
        if register is None and person in people:
            register = people[person].get("register")
        return person, register, None

    def _apply_registers(self, plan, user_request, manner="none",
                         intent_recipient=None):
        """R6 precedence: curated profile > stored prefs > session drift >
        register default. effective = stored(person) x drift, clamped.
        Motion params take the full factor; dwell takes half (R7).
        'manner'/'intent_recipient' are intent-header labels, not scanned
        words."""
        registers = self.people.get("registers", {})
        people = self.people.get("people", {})
        bounds = self.people.get("bounds", {})
        prefs = self._load_preferences()      # reloaded per plan (runtime edits)
        stored = prefs.get("stored", {})
        req_person, default_reg, problem = self._resolve_person(
            user_request, manner, intent_recipient)
        if isinstance(problem, tuple):
            print(f"[PLANNER ERROR] Recipient conflict: the request text "
                  f"suggests '{problem[1]}' but the intent header says "
                  f"'{problem[2]}'. Asking instead of guessing - who is "
                  f"the action for?")
            return None
        if problem is not None:
            print(f"[PLANNER ERROR] Unknown person '{problem}' - not in "
                  f"people.yaml. Asking instead of guessing (rule 21).")
            return None

        d_lo, d_hi = bounds.get("drift", [0.3, 1.5])
        s_lo, s_hi = bounds.get("speed", [10, 100])
        o_lo, o_hi = bounds.get("omega_scale", [0.3, 1.5])
        dwell_floor = bounds.get("dwell_floor_s", 1.5)

        for step in plan:
            action = step.get("action")
            if action not in self.MOTION_ACTIONS \
                    and action not in ("stir", "pour", "hand_over"):
                step.pop("recipient", None)
                continue
            rec = step.pop("recipient", None)
            person, register = req_person, default_reg
            if rec in people:
                person, register = rec, people[rec].get("register", register)
            elif rec in registers:
                register = rec
            if register is None or register not in registers:
                continue
            # effective multiplier (R6): stored prefs of the person the
            # action is FOR, x the session drift. Clamped (R8).
            p_mult = stored.get(person or "", {}).get("mult", 1.0)
            eff = min(d_hi, max(d_lo, p_mult * self.session_drift))
            values = registers[register]

            def scale(v):
                return int(min(s_hi, max(s_lo, round(v * eff))))

            if action in ("stir", "pour"):
                step["omega_scale"] = round(min(o_hi, max(
                    o_lo, values.get("omega_scale", 1.0) * eff)), 3)
            if action == "stir":
                step["revolutions"] = self._resolve_revolutions(
                    manner, step.get("revolutions"))
            if action in ("pick", "pick_and_place"):
                step["gripper_force"] = self._resolve_gripper_force(
                    step.get("object_id"), manner)
            if action in ("stir", "hand_over") or action in self.MOTION_ACTIONS:
                if action != "pour":
                    step["speed"] = scale(values.get("speed", 30))
                    step["accel"] = scale(values.get("accel", 100))
            if action == "hand_over" and "confirm_dwell_s" in values:
                d = values["confirm_dwell_s"]
                if abs(eff - 1.0) > 1e-6:            # R7: half strength
                    half = 1.0 + (eff - 1.0) * 0.5
                    d = max(dwell_floor, d / half)
                step["confirm_dwell_s"] = round(d, 2)

        # Drop consecutive duplicate steps (constrained-decoding artifact)
        deduped = []
        for step in plan:
            if not deduped or step != deduped[-1]:
                deduped.append(step)
        return deduped

    def _resolve_revolutions(self, manner, emitted=None):
        """Stir extent, clamped to the declared bounds [1, 6].
        Presence-based: an emitted value IS the user's wish (rule 13: the
        LLM emits revolutions only when the request states a count or a
        thoroughness degree) -> take it as stated, clamped. Absent -> the
        manner label ('well' -> 4) -> default 2. Never above the maximum."""
        lo, hi = self._REVOLUTIONS_BOUNDS
        if isinstance(emitted, (int, float)) and not isinstance(emitted, bool):
            v = int(emitted)                 # user stated a count
        else:
            v = self._REVOLUTIONS_MAP.get(manner, self._REVOLUTIONS_DEFAULT)
        clamped = int(max(lo, min(hi, v)))
        if clamped != v:
            print(f"[RESOLVER] revolutions {v} out of bounds -> {clamped} "
                  f"(declared [{lo}, {hi}])")
        return clamped

    def _resolve_gripper_force(self, object_id, manner="none"):
        """Object sets the band, instruction moves inside it. Ceiling from
        the object's handling class (scene.yaml); manner "gentle" steps the
        force down once, clamped at the class floor so the hold is never
        weakened past what the object tolerates ("pour gently" on a fragile
        cup still grips at 60+, never 20). Unknown object/class -> default
        ceiling (fail firm, never fail loose)."""
        lo, hi = self._FORCE_BOUNDS
        obj = self.scene.get("poses", {}).get(object_id, {})
        cls = obj.get("handling_class", "standard")
        ceiling = self._FORCE_MAP.get(cls, self._FORCE_DEFAULT)
        floor = max(lo, self._FORCE_FLOOR.get(cls, lo))
        v = ceiling
        if manner == "gentle":
            v = max(floor, ceiling - self._FORCE_GENTLE_STEP)
            if v != ceiling:
                print(f"[RESOLVER] gentle pick: force {ceiling} -> {v} "
                      f"(class '{cls}', floor {floor})")
        return int(max(lo, min(hi, v)))

    def _extract_json(self, text):
        """-> (intent, plan) or None. Expects the {intent, plan} envelope;
        tolerates a bare plan array (no-schema fallback path) by treating
        the intent as all-none."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[PLANNER ERROR] JSON parse failed: {e}")
            print(f"Text was: {text[:300]}")
            return None
        if isinstance(obj, list):
            print("[PLANNER WARN] bare plan array (no intent header) - "
                  "treating intent as all-none.")
            return {"manner": "none", "adjustment": "none", "store": False}, obj
        if not isinstance(obj, dict) or "plan" not in obj:
            print("[PLANNER ERROR] Response is not the {intent, plan} envelope.")
            return None
        intent = obj.get("intent")
        if intent is None:
            print("[PLANNER WARN] no intent header - treating as all-none.")
            intent = {"manner": "none", "adjustment": "none", "store": False}
        return intent, obj["plan"]

    def _validate(self, plan, intent):
        # Intent header first (post-hoc backstop for the no-schema fallback
        # path; with constrained decoding these can never be wrong).
        valid_recipients = set(self.people.get("people", {})) | {"none"}
        if not isinstance(intent, dict) \
                or intent.get("manner") not in INTENT_MANNERS \
                or intent.get("adjustment") not in INTENT_ADJUSTMENTS \
                or not isinstance(intent.get("store"), bool) \
                or intent.get("recipient") not in valid_recipients:
            print(f"[PLANNER ERROR] Bad intent header: {intent}")
            return False
        if not isinstance(plan, list):
            print("[PLANNER ERROR] Plan is not a JSON array.")
            return False

        valid_actions = {p["id"] for p in self.registry.get("primitives", [])}
        valid_objects = set(self.scene.get("poses", {}).keys())

        for i, step in enumerate(plan):
            action = step.get("action")
            if action not in valid_actions:
                print(f"[PLANNER ERROR] Step {i}: unknown action '{action}'")
                return False
            for key in ["object_id", "cup_id", "destination_id", "location_id"]:
                if key in step and step[key] not in valid_objects:
                    print(f"[PLANNER ERROR] Step {i}: unknown object '{step[key]}'")
                    return False
            # Picks go through the camera: only YOLO-detectable objects.
            if action in ("pick", "pick_and_place") \
                    and step.get("object_id") not in detected_object_ids(self.scene):
                print(f"[PLANNER ERROR] Step {i}: '{step.get('object_id')}' is not a "
                      f"camera-detected object - it cannot be picked")
                return False
            # ADDED: recipient must be a known person or register
            if "recipient" in step and step["recipient"] is not None:
                r = step["recipient"]
                if r not in self.people.get("people", {}) \
                        and r not in self.people.get("registers", {}):
                    print(f"[PLANNER ERROR] Step {i}: unknown recipient '{r}'")
                    return False
            # ADDED: handover knobs within declared bounds
            if "confirm_dwell_s" in step:
                v = step["confirm_dwell_s"]
                if not isinstance(v, (int, float)) or not (0.5 <= v <= 10):
                    print(f"[PLANNER ERROR] Step {i}: confirm_dwell_s={v} out of range")
                    return False
            # ADDED: motion profile values must be sane percents
            for key in ["speed", "accel"]:
                if key in step:
                    v = step[key]
                    if not isinstance(v, (int, float)) or isinstance(v, bool) \
                            or not (1 <= v <= 100):
                        print(f"[PLANNER ERROR] Step {i}: {key}={v} out of range 1-100")
                        return False
            # ADDED: pose-reference backstop (covers the no-schema fallback
            # path; with schema constraint these can never be generated)
            for key in ["target_pose", "pose"]:
                if key in step and not self._valid_pose_ref(step[key]):
                    print(f"[PLANNER ERROR] Step {i}: unresolvable {key} '{step[key]}'")
                    return False
        return True

    def _valid_pose_ref(self, ref):
        # executor._resolve() also accepts a raw 6-vector inline
        if isinstance(ref, list):
            return len(ref) == 6 and all(isinstance(v, (int, float)) for v in ref)
        return ref in self.pose_refs
