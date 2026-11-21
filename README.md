# MANNER: personalizing actions to user

A robot assistant driven by natural language. A Qwen LLM (DashScope) turns a
request into a validated plan. A behaviour tree then runs that plan on the
FAIRINO F5 with learned skills: DMP stirring, DMP pouring, pick/place and hand-over.
Motion speed adapts to the person the action is for (`config/people.yaml`).

## Project layout

```
Dmp_test/
├── dmp_assistant/              # the Python package
│   ├── web/                    # web front-end (main entry point)
│   │   ├── server.py           # Flask app: face login, session API, enrollment
│   │   ├── logbuffer.py        # stdout -> timestamped log lines the page polls
│   │   ├── present.py          # planner messages -> plain chat notices
│   │   ├── templates/          # base / login / app / enroll pages
│   │   └── static/             # CSS, JS, img/robot_logo.* (logo)
│   ├── auth/face_auth.py       # face identification against data/faces/
│   ├── session.py              # one user's conversation (shared by web + terminal)
│   ├── launcher.py             # same session in the terminal
│   ├── paths.py                # all file locations, anchored on the project root
│   ├── planning/
│   │   ├── planner.py          # LLM -> {intent, plan}; users, drift, stored prefs
│   │   └── schema_builder.py   # JSON schema for constrained decoding
│   ├── execution/
│   │   └── executor.py         # plan -> py_trees behaviour tree -> robot
│   ├── skills/                 # robot skills (importable + runnable alone)
│   │   ├── motion.py           # movel / movej / IK
│   │   ├── gripper.py          # gripper_open / gripper_close
│   │   ├── robot_state.py      # current joints / TCP pose
│   │   ├── dmp.py              # RhythmicDMP runtime
│   │   ├── pour.py             # DMP pouring skill
│   │   └── hand_over.py        # robot-to-human hand-over
│   ├── perception/             # camera pick
│   │   ├── camera.py           # RealSense colour + aligned depth
│   │   ├── detector.py         # YOLOv8-OBB (model/best.pt)
│   │   ├── locator.py          # detection + depth + hand-eye -> grasp poses (live view)
│   │   ├── live.py             # one thread: RealSense + YOLO for the web view AND the pick
│   │   └── geometry.py         # frame math, Park-Martin hand-eye solver
│   ├── tools/record_dmp.py     # record a demonstration and fit a DMP
│   ├── tools/calibrate_hand_eye.py  # eye-in-hand calibration (chessboard, OpenCV)
│   └── evaluation/lexicon_battery.py   # seeded phrasing battery (stub / live)
├── config/                     # static config, edited by humans
│   ├── registry.yaml           # skill catalogue shown to the LLM
│   ├── scene.yaml              # named poses
│   ├── people.yaml             # users, registers, adjustment factors, bounds
│   ├── perception.yaml         # camera pick: scan pose, thresholds, gripper, workspace
│   └── hand_eye.yaml           # T_flange_cam (written by calibrate_hand_eye)
├── model/                      # YOLOv8-OBB weights: best.pt, last.pt
├── data/
│   ├── preferences.yaml        # stored per-user speed multipliers (runtime state)
│   ├── faces/<name>/*.jpg      # enrolled face photos (biometric, git-ignored)
│   ├── dmp/                    # learned primitives (.npy)
│   └── demos/                  # raw demonstrations / verification logs (.npz)
├── tests/
│   ├── test_intent_gate.py     # offline planner checks (LLM stubbed)
│   ├── test_web_session.py     # offline web flow: login -> plan -> execute -> logout
│   ├── test_perception.py      # offline camera pick: frames, hand-eye, locator, executor
│   ├── test_pour_guard.py      # offline pour: wrist guard, realign, joint-6 fallback
│   └── manual/                 # scripts that talk to the robot / simulator
├── examples/                   # scripted demos without the planner
├── docs/state_model.yaml       # executor belief-state model (documentation)
├── results/battery/            # evaluation results (JSON / JSONL)
├── logs/                       # old FAIRINO SDK logs
└── archive/legacy/             # superseded experiments, kept for reference
```

## Setup

```bash
conda activate fairino                       # Python 3.10 (the SDK .so is built for 3.10)
pip install -r requirements.txt      # face-recognition compiles dlib (needs cmake) and downloads ~100 MB of models
export DASHSCOPE_API_KEY=<your key>          # never hard-code it in a file
```

The system runs on a FAIRINO F5 collaborative arm. The FAIRINO SDK is not a
pip package and is **not part of this repository** - download it from
FAIRINO's official SDK page (https://github.com/FAIR-INNOVATION/fairino-python-sdk;
the Linux build targets Python 3.10) and make it importable as `fairino`,
e.g. through a `site-packages/fairino-sdk.pth` pointing at the SDK's `linux`
directory, or `export PYTHONPATH=/path/to/fairino-sdk/linux`.

## Web session with face login

```bash
python -m dmp_assistant.web.server        # --port 5000  --robot-ip ...  --no-browser
```

1. The browser opens `http://127.0.0.1:5000`. The login page uses the webcam, and the
   session opens once the same enrolled person is recognised on 3 frames in a row.
2. Session page (English UI): ROBOT logo on the left of the header, the
   logged-in person on the right.
   - **Left, chat:** requests go in here, as they used to in the terminal.
     Qwen's answer is shown as a readable plan: steps in plain words, speed and
     tempo, and chips for the intent (gentle / brisk, who it is for, slower /
     faster, remember speed). **Run on robot** and **Discard** act on the plan.
     The "remember this speed?" question gets **Yes** / **No** buttons, and
     planner errors are rewritten as plain sentences.
   - **Right, top: live camera.** The RealSense image with the YOLO boxes
     (MJPEG, about 10 fps). While a pick looks for an object, its box turns
     orange and the header line says "locating bottle", then where it was
     found. **Hide** stops the stream. Start the server with `--no-camera` to
     run without it.
   - **Right, logs:** everything the planner, executor and skills print, live
     and timestamped. Qwen's raw JSON is formatted and coloured. You can filter
     All / Planner / Robot / Errors. On narrow screens the panel becomes a drawer
     that the **Logs** button opens.
3. **Sign out** closes the session and returns to the face login.

The logged-in person becomes the planner's `current_user`, so their register
and stored preferences apply. `user <name>` is disabled on the web.

**Voice input (push-to-talk):** hold the mic button in the composer and
speak; release to send. The browser records a 16 kHz mono WAV and POSTs it
to `/api/asr`, which transcribes it with DashScope `qwen3-asr-flash` (the
same `DASHSCOPE_API_KEY` as the planner; the language is auto-detected -
English and Chinese both work). The transcript lands in the input bar and
flows through the normal plan-and-run path. Code: `dmp_assistant/web/asr.py`
and the `/api/asr` route in `dmp_assistant/web/server.py`; the mic button
and the WAV recording live in `dmp_assistant/web/static/app.js`.

**Enrolling a face:** the terminal prints an `Enrollment : .../enroll?token=...`
link at start-up (a new token on every run). Pick the person, who must be in
`config/people.yaml`, and capture about 5 photos. They are saved in
`data/faces/<name>/`. You can also drop photos there by hand, with one face per
photo, and restart.

**Logo:** put the file at `dmp_assistant/web/static/img/robot_logo.png` (or `.svg`/`.jpg`).
Until it is there, the header shows a text "ROBOT" mark.

**Security limits:** the server listens on `127.0.0.1` only and allows one active
session at a time (there is one robot). Face recognition identifies people; it
does not secure access. A photo of an enrolled person can fool it, because
there is no liveness check.

## Camera pick (YOLO + RealSense)

Objects declared in `config/scene.yaml` with `detect: <class>` (`bottle`,
`cylinder`, `hollow`, `sanitizer`, the classes of `model/best.pt`) have no
stored pose. "pick the bottle" makes the executor:

1. go to `locate.scan_pose` (`perception.yaml`, default `initial_pose`) and open the gripper;
2. detect the class on 5 frames (YOLOv8-OBB). Each frame gives the box centre
   and the top height (nearest depths in the box). The camera point goes to
   the base frame through `hand_eye.yaml` and the flange pose. The result is
   the median of the frames; too few frames or too much spread fails the pick;
3. compute a top-down grasp: rx/ry from the scan pose, rz turned so the jaws close
   across the object's long axis (round objects keep the scan rz), fingertip
   `depth_mm` below the top, never under `table_z_mm`, inside `workspace`;
4. hover, re-measure and correct XY, descend, close, check the gripper did not
   close on air, retract.

**Every pick uses the camera.** Only these camera objects can be picked; the
planner refuses "pick the red block", and the executor raises an error for any
object without a `detect` class. There is no stored-pose fallback. Stored poses
(`red_block.grasp_pose`, ...) are still valid `go_to` targets and stir centres.

**Calibrate the camera first** (eye-in-hand, printed chessboard fixed on the table):

```bash
python -m dmp_assistant.tools.calibrate_hand_eye --cols 9 --rows 6 --square-mm 25
```

`d` drag-teach on/off, `space` capture (drag off, board visible), `c` compute and
save. Take 12-20 captures with different positions AND wrist tilts. The spread
printed for the board should be a few mm. Until then `hand_eye.yaml` holds
the `T_gc` imported from `grasping-v6`.

**Check before picking** (no motion, prints base coordinates of each detection):

```bash
python -m dmp_assistant.perception.locator --robot-ip 192.168.58.4
```

## Running

Run the `python -m ...` commands from the project root (`Dmp_test/`). A script
run by its file path, such as `python dmp_assistant/launcher.py`, works from
any folder.

| What | Command |
|---|---|
| Web session (face login) | `python -m dmp_assistant.web.server` |
| Terminal session | `python -m dmp_assistant.launcher` |
| Offline planner tests | `python tests/test_intent_gate.py` |
| Offline web flow test | `python tests/test_web_session.py` |
| Phrasing battery, offline | `python -m dmp_assistant.evaluation.lexicon_battery --stub --n 100` |
| Phrasing battery, live Qwen | `python -m dmp_assistant.evaluation.lexicon_battery --n 100 --seed 42` |
| Pour skill alone | `python -m dmp_assistant.skills.pour [tilt_deg] [dwell_s] [speed] [revs]` |
| Hand-over alone | `python -m dmp_assistant.skills.hand_over` |
| Record a pouring DMP | `python -m dmp_assistant.tools.record_dmp` (`inspect` to view the saved one) |
| Offline camera-pick test | `python tests/test_perception.py` |
| Offline pour-guard test | `python tests/test_pour_guard.py` |
| Hand-eye calibration | `python -m dmp_assistant.tools.calibrate_hand_eye` |
| Live detection view | `python -m dmp_assistant.perception.locator [--robot-ip ...]` |
| Executor smoke check (sim) | `python tests/manual/check_executor_bt.py` |

Session commands: `user <name>`, `state`, `reset`, `reset prefs`, `quit`.

## Robot IPs

- Real robot: `192.168.58.4`. The launcher, `pour.py`, `hand_over.py`, `robot_state.py` and `record_dmp.py` use it.
- Simulator: `192.168.85.130`. `motion.py` (its `__main__` only), `tests/manual/*` and `examples/` use it.

Check the IP in the script before you run anything on the arm.

## Conventions

- Get file paths from `dmp_assistant.paths`. Never write bare `"scene.yaml"`-style relative paths.
- Import across modules absolutely, for example `from dmp_assistant.skills.motion import movel`.
- A module you can run alone inserts the project root into `sys.path` under `if __name__ == "__main__":`.
  Nothing else changes `sys.path`.
- The FAIRINO SDK writes `fairino_<date>.log` into the current working directory.
