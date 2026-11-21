"""Single source of truth for file locations.

Every path is anchored on the project root (the folder that contains
dmp_assistant/), so scripts behave the same whatever the working
directory is.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
DMP_DIR = DATA_DIR / "dmp"
DEMOS_DIR = DATA_DIR / "demos"
RESULTS_DIR = ROOT / "results" / "battery"
LOGS_DIR = ROOT / "logs"

# Static configuration (edited by humans)
REGISTRY_YAML = CONFIG_DIR / "registry.yaml"
SCENE_YAML = CONFIG_DIR / "scene.yaml"
PEOPLE_YAML = CONFIG_DIR / "people.yaml"
PERCEPTION_YAML = CONFIG_DIR / "perception.yaml"
HAND_EYE_YAML = CONFIG_DIR / "hand_eye.yaml"

# Runtime state (written by the planner's "remember" gate)
PREFERENCES_YAML = DATA_DIR / "preferences.yaml"

# Enrolled face photos for authentication: data/faces/<name>/*.jpg
FACES_DIR = DATA_DIR / "faces"

# Learned primitives
POURING_DMP = DMP_DIR / "learned_pouring_dmp.npy"
STIRRING_DMP = DMP_DIR / "learned_stirring_dmp.npy"

# Object detector (YOLOv8-OBB trained on the table objects)
MODEL_DIR = ROOT / "model"
YOLO_WEIGHTS = MODEL_DIR / "best.pt"

# Hand-eye calibration samples (robot pose + board pose per capture)
CALIB_DIR = DATA_DIR / "calibration"
