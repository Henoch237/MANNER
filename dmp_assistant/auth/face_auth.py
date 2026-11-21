"""Face identification against enrolled photos.

Enrolled faces live in data/faces/<name>/*.jpg, one folder per person of
config/people.yaml. A folder whose name is not in the roster is ignored,
so recognition can never log in someone the planner does not know.

face_recognition (dlib) is imported lazily: the rest of the project runs
without it.
"""
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Euclidean distance between 128-d face encodings; face_recognition's own
# default is 0.6. Lower = stricter (fewer false accepts, more retries).
DEFAULT_TOLERANCE = 0.5


def _fr():
    try:
        # Import the models first: when they fail, face_recognition itself
        # prints a misleading hint and calls quit() instead of raising.
        import face_recognition_models  # noqa: F401
        import face_recognition
    except ImportError as e:
        if e.name == "pkg_resources":
            # face_recognition_models still uses pkg_resources, which
            # setuptools >= 81 no longer ships.
            raise RuntimeError(
                'face_recognition needs pkg_resources: pip install "setuptools<81"') from e
        raise RuntimeError(
            f"face_recognition is not usable ({e}): pip install face-recognition") from e
    return face_recognition


def encode_single_face(rgb):
    """-> (encoding, None) when the image holds exactly one face,
    else (None, "no_face" | "multiple")."""
    fr = _fr()
    boxes = fr.face_locations(rgb)
    if not boxes:
        return None, "no_face"
    if len(boxes) > 1:
        return None, "multiple"
    return fr.face_encodings(rgb, known_face_locations=boxes)[0], None


class FaceDB:
    def __init__(self, faces_dir, roster, tolerance=DEFAULT_TOLERANCE):
        self.faces_dir = Path(faces_dir)
        self.roster = set(roster)
        self.tolerance = tolerance
        self.names = []          # one entry per encoding
        self.encodings = []

    def load(self):
        """(Re)encode every enrolled photo. -> {name: n_photos_used}."""
        fr = _fr()
        names, encodings, counts = [], [], {}
        if self.faces_dir.is_dir():
            for person_dir in sorted(p for p in self.faces_dir.iterdir() if p.is_dir()):
                name = person_dir.name
                if name not in self.roster:
                    print(f"[FACE] ignoring '{name}': not in people.yaml")
                    continue
                for img_path in sorted(person_dir.iterdir()):
                    if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    enc, problem = encode_single_face(fr.load_image_file(str(img_path)))
                    if problem:
                        print(f"[FACE] skipping {img_path.name} ({name}): {problem}")
                        continue
                    names.append(name)
                    encodings.append(enc)
                    counts[name] = counts.get(name, 0) + 1
        self.names, self.encodings = names, encodings
        print(f"[FACE] enrolled: {counts or 'nobody yet'}")
        return counts

    @property
    def empty(self):
        return not self.encodings

    def identify(self, rgb):
        """-> (status, name, distance). status: "match", "unknown",
        "no_face", "multiple" or "no_enrollment"."""
        if self.empty:
            return "no_enrollment", None, None
        enc, problem = encode_single_face(rgb)
        if problem:
            return problem, None, None
        dists = _fr().face_distance(np.array(self.encodings), enc)
        # Best distance per person, so one bad photo cannot dominate.
        best = {}
        for name, d in zip(self.names, dists):
            best[name] = min(best.get(name, np.inf), float(d))
        name, dist = min(best.items(), key=lambda kv: kv[1])
        if dist > self.tolerance:
            return "unknown", None, dist
        return "match", name, dist
