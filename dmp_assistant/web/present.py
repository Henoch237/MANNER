"""Turn the planner's console messages into chat notices.

The planner reports through print() (the terminal launcher relies on it).
The chat shows a friendly sentence instead; the raw line stays in the log
panel. Lines that the chat renders from structured data (plan, drift,
store question, robot state) or that are pure progress are dropped here.
"""
import re

# (pattern on the text after the [TAG] prefix, level, message template)
_RULES = [
    (r"Unknown person '(?P<n>[^']+)'", "error",
     "I don't know “{n}” — they are not in the people list. Who is this for?"),
    (r"Recipient conflict: .*suggests '(?P<a>[^']+)' but the intent header says '(?P<b>[^']+)'",
     "error", "Is this for {a} or for {b}? Please say who it is for."),
    (r"R5: '(?P<n>[^']+)' is not the session owner", "error",
     "You can only save your own speed preference, not {n}'s. Nothing was saved."),
    (r"Empty plan", "error",
     "I couldn't turn that into robot actions. Try rephrasing the request. "
     "Only camera objects can be picked: bottle, cylinder, hollow, sanitizer."),
    (r"Step (?P<i>\d+): '(?P<v>[^']+)' is not a camera-detected object", "error",
     "“{v}” can't be picked: picks use the camera, which knows bottle, cylinder, "
     "hollow and sanitizer."),
    (r"API status: (?P<s>\d+)", "error",
     "The language model is unavailable right now (API status {s})."),
    (r"JSON parse failed|Response is not the \{intent, plan\} envelope|Bad intent header",
     "error", "The language model returned an answer I could not read. Please try again."),
    (r"Step (?P<i>\d+): unknown (?P<what>action|object|recipient) '(?P<v>[^']+)'", "error",
     "The plan used an unknown {what} “{v}” (step {i}). Please rephrase."),
    (r"Step (?P<i>\d+): (?P<rest>.*out of range.*)", "error",
     "The plan had an invalid value (step {i}: {rest})."),
    (r"Step (?P<i>\d+): unresolvable (?P<k>\w+) '(?P<v>[^']+)'", "error",
     "The plan referenced an unknown pose “{v}” (step {i})."),
    (r"stored '(?P<n>[^']+)': (?P<old>[\d.]+) x drift [\d.]+ -> (?P<new>[\d.]+)", "success",
     "Saved as your default speed (multiplier {old} → {new})."),
    (r"store declined", "info", "OK — your default speed is unchanged."),
    (r"store confirmation expired", "info",
     "Speed not saved (no yes/no answer). Handling your new request instead."),
    (r"nothing to store for", "info",
     "Nothing to save — the speed is already at your default."),
    (r"reset stored prefs for", "success", "Your saved speed preference was deleted."),
    (r"has no stored prefs", "info", "You have no saved speed preference."),
    (r"new session - history and drift cleared", "success",
     "New conversation started — history and speed adjustments cleared."),
    (r"user switching is disabled", "info",
     "Switching user is disabled here. Sign out and sign in with face recognition."),
    (r"pending store discarded", "info", "The pending speed question was cancelled."),
    (r"no plan to execute", "info", "There is no plan waiting to run."),
]

# Rendered from structured data or pure progress -> not a chat notice.
_SKIP = re.compile(
    r"Remember this speed|^drift x|current_user ->|Asking Qwen|Raw response|"
    r"Plan validated|Final plan|Validation failed|Schema constraint|could not store history|"
    r"bare plan array|no intent header")

_TAG = re.compile(r"^\[(?P<tag>[A-Z ]+?)\]\s*(?P<text>.*)$")


def notices(lines):
    """[{"text": ...}] log lines -> [{"level", "text"}] chat notices."""
    out = []
    for line in lines:
        m = _TAG.match(line["text"].strip())
        if not m:
            continue
        tag, text = m["tag"], m["text"]
        if not (tag.startswith("PLANNER") or tag.startswith("SESSION")):
            continue
        if _SKIP.search(text):
            continue
        for pattern, level, template in _RULES:
            hit = re.search(pattern, text)
            if hit:
                out.append({"level": level, "text": template.format(**hit.groupdict())})
                break
        else:
            level = "error" if "ERROR" in tag else "warn" if "WARN" in tag else "info"
            out.append({"level": level, "text": text})
    return out
