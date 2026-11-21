// Human-readable rendering of Qwen's answer: plan steps, intent, raw JSON.
"use strict";

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}

// "red_block.approach_pose" -> "red block · approach pose"
function humanize(id) {
  return String(id).replace(/\.value$/, "").split(".")
    .map((p) => p.replace(/_/g, " ")).join(" · ");
}

function cap(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

const STEP_TITLES = {
  go_to: (s) => `Move to ${humanize(s.target_pose)}`,
  pick: (s) => `Pick the ${humanize(s.object_id)}`,
  place: (s) => `Place it at the ${humanize(s.location_id)}`,
  pick_and_place: (s) => `Pick the ${humanize(s.object_id)} and put it in the ${humanize(s.destination_id)}`,
  stir: (s) => `Stir the ${humanize(s.cup_id || "coffee_cup")} — ${s.revolutions || 2} turns`,
  pour: () => "Pour in place",
  hand_over: () => "Hand the object over",
  gripper_open: () => "Open the gripper",
  gripper_close: () => "Close the gripper",
};
// Keys already said in the title.
const TITLE_KEYS = new Set(["action", "target_pose", "object_id", "location_id",
  "destination_id", "cup_id", "revolutions"]);

function stepTitle(step) {
  const f = STEP_TITLES[step.action];
  return f ? f(step) : cap(humanize(step.action));
}

const META_ORDER = ["speed", "accel", "omega_scale", "confirm_dwell_s"];

function stepMeta(step) {
  const out = [];
  const rank = (k) => (META_ORDER.includes(k) ? META_ORDER.indexOf(k) : META_ORDER.length);
  const entries = Object.entries(step).sort(([a], [b]) => rank(a) - rank(b));
  for (const [k, v] of entries) {
    if (TITLE_KEYS.has(k)) continue;
    if (k === "speed") out.push(`speed ${v}%`);
    else if (k === "accel") out.push(`accel ${v}%`);
    else if (k === "omega_scale") out.push(`tempo ×${v}`);
    else if (k === "confirm_dwell_s") out.push(`holds ${v} s before release`);
    else out.push(`${humanize(k)}: ${typeof v === "object" ? JSON.stringify(v) : v}`);
  }
  return out;
}

const ADJUSTMENTS = {
  down_strong: "Much slower", down: "Slower", down_slight: "A bit slower",
  up_slight: "A bit faster", up: "Faster", up_strong: "Much faster",
};

// Intent header -> [{label, tone}]
function intentChips(intent) {
  if (!intent) return [];
  const chips = [];
  if (intent.recipient && intent.recipient !== "none") chips.push({ label: `For ${cap(intent.recipient)}`, tone: "who" });
  if (intent.manner === "gentle") chips.push({ label: "Gentle", tone: "calm" });
  if (intent.manner === "brisk") chips.push({ label: "Brisk", tone: "brisk" });
  if (ADJUSTMENTS[intent.adjustment]) chips.push({ label: ADJUSTMENTS[intent.adjustment], tone: "speed" });
  if (intent.store) chips.push({ label: "Remember speed", tone: "store" });
  return chips;
}

// Pretty, syntax-coloured JSON built from DOM nodes (never innerHTML).
function jsonView(value) {
  const pre = el("pre", "json");
  const put = (text, cls) => pre.append(cls ? el("span", cls, text) : document.createTextNode(text));
  const walk = (v, depth) => {
    const pad = "  ".repeat(depth + 1), end = "  ".repeat(depth);
    if (Array.isArray(v)) {
      if (!v.length) return put("[]");
      put("[\n");
      v.forEach((x, i) => { put(pad); walk(x, depth + 1); put(i < v.length - 1 ? ",\n" : "\n"); });
      put(end + "]");
    } else if (v && typeof v === "object") {
      const keys = Object.keys(v);
      if (!keys.length) return put("{}");
      put("{\n");
      keys.forEach((k, i) => {
        put(pad); put(JSON.stringify(k), "j-key"); put(": ");
        walk(v[k], depth + 1); put(i < keys.length - 1 ? ",\n" : "\n");
      });
      put(end + "}");
    } else if (typeof v === "string") put(JSON.stringify(v), "j-str");
    else if (typeof v === "number") put(String(v), "j-num");
    else if (typeof v === "boolean") put(String(v), "j-bool");
    else put("null", "j-null");
  };
  walk(value, 0);
  return pre;
}

// Raw model text -> formatted JSON view, or the text itself if not JSON.
function rawView(text) {
  let t = String(text).trim();
  if (t.startsWith("```")) t = t.replace(/^```\w*\n?/, "").replace(/```$/, "").trim();
  try { return jsonView(JSON.parse(t)); } catch (_) { return el("pre", "json", text); }
}
