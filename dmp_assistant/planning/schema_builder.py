import yaml

from dmp_assistant import paths

# ------------------------------------------------------------------
# Intent header (lexicon-freedom contract). The LLM routes arbitrary
# phrasing onto these CLOSED labels; planner.py maps labels to the
# declared values in people.yaml. The LLM never emits numbers.
# Imported by planner.py for post-hoc validation (fallback path).
# ------------------------------------------------------------------
INTENT_MANNERS = ["gentle", "brisk", "well", "none"]
INTENT_ADJUSTMENTS = ["down_strong", "down", "down_slight",
                      "up_slight", "up", "up_strong", "none"]


def intent_schema(people_names):
    """The intent header: closed labels only. 'recipient' is enum-bound to
    the people.yaml roster (+ 'none') - the LLM can route any phrasing
    onto a known person but can never invent a name."""
    return {
        "type": "object",
        "properties": {
            "manner": {"type": "string", "enum": INTENT_MANNERS},
            "adjustment": {"type": "string", "enum": INTENT_ADJUSTMENTS},
            "store": {"type": "boolean"},
            "recipient": {"type": "string",
                          "enum": sorted(people_names) + ["none"]}
        },
        "required": ["manner", "adjustment", "store", "recipient"],
        "additionalProperties": False
    }


def detected_object_ids(scene):
    """Objects the camera can find (scene.yaml entries with `detect`): the
    only valid targets of pick / pick_and_place."""
    return [k for k, v in (scene.get("poses") or {}).items()
            if isinstance(v, dict) and "detect" in v]


def resolvable_pose_refs(scene_path=paths.SCENE_YAML):
    """Every pose reference string that executor._resolve() can turn into a
    6-vector: any scene path ending at a node that carries 'value', 'ref'
    (aliases resolve through their ref chain), or 'derived_from' +
    'offset_z_mm' (computed approach poses, generated at resolve time).
    A trailing '.value' on a direct-value node also resolves, so it is
    included as well."""
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = yaml.safe_load(f)

    refs = set()

    def walk(node, path):
        if isinstance(node, dict):
            has_value = isinstance(node.get("value"), list)
            is_computed = "derived_from" in node and "offset_z_mm" in node
            if has_value or "ref" in node or is_computed:
                refs.add(".".join(path))
                if has_value:
                    refs.add(".".join(path + ["value"]))
            for k, v in node.items():
                if isinstance(v, dict):
                    walk(v, path + [k])

    walk(scene.get("poses", {}), [])
    return sorted(refs)


def build_schema(registry_path=paths.REGISTRY_YAML, scene_path=paths.SCENE_YAML,
                 people_path=paths.PEOPLE_YAML):
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = yaml.safe_load(f)
    people_names = []
    try:
        with open(people_path, "r", encoding="utf-8") as f:
            people_names = list((yaml.safe_load(f) or {}).get("people", {}))
    except FileNotFoundError:
        pass

    valid_objects = list(scene.get("poses", {}).keys())
    detected_objects = detected_object_ids(scene)
    valid_pose_refs = resolvable_pose_refs(scene_path)

    action_schemas = []
    for p in registry.get("primitives", []):
        if p.get("type") != "composite":
            continue

        action_id = p["id"]
        props = {"action": {"type": "string", "const": action_id}}
        required = ["action"]

        for param in p.get("parameters", []):
            pname = param["name"]
            ptype = param.get("type", "string")

            if ptype == "detected_object_reference":
                props[pname] = {"type": "string", "enum": detected_objects}
            elif ptype == "object_reference":
                props[pname] = {"type": "string", "enum": valid_objects}
            elif ptype == "pose_reference":
                props[pname] = {"type": "string", "enum": valid_pose_refs}
            elif ptype == "integer":
                props[pname] = {"type": "integer"}
            elif ptype == "float":
                props[pname] = {"type": "number"}
            elif ptype == "enum":
                props[pname] = {"type": "string", "enum": param.get("options", [])}
            else:
                props[pname] = {"type": "string"}

            if "default" not in param:
                required.append(pname)

        action_schemas.append({
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": False
        })

    # Envelope: the LLM answers with an intent header (closed labels) plus
    # the plan array. Constrained decoding keeps the label alphabet closed.
    return {
        "type": "object",
        "properties": {
            "intent": intent_schema(people_names),
            "plan": {
                "type": "array",
                "items": {"oneOf": action_schemas}
            }
        },
        "required": ["intent", "plan"],
        "additionalProperties": False
    }


if __name__ == "__main__":
    import json
    print(json.dumps(build_schema(), indent=2))