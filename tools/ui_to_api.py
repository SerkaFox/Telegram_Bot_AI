#!/usr/bin/env python3
"""Convert a ComfyUI UI-format workflow (with embedded subgraphs) into a flat API-format
prompt dict, using a running ComfyUI server's /object_info to resolve widget ordering.
"""
import json
import sys
import urllib.parse
import urllib.request

COMFY_BASE = "http://127.0.0.1:8188"

UI_ONLY_TYPES = {
    "MarkdownNote",
    "Note",
    "Reroute (rgthree)",
    "Label (rgthree)",
    "Fast Groups Bypasser (rgthree)",
    "Fast Groups Muter (rgthree)",
}

_object_info_cache: dict = {}


def get_object_info(class_type: str) -> dict:
    if class_type not in _object_info_cache:
        url = f"{COMFY_BASE}/object_info/" + urllib.parse.quote(class_type, safe="")
        with urllib.request.urlopen(url) as resp:
            data = json.load(resp)
        info = data.get(class_type)
        if not info:
            raise RuntimeError(f"Unknown node class in object_info: {class_type}")
        _object_info_cache[class_type] = info
    return _object_info_cache[class_type]


# comfy-core's ResizeImageMaskNode shows extra conditional ("dotted") widgets in the UI
# depending on the selected resize_type, which /object_info does not expose at all.
# Ground-truthed against a known-good API export of the same node.
_RESIZE_IMAGE_MASK_NODE_ORDER = {
    3: ["input", "resize_type", "resize_type.multiplier", "scale_method"],
    4: ["input", "resize_type", "resize_type.width", "resize_type.height", "scale_method"],
}


def input_order(node: dict) -> list[str]:
    if node["type"] == "ResizeImageMaskNode":
        wv = node.get("widgets_values") or []
        order = _RESIZE_IMAGE_MASK_NODE_ORDER.get(len(wv))
        if order:
            return order
    info = get_object_info(node["type"])
    req = list((info["input"].get("required") or {}).keys())
    opt = list((info["input"].get("optional") or {}).keys())
    return req + opt


def is_disabled(node: dict) -> bool:
    return node.get("mode", 0) != 0


def flatten_subgraph(subgraph_def: dict, prefix: str, resolved_outer_inputs: list):
    """Expand a single (non-nested) subgraph instance into flat raw nodes (UI-shaped)."""
    in_node_id = subgraph_def["inputNode"]["id"]
    out_node_id = subgraph_def["outputNode"]["id"]

    raw_nodes = {n["id"]: n for n in subgraph_def["nodes"]}
    disabled = {nid for nid, n in raw_nodes.items() if is_disabled(n)}
    resolved_inputs = {nid: {} for nid in raw_nodes}  # nid -> slot_idx -> origin
    output_resolution = {}

    for link in subgraph_def["links"]:
        origin_id = link["origin_id"]
        origin_slot = link["origin_slot"]
        target_id = link["target_id"]
        target_slot = link["target_slot"]

        if origin_id != in_node_id and origin_id in disabled:
            continue  # disabled (muted/bypassed) source: treat as unconnected

        if origin_id == in_node_id:
            origin = resolved_outer_inputs[origin_slot]
        else:
            origin = (f"{prefix}:{origin_id}", origin_slot)

        if target_id == out_node_id:
            output_resolution[target_slot] = origin
            continue
        if target_id in disabled:
            continue

        resolved_inputs[target_id][target_slot] = origin

    flat = {}
    for nid, n in raw_nodes.items():
        if nid in disabled:
            continue
        fid = f"{prefix}:{nid}"
        named = {}
        skip = set()
        for slot_idx, inp in enumerate(n.get("inputs") or []):
            if slot_idx in resolved_inputs[nid]:
                named[inp["name"]] = {"origin": resolved_inputs[nid][slot_idx], "has_widget": "widget" in inp}
            elif "widget" not in inp:
                skip.add(inp["name"])
        flat[fid] = {
            "type": n["type"],
            "node": n,
            "widgets_values": n.get("widgets_values") or [],
            "named_inputs": named,
            "skip_names": skip,
        }

    return flat, output_resolution


def convert(ui_workflow: dict) -> dict:
    nodes_by_id = {n["id"]: n for n in ui_workflow["nodes"]}
    links_by_id = {l[0]: l for l in ui_workflow["links"]}
    subgraphs_by_id = {sg["id"]: sg for sg in ui_workflow.get("definitions", {}).get("subgraphs", [])}

    get_node_key = {n["id"]: n["widgets_values"][0] for n in ui_workflow["nodes"] if n["type"] == "GetNode"}
    set_node_key = {n["id"]: n["widgets_values"][0] for n in ui_workflow["nodes"] if n["type"] == "SetNode"}

    subgraph_output_map: dict = {}
    set_origin: dict = {}
    bypass_origin: dict = {}
    bypass_resolved: set = set()
    flat: dict = {}

    def resolve_origin(node_id, slot):
        if node_id in subgraph_output_map:
            return subgraph_output_map[node_id][slot]
        if node_id in get_node_key:
            return set_origin[get_node_key[node_id]]
        if node_id in bypass_origin:
            return bypass_origin[node_id]
        return (str(node_id), slot)

    def is_blocked(origin_id):
        n = nodes_by_id.get(origin_id)
        if n is None:
            return False
        if n["type"] in subgraphs_by_id and origin_id not in subgraph_output_map:
            return True
        if n["type"] == "GetNode" and get_node_key[origin_id] not in set_origin:
            return True
        if is_disabled(n) and origin_id not in bypass_resolved:
            return True
        return False

    def origin_absent(origin_id):
        """True if this link source is a disabled node with no bypass passthrough."""
        n = nodes_by_id.get(origin_id)
        return n is not None and is_disabled(n) and origin_id not in bypass_origin

    remaining = list(ui_workflow["nodes"])
    safety = 0
    while remaining:
        safety += 1
        if safety > 20:
            raise RuntimeError(f"Could not resolve subgraph dependency order: {[n['id'] for n in remaining]}")
        progressed = False
        next_remaining = []
        for n in remaining:
            nid = n["id"]
            ntype = n["type"]
            if ntype in UI_ONLY_TYPES or ntype == "GetNode":
                progressed = True
                continue

            blocked = False
            for inp in n.get("inputs") or []:
                link_id = inp.get("link")
                if link_id is None:
                    continue
                origin_id = links_by_id[link_id][1]
                if is_blocked(origin_id):
                    blocked = True
                    break
            if blocked:
                next_remaining.append(n)
                continue

            if is_disabled(n):
                # bypassed node: ComfyUI reroutes a matching-type input straight to the
                # output, so find the connected input whose type matches an output type;
                # if exactly one matches, treat it as a transparent passthrough. Loaders
                # and other zero/ambiguous-input nodes are dead ends and stay unconnected.
                out_types = {o.get("type") for o in (n.get("outputs") or [])}
                candidates = [
                    inp for inp in (n.get("inputs") or [])
                    if inp.get("link") is not None and inp.get("type") in out_types
                ]
                if len(candidates) == 1:
                    l = links_by_id[candidates[0]["link"]]
                    bypass_origin[nid] = resolve_origin(l[1], l[2])
                bypass_resolved.add(nid)
                progressed = True
                continue

            if ntype == "SetNode":
                inputs = n.get("inputs") or []
                link_id = inputs[0]["link"] if inputs else None
                if link_id is not None:
                    l = links_by_id[link_id]
                    if not origin_absent(l[1]):
                        set_origin[set_node_key[nid]] = resolve_origin(l[1], l[2])
                progressed = True
                continue

            if ntype in subgraphs_by_id:
                ext_inputs = []
                for inp in n.get("inputs") or []:
                    link_id = inp.get("link")
                    if link_id is None:
                        ext_inputs.append(("LITERAL", None))
                        continue
                    l = links_by_id[link_id]
                    if origin_absent(l[1]):
                        ext_inputs.append(("LITERAL", None))
                        continue
                    ext_inputs.append(resolve_origin(l[1], l[2]))
                sg_def = subgraphs_by_id[ntype]
                flat_nodes, out_map = flatten_subgraph(sg_def, str(nid), ext_inputs)
                flat.update(flat_nodes)
                subgraph_output_map[nid] = out_map
            else:
                named = {}
                skip = set()
                for inp in n.get("inputs") or []:
                    link_id = inp.get("link")
                    if link_id is None or origin_absent(links_by_id[link_id][1]):
                        if "widget" not in inp:
                            skip.add(inp["name"])
                        continue
                    l = links_by_id[link_id]
                    named[inp["name"]] = {"origin": resolve_origin(l[1], l[2]), "has_widget": "widget" in inp}
                flat[str(nid)] = {
                    "type": ntype,
                    "node": n,
                    "widgets_values": n.get("widgets_values") or [],
                    "named_inputs": named,
                    "skip_names": skip,
                }
            progressed = True
        remaining = next_remaining
        if not progressed and remaining:
            raise RuntimeError(f"Stuck resolving nodes: {[n['id'] for n in remaining]}")

    primitive_values = {}
    for nid, node in flat.items():
        if node["type"] == "PrimitiveNode":
            wv = node["widgets_values"]
            primitive_values[nid] = wv[0] if wv else None

    def origin_value(origin):
        node_id, slot = origin
        if node_id in primitive_values:
            return primitive_values[node_id]
        return [node_id, slot]

    api: dict = {}
    for nid, node in flat.items():
        if node["type"] == "PrimitiveNode":
            continue
        order = input_order(node["node"])
        named_inputs = node["named_inputs"]
        skip_names = node.get("skip_names") or set()
        raw_wv = node["widgets_values"]
        api_inputs = {}

        if isinstance(raw_wv, dict):
            # some nodes (e.g. VHS_VideoCombine) serialize widgets_values as a name->value
            # dict instead of a positional list.
            for name in order:
                if name in skip_names:
                    continue
                entry = named_inputs.get(name)
                if entry is not None:
                    api_inputs[name] = origin_value(entry["origin"])
                    continue
                if name in raw_wv:
                    api_inputs[name] = raw_wv[name]
            api[nid] = {"class_type": node["type"], "inputs": api_inputs}
            continue

        widgets_values = list(raw_wv)
        wv_idx = 0
        for name in order:
            if name in skip_names:
                continue
            entry = named_inputs.get(name)
            if entry is not None:
                api_inputs[name] = origin_value(entry["origin"])
                if entry["has_widget"]:
                    if wv_idx < len(widgets_values):
                        wv_idx += 1
                    if name in ("seed", "noise_seed") and wv_idx < len(widgets_values) and isinstance(widgets_values[wv_idx], str):
                        wv_idx += 1
                continue
            # not connected: either a pure widget (consume value) or an unconnected optional socket (skip)
            if wv_idx >= len(widgets_values):
                continue
            value = widgets_values[wv_idx]
            wv_idx += 1
            api_inputs[name] = value
            if name in ("seed", "noise_seed") and wv_idx < len(widgets_values) and isinstance(widgets_values[wv_idx], str):
                wv_idx += 1

        # dynamic nodes (e.g. rgthree Context Switch/Merge) declare inputs at runtime and
        # are absent from /object_info's static schema; include any linked input missed above.
        for name, entry in named_inputs.items():
            if name in api_inputs or name in skip_names:
                continue
            api_inputs[name] = origin_value(entry["origin"])

        api[nid] = {"class_type": node["type"], "inputs": api_inputs}

    return api


if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2]
    with open(src, "r", encoding="utf-8") as f:
        wf = json.load(f)
    api = convert(wf)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(api, f, indent=2, ensure_ascii=False)
    print(f"Wrote {dst} ({len(api)} nodes)")
