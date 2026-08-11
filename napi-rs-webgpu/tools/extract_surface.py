#!/usr/bin/env python3
"""Derive the WebGPU surface `napi-rs-webgpu` must provide, from the bindings it replaces.

The vendored `webgpu_sys` files are generated from WebGPU's IDL, so they already
carry every fact the new crate needs: the JavaScript class name, each member's
JavaScript name, whether it is a getter, a setter, a method or a constructor,
whether it can throw, and the Rust signature web-sys chose for it. Reading them is
therefore more reliable than hand-typing 400 members, and it can be re-run when the
bindings are re-vendored.

`backend/webgpu.rs` is consulted too, to mark which members are actually reached.
476 of the generated methods are never called, and the new crate has no reason to
carry them.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ATTR = re.compile(r"#\s*\[\s*wasm_bindgen\s*\((?P<args>[^)]*)\)\s*\]", re.S)
TYPE_DECL = re.compile(r"pub type (?P<name>\w+)\s*;")
FN_DECL = re.compile(
    r"pub fn (?P<name>\w+)\s*\((?P<args>[^;]*?)\)\s*(?:->\s*(?P<ret>[^;]+?))?\s*;", re.S
)
ENUM_DECL = re.compile(r"pub enum (?P<name>\w+)\s*\{(?P<body>[^}]*)\}", re.S)
EXTERN_BLOCK = re.compile(r"extern\s*.C.\s*\{", re.S)
IMPL_FN = re.compile(
    r"(?P<deprecated>#\s*\[deprecated[^\]]*\]\s*)?"
    r"pub fn (?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*(?:->\s*(?P<ret>[^{;]+?))?\s*\{",
    re.S,
)
ENUM_VARIANT = re.compile(r"(?P<name>\w+)\s*=\s*\"(?P<js>[^\"]*)\"")


def matching_brace(text: str, open_index: int) -> int:
    """Index just past the `}` closing the brace at `open_index`."""
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(text)


def impl_owner(text: str, position: int) -> str | None:
    """The type of the nearest enclosing `impl` before `position`."""
    matches = list(re.finditer(r"impl\s+(?P<name>\w+)\s*\{", text[:position]))
    return matches[-1].group("name") if matches else None


def parse_args(args: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    depth = 0
    current = ""
    for char in args:
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        if char == "," and depth == 0:
            out.append(current)
            current = ""
        else:
            current += char
    out.append(current)
    parsed = []
    for arg in out:
        arg = " ".join(arg.split())
        if not arg or ":" not in arg:
            continue
        name, _, ty = arg.partition(":")
        parsed.append({"name": name.strip(), "type": ty.strip()})
    return parsed


def attr_keys(args: str) -> dict[str, object]:
    """`method , getter , js_class = "GPUBuffer" , js_name = size` → a dict."""
    keys: dict[str, object] = {"extends": []}
    for entry in re.split(r",(?![^<]*>)", args):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            key, _, value = entry.partition("=")
            key, value = key.strip(), value.strip().strip('"')
            if key == "extends":
                keys["extends"].append(value.replace(" ", ""))
            else:
                keys[key] = value
        else:
            keys[entry] = True
    return keys


def member_kind(keys: dict[str, object]) -> str:
    if "constructor" in keys:
        return "constructor"
    if "getter" in keys:
        return "getter"
    if "setter" in keys:
        return "setter"
    if "method" in keys:
        return "method"
    return "static"


def js_member_name(keys: dict[str, object], rust_name: str, kind: str) -> str:
    for key in ("getter", "setter"):
        if isinstance(keys.get(key), str):
            return keys[key]
    if isinstance(keys.get("js_name"), str):
        return keys["js_name"]
    if kind in ("getter", "setter"):
        return re.sub(r"^(get|set)_", "", rust_name)
    return rust_name


def main(gen_dir: str, backend_path: str, out_path: str) -> int:
    gen = Path(gen_dir)
    backend = Path(backend_path).read_text()
    interfaces: dict[str, dict] = {}
    enums: dict[str, dict] = {}

    for path in sorted(gen.glob("gen_*.rs")):
        text = path.read_text()
        # Strip doc comments so the declaration regexes see clean text.
        text = re.sub(r"#\s*\[doc\s*=\s*\"(?:[^\"\\]|\\.)*\"\s*\]", "", text)

        for match in ENUM_DECL.finditer(text):
            variants = [
                {"rust": v.group("name"), "js": v.group("js")}
                for v in ENUM_VARIANT.finditer(match.group("body"))
            ]
            if variants:
                enums[match.group("name")] = {"variants": variants}

        # Walk attribute/declaration pairs in order; an attribute applies to the
        # next declaration after it.
        events = []
        for m in ATTR.finditer(text):
            events.append((m.start(), "attr", attr_keys(m.group("args"))))
        for m in TYPE_DECL.finditer(text):
            events.append((m.start(), "type", m.group("name")))
        extern_start = EXTERN_BLOCK.search(text)
        extern_span = (extern_start.end(), matching_brace(text, extern_start.end() - 1)) if extern_start else (0, 0)
        for m in FN_DECL.finditer(text):
            if not (extern_span[0] <= m.start() < extern_span[1]):
                continue
            events.append(
                (
                    m.start(),
                    "fn",
                    {
                        "name": m.group("name"),
                        "args": parse_args(m.group("args")),
                        "ret": " ".join((m.group("ret") or "()").split()),
                    },
                )
            )
        events.sort(key=lambda e: e[0])

        # Dictionary constructors and the deprecated chaining builders live in a
        # plain `impl`, outside the extern block.
        for m in IMPL_FN.finditer(text):
            if extern_span[0] <= m.start() < extern_span[1]:
                continue
            owner = impl_owner(text, m.start())
            if owner is None:
                continue
            name = m.group("name")
            interfaces.setdefault(
                owner, {"js_class": owner, "extends": [], "members": []}
            )["members"].append(
                {
                    "rust": name,
                    "js": name,
                    "kind": "builder" if m.group("deprecated") else "constructor",
                    "catch": False,
                    "args": parse_args(m.group("args")),
                    "ret": " ".join((m.group("ret") or "()").split()),
                    "used": bool(re.search(rf"{re.escape(owner)}::{re.escape(name)}\s*\(", backend)),
                }
            )

        pending: dict[str, object] = {}
        current_type: str | None = None
        for _, kind, payload in events:
            if kind == "attr":
                pending = payload
            elif kind == "type":
                current_type = payload
                entry = interfaces.setdefault(
                    current_type, {"js_class": current_type, "extends": [], "members": []}
                )
                # The declaration carries the authoritative class name and
                # inheritance; the impl pass may have created the entry first with
                # placeholders.
                entry["js_class"] = pending.get("js_name", current_type)
                entry["extends"] = pending.get("extends", [])
                pending = {}
            elif kind == "fn":
                if current_type is None:
                    pending = {}
                    continue
                kind_name = member_kind(pending)
                receiver = pending.get("js_class")
                owner = current_type
                # A declaration's receiver is its first `this:` parameter.
                args = payload["args"]
                if args and args[0]["name"] == "this":
                    owner = args[0]["type"].lstrip("&")
                    args = args[1:]
                interfaces.setdefault(
                    owner,
                    {"js_class": receiver or owner, "extends": [], "members": []},
                )["members"].append(
                    {
                        "rust": payload["name"],
                        "js": js_member_name(pending, payload["name"], kind_name),
                        "kind": kind_name,
                        "catch": bool(pending.get("catch")),
                        "args": args,
                        "ret": payload["ret"],
                        "used": bool(
                            re.search(rf"\.{re.escape(payload['name'])}\s*\(", backend)
                            or re.search(
                                rf"{re.escape(owner)}::{re.escape(payload['name'])}\s*\(",
                                backend,
                            )
                        ),
                    }
                )
                pending = {}

    for name, spec in interfaces.items():
        spec["used"] = any(m["used"] for m in spec["members"]) or bool(
            re.search(rf"webgpu_sys::{re.escape(name)}\b", backend)
        )
    for name, spec in enums.items():
        spec["used"] = bool(re.search(rf"webgpu_sys::{re.escape(name)}\b", backend))

    spec = {"interfaces": interfaces, "enums": enums}
    Path(out_path).write_text(json.dumps(spec, indent=1))

    used_interfaces = {k: v for k, v in interfaces.items() if v["used"]}
    used_members = sum(
        1 for v in interfaces.values() for m in v["members"] if m["used"]
    )
    used_enums = {k: v for k, v in enums.items() if v["used"]}
    print(f"interfaces      : {len(interfaces):4}  ({len(used_interfaces)} reached)")
    print(
        f"members         : {sum(len(v['members']) for v in interfaces.values()):4}"
        f"  ({used_members} reached)"
    )
    print(f"string enums    : {len(enums):4}  ({len(used_enums)} reached)")
    print(
        f"enum variants   : {sum(len(v['variants']) for v in used_enums.values()):4}"
        "  (in reached enums)"
    )
    kinds: dict[str, int] = {}
    for v in interfaces.values():
        for m in v["members"]:
            if m["used"]:
                kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    print(f"reached by kind : {kinds}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
