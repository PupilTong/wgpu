#!/usr/bin/env python3
"""Mark in `tools/surface.json` exactly what `wgpu` reaches, by asking the compiler.

`tools/extract_surface.py` derives the whole WebGPU surface from the IDL-generated
bindings and guesses which parts are reached by grepping `wgpu`'s backend for
`.member(`. The guess over-includes: a method name matches wherever it appears,
whatever the receiver, so `width`, `size`, `label` and their like are marked
reached on every type that declares them.

This asks a question the compiler can answer exactly. It empties the surface, has
`tools/generate.py` emit that, builds `wgpu` against it, and reads the errors: a
member `wgpu` needs is a `no method named X found for struct Y`, a type it needs
is a `cannot find type X`. Those go back into the spec and it builds again, until
the build is clean. What is left marked `used` is then a *proved* minimal set —
removing any of it breaks a build, and adding to it cannot be justified.

    python3 tools/shake.py

Only `used` flags change; every declaration stays in the file, so a later `wgpu`
that reaches further is one re-run away from having its bindings back.

Two things are deliberately not shaken:

* enum variants. Dropping a variant `wgpu` never *writes* is invisible to the
  compiler and changes what happens at run time — a value coming back from
  JavaScript would decode to `__Invalid` instead of itself. Whole enums that
  nothing names are dropped; the variants of a kept enum are all kept.
* the crate's own entry points (`install`, `install_gpu`, `adopt_js_value` and
  the `futures` and `callback` modules). They are how an addon reaches the crate
  at all, and `wgpu` does not call them by construction.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

CRATE = Path(__file__).resolve().parent.parent
SPEC = CRATE / "tools" / "surface.json"
GENERATE = CRATE / "tools" / "generate.py"
WGPU = CRATE.parent

# The builds a member has to be reached by one of to count as reached. These are
# what CI checks for WASI, which is every configuration in which the Node-API
# bindings are selected at all.
# The crate's own build comes first and short-circuits the rest: while a member
# it needs is still missing, every downstream build fails the same way, and
# building `wgpu` five times to learn that again is wasted.
BUILDS = [
    ["-p", "napi-rs-webgpu", "--target", "wasm32-wasip1"],
] + [
    ["-p", "wgpu", "--target", t, "--no-default-features", "--features", f]
    for t in ("wasm32-wasip1", "wasm32-wasip1-threads")
    for f in ("webgpu,wgsl", "webgpu,wgsl,std")
] + [
    ["-p", "wgpu-types", "--target", "wasm32-wasip1", "--no-default-features", "--features", "web"],
]

# rustc names what is missing in a handful of shapes. Each of these yields either
# a `(type, member)` pair or a bare type name.
# The owner is captured whole, generics and all: `wgpu` reaches most members
# through `DefinedNonNullJsValue<Gpu>`, whose `Deref` target is what actually
# declares them, so the name that matters is inside the angle brackets. Which of
# the names in the expression is a real interface is decided in `resolve_owner`.
MEMBER_ERRORS = [
    re.compile(r"no method named `(?P<member>\w+)` found for \w+ `(?P<owner>[^`]+)`"),
    re.compile(r"no function or associated item named `(?P<member>\w+)` found for \w+ `(?P<owner>[^`]+)`"),
    re.compile(r"no associated item named `(?P<member>\w+)` found for \w+ `(?P<owner>[^`]+)`"),
]
TYPE_ERRORS = [
    re.compile(r"cannot find (?:type|struct, variant or union type|value|function) `(?P<name>\w+)` in "),
    re.compile(r"failed to resolve: use of undeclared (?:type|crate or module) `(?P<name>\w+)`"),
    re.compile(r"could not find `(?P<name>\w+)` in `\w+`"),
    re.compile(r"unresolved import `[\w:]*::(?P<name>\w+)`"),
    re.compile(r"no `(?P<name>\w+)` in (?:the root|`[\w:]+`)"),
]

MAX_ROUNDS = 40


def run(command: list[str], cwd: Path) -> tuple[int, str]:
    finished = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    return finished.returncode, finished.stdout + finished.stderr


def generate() -> None:
    code, output = run([sys.executable, str(GENERATE)], CRATE)
    if code != 0:
        raise SystemExit(f"tools/generate.py failed:\n{output}")


def compile_all() -> str:
    """Every build's diagnostics, concatenated. Empty when they all pass."""
    diagnostics = []
    for index, build in enumerate(BUILDS):
        code, output = run(["cargo", "check", *build, "--message-format", "short"], WGPU)
        if code != 0:
            diagnostics.append(output)
            if index == 0:
                # The bindings crate itself does not build; everything after
                # would only repeat its errors.
                break
    return "\n".join(diagnostics)


def close_over_signatures(interfaces: dict, enums: dict) -> int:
    """Enable every type an already-enabled declaration mentions.

    A member cannot be emitted without the types in its signature, and an
    interface cannot be emitted without its ancestors. The compiler would say so
    eventually — as an unresolved import — but deriving it here saves a round per
    level of nesting and keeps the spec self-consistent between them.
    """
    added = 0
    for entry in interfaces.values():
        if not entry["used"]:
            continue
        mentioned = list(entry.get("extends", []))
        for member in entry["members"]:
            if member["used"]:
                mentioned += [argument["type"] for argument in member["args"]]
                mentioned.append(member["ret"])
        for text in mentioned:
            for name in re.findall(r"\w+", text):
                for table in (interfaces, enums):
                    if name in table and not table[name]["used"]:
                        table[name]["used"] = True
                        added += 1
    return added


def ancestry(name: str, interfaces: dict) -> list[dict]:
    """`name`'s entry, then its parents' — the `Deref` chain, in order."""
    chain, seen = [], set()
    queue = [name]
    while queue:
        current = queue.pop(0)
        if current in seen or current not in interfaces:
            continue
        seen.add(current)
        chain.append(interfaces[current])
        for parent in interfaces[current].get("extends", []):
            queue += re.findall(r"\w+", parent)
    return chain


def resolve_owner(expression: str, interfaces: dict) -> str | None:
    """The interface a `no method named ...` error is really about.

    `&DefinedNonNullJsValue<Gpu>` is `Gpu`'s error: the wrapper derefs to it and
    declares nothing itself. The innermost name that the spec knows is the answer,
    so the names are tried right to left.
    """
    for name in reversed(re.findall(r"\w+", expression)):
        if name in interfaces:
            return name
    return None


def wanted(diagnostics: str) -> tuple[set[tuple[str, str]], set[str]]:
    members: set[tuple[str, str]] = set()
    types: set[str] = set()
    for line in diagnostics.splitlines():
        for pattern in MEMBER_ERRORS:
            found = pattern.search(line)
            if found:
                members.add((found.group("owner"), found.group("member")))
        for pattern in TYPE_ERRORS:
            found = pattern.search(line)
            if found:
                types.add(found.group("name"))
    return members, types


def main() -> int:
    spec = json.loads(SPEC.read_text())
    interfaces, enums = spec["interfaces"], spec["enums"]
    namespaces = spec.get("namespaces", {})

    before = (
        sum(1 for v in interfaces.values() if v["used"]),
        sum(1 for v in interfaces.values() for m in v["members"] if m["used"]),
        sum(1 for v in enums.values() if v["used"]),
    )

    # Empty it. The `builder` members are web-sys' deprecated chaining setters,
    # which `tools/generate.py` never emits; leaving them alone keeps the diff to
    # what this tool is actually deciding.
    for entry in interfaces.values():
        entry["used"] = False
        for member in entry["members"]:
            if member["kind"] != "builder":
                member["used"] = False
    for entry in enums.values():
        entry["used"] = False
    for entry in namespaces.values():
        entry["used"] = False
        for constant in entry["constants"]:
            constant["used"] = False

    for round_number in range(1, MAX_ROUNDS + 1):
        SPEC.write_text(json.dumps(spec, indent=2) + "\n")
        generate()
        diagnostics = compile_all()
        if not diagnostics:
            print(f"converged after {round_number - 1} rounds")
            break

        members, types = wanted(diagnostics)
        added = 0

        for owner_expression, member_name in members:
            owner = resolve_owner(owner_expression, interfaces)
            if owner is None:
                continue
            interfaces[owner]["used"] = True
            # The receiver rustc names is not always the type that declares the
            # member: `GpuValidationError` reaches `message` through its `Deref`
            # to `GpuError`. Walk up until something declares it.
            for entry in ancestry(owner, interfaces):
                found = False
                for member in entry["members"]:
                    if member["rust"] == member_name:
                        found = True
                        if not member["used"]:
                            member["used"] = True
                            entry["used"] = True
                            added += 1
                if found:
                    break

        for name in types:
            if name in interfaces and not interfaces[name]["used"]:
                interfaces[name]["used"] = True
                added += 1
            elif name in enums and not enums[name]["used"]:
                enums[name]["used"] = True
                added += 1
            if name in namespaces and not namespaces[name]["used"]:
                # The module itself is missing, so its constants cannot have been
                # named yet; a later round asks for the ones `wgpu` reads.
                namespaces[name]["used"] = True
                added += 1
            for entry in namespaces.values():
                for constant in entry["constants"]:
                    if constant["name"] == name and not constant["used"]:
                        constant["used"] = True
                        entry["used"] = True
                        added += 1

        # Adding a declaration can pull in the types it names, and those can
        # pull in more; settle that before building again.
        while True:
            more = close_over_signatures(interfaces, enums)
            if not more:
                break
            added += more

        print(f"round {round_number}: +{added}")
        if added == 0:
            print("\nnothing left to add, but the build is still failing:\n")
            print("\n".join(diagnostics.splitlines()[:40]))
            return 1
    else:
        print(f"gave up after {MAX_ROUNDS} rounds")
        return 1

    after = (
        sum(1 for v in interfaces.values() if v["used"]),
        sum(1 for v in interfaces.values() for m in v["members"] if m["used"]),
        sum(1 for v in enums.values() if v["used"]),
    )
    for label, was, now in zip(("interfaces", "members", "enums"), before, after):
        print(f"{label:11}: {was:4} -> {now:4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
