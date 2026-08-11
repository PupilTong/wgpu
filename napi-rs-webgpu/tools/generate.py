#!/usr/bin/env python3
"""Emit `src/webgpu/generated/` from `tools/surface.json`.

`tools/extract_surface.py` derives what WebGPU surface this crate must provide;
this script turns that description into the declarations, written in the member
DSL `src/dsl.rs` defines. Nothing here decides *what* is bound — every name, every
kind and every signature comes from the spec file, and only entries the spec marks
`used` are emitted, because the surface was cut down to what `wgpu`'s backend
actually reaches.

Run from the crate root:

    python3 tools/generate.py

Three things are policy rather than transcription, and each is a table below:
`MODULES` says which file a type lands in, `TYPES` maps the spec's Rust signatures
(which are web-sys' — they name `js_sys` types this crate does not have) onto this
crate's own types, and `REPAIRS` fixes the two members the extractor's regex
truncates.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CRATE = Path(__file__).resolve().parent.parent
SPEC = CRATE / "tools" / "surface.json"
OUT = CRATE / "src" / "webgpu" / "generated"

# --------------------------------------------------------------------------------
# Policy: which file each type lands in.
#
# The spec has no notion of grouping, so this is the one place the output's shape
# is decided. Interfaces sit with the interfaces they are reached from;
# dictionaries sit with the dictionaries they are built alongside, which is why
# `GpuQuerySetDescriptor` is in `query` and not in `descriptors`. Anything the spec
# grows that is not named here lands in `FALLBACK_MODULE` and is reported by the
# summary, so a re-derived surface is noticed rather than silently misfiled.
# --------------------------------------------------------------------------------

MODULES: dict[str, tuple[str, list[str]]] = {
    "adapter": (
        "The entry point: the `GPU` object, the adapter it hands out, and the two\n\
//! read-only sets describing what that adapter supports.",
        [
            "Gpu",
            "GpuAdapter",
            "GpuAdapterInfo",
            "GpuSupportedFeatures",
            "GpuSupportedLimits",
            "WgslLanguageFeatures",
        ],
    ),
    "device": (
        "The device, its queue, its canvas context, and the errors it reports.",
        [
            "GpuCanvasContext",
            "GpuDevice",
            "GpuDeviceLostInfo",
            "GpuError",
            "GpuInternalError",
            "GpuOutOfMemoryError",
            "GpuQueue",
            "GpuUncapturedErrorEvent",
            "GpuValidationError",
        ],
    ),
    "resources": (
        "The objects a device allocates: buffers, textures, samplers and the bind\n\
//! groups that name them.",
        [
            "GpuBindGroup",
            "GpuBindGroupLayout",
            "GpuBuffer",
            "GpuExternalTexture",
            "GpuSampler",
            "GpuTexture",
            "GpuTextureView",
        ],
    ),
    "pipeline": (
        "Shader modules, their compilation diagnostics, and the pipelines built from\n\
//! them.",
        [
            "GpuCompilationInfo",
            "GpuCompilationMessage",
            "GpuComputePipeline",
            "GpuPipelineLayout",
            "GpuRenderPipeline",
            "GpuShaderModule",
        ],
    ),
    "encoder": (
        "Command recording: the encoder, the two pass encoders, render bundles, and\n\
//! the command buffer they produce.",
        [
            "GpuCommandBuffer",
            "GpuCommandEncoder",
            "GpuComputePassEncoder",
            "GpuRenderBundle",
            "GpuRenderBundleEncoder",
            "GpuRenderPassEncoder",
        ],
    ),
    "query": (
        "Query sets and the timestamp writes that fill them.",
        [
            "GpuComputePassTimestampWrites",
            "GpuQuerySet",
            "GpuQuerySetDescriptor",
            "GpuRenderPassTimestampWrites",
        ],
    ),
    "descriptors": (
        "The dictionaries that describe an object about to be created.",
        [
            "GpuBindGroupDescriptor",
            "GpuBindGroupLayoutDescriptor",
            "GpuBufferDescriptor",
            "GpuCanvasConfiguration",
            "GpuCanvasToneMapping",
            "GpuCommandBufferDescriptor",
            "GpuCommandEncoderDescriptor",
            "GpuComputePassDescriptor",
            "GpuComputePipelineDescriptor",
            "GpuDeviceDescriptor",
            "GpuExternalTextureDescriptor",
            "GpuObjectDescriptorBase",
            "GpuPipelineDescriptorBase",
            "GpuPipelineLayoutDescriptor",
            "GpuQueueDescriptor",
            "GpuRenderBundleDescriptor",
            "GpuRenderBundleEncoderDescriptor",
            "GpuRenderPassDescriptor",
            "GpuRenderPipelineDescriptor",
            "GpuRequestAdapterOptions",
            "GpuSamplerDescriptor",
            "GpuShaderModuleCompilationHint",
            "GpuShaderModuleDescriptor",
            "GpuTextureDescriptor",
            "GpuTextureViewDescriptor",
        ],
    ),
    "layouts": (
        "The dictionaries describing a binding: what a bind group holds, what a bind\n\
//! group layout permits, and how vertex buffers are read.",
        [
            "GpuBindGroupEntry",
            "GpuBindGroupLayoutEntry",
            "GpuBufferBinding",
            "GpuBufferBindingLayout",
            "GpuExternalTextureBindingLayout",
            "GpuSamplerBindingLayout",
            "GpuStorageTextureBindingLayout",
            "GpuTextureBindingLayout",
            "GpuVertexAttribute",
            "GpuVertexBufferLayout",
        ],
    ),
    "state": (
        "The fixed-function state a render pipeline is assembled from, and the\n\
//! attachments a render pass writes.",
        [
            "GpuBlendComponent",
            "GpuBlendState",
            "GpuColorTargetState",
            "GpuDepthStencilState",
            "GpuFragmentState",
            "GpuMultisampleState",
            "GpuPrimitiveState",
            "GpuProgrammableStage",
            "GpuRenderPassColorAttachment",
            "GpuRenderPassDepthStencilAttachment",
            "GpuStencilFaceState",
            "GpuVertexState",
        ],
    ),
    "geometry": (
        "Sizes, origins, colours, and the source and destination of a copy.",
        [
            "GpuColorDict",
            "GpuCopyExternalImageDestInfo",
            "GpuCopyExternalImageSourceInfo",
            "GpuExtent3dDict",
            "GpuOrigin2dDict",
            "GpuOrigin3dDict",
            "GpuTexelCopyBufferInfo",
            "GpuTexelCopyBufferLayout",
            "GpuTexelCopyTextureInfo",
        ],
    ),
}

# Where the string enums go. They are all in one file because a string enum has no
# dependencies and reads as a table.
ENUM_MODULE = "enums"
ENUM_MODULE_DOC = "WebGPU's string enums."

# Where the namespace constants go. A WebIDL namespace is a bag of constants with
# no object behind it, so it needs neither a handle nor a call.
NAMESPACE_MODULE = "namespaces"
NAMESPACE_MODULE_DOC = "WebGPU's namespaces, which are constants and nothing else."

# Where a type the tables do not name lands. Reported by the summary.
FALLBACK_MODULE = "misc"
FALLBACK_MODULE_DOC = (
    "Types `tools/generate.py` has no home for yet — see its `MODULES` table."
)

# --------------------------------------------------------------------------------
# Policy: the spec's types, which are web-sys', spelled as this crate's.
# --------------------------------------------------------------------------------

# Types this crate defines itself, reached as `crate::Name` because nothing in a
# generated file is imported from the crate root.
CRATE_TYPES = {
    "js_sys::ArrayBuffer": "crate::ArrayBuffer",
    "js_sys::Uint8Array": "crate::Uint8Array",
    "js_sys::Uint32Array": "crate::Uint32Array",
    "js_sys::Object": "crate::Object",
    "js_sys::JsValue": "crate::JsValue",
    "js_sys::JsString": "crate::JsString",
    "js_sys::Number": "crate::Number",
    "js_sys::Undefined": "crate::Undefined",
    "wasm_bindgen::JsValue": "crate::JsValue",
    "JsValue": "crate::JsValue",
    # A JavaScript function has no binding of its own: it is held as the value it
    # is and called through `crate::napi::rt::call`.
    "js_sys::Function": "crate::JsValue",
}

# Types whose generic parameter is part of the crate's spelling too: `Object<T>`
# names what a `record`'s values are, `JsOption<T>` and `Iterator<T>` what they
# hold. Everything else in `CRATE_TYPES` is one JavaScript value whatever web-sys
# writes inside the angle brackets, so its parameters are dropped.
GENERIC_CRATE_TYPES = {
    "js_sys::Object": "crate::Object",
    "js_sys::JsOption": "crate::JsOption",
    "js_sys::Iterator": "crate::JsIterator",
}

# `web_sys::X` in the spec, `crate::X` here: `src/dom.rs` declares them.
DOM_TYPES = {
    "Document",
    "Element",
    "Event",
    "EventTarget",
    "HtmlCanvasElement",
    "HtmlElement",
    "HtmlImageElement",
    "HtmlMediaElement",
    "HtmlVideoElement",
    "ImageBitmap",
    "ImageData",
    "MediaQueryList",
    "Navigator",
    "Node",
    "NodeList",
    "OffscreenCanvas",
    "VideoFrame",
    "Window",
    "WorkerGlobalScope",
    "WorkerNavigator",
}

# Rust's own types, which pass through unchanged.
PRIMITIVES = {
    "()",
    "Self",
    "bool",
    "f32",
    "f64",
    "i32",
    "str",
    "u16",
    "u32",
    "u8",
    "usize",
}

# --------------------------------------------------------------------------------
# The two members the extractor parses wrongly.
#
# `FN_DECL` in `tools/extract_surface.py` ends the argument list at the first `)`,
# which for an argument that is itself a function type is the `)` inside `fn(..)`.
# The declaration in `wgpu/src/backend/webgpu/webgpu_sys/gen_GpuSupportedFeatures.rs`
# reads
#
#     pub fn for_each(
#         this: &GpuSupportedFeatures,
#         callback: &::js_sys::Function<fn(::js_sys::JsString) -> ::js_sys::Undefined>,
#     ) -> Result<(), JsValue>;
#
# so the argument is one callback and the result is a throwing `()`. Repairing it
# here rather than in the extractor keeps `surface.json` the thing this script
# reads, unedited.
# --------------------------------------------------------------------------------

REPAIRS: dict[tuple[str, str], dict] = {
    (interface, "for_each"): {
        "args": [{"name": "callback", "type": "&js_sys::Function"}],
        "ret": "Result<(), JsValue>",
        "catch": True,
    }
    for interface in ("GpuSupportedFeatures", "WgslLanguageFeatures")
}

HEADER = """\
//! {doc}
//!
//! Generated by `tools/generate.py` from `tools/surface.json`. Do not edit by
//! hand: change the spec or the generator and re-run it.
"""

# rustfmt's `max_width` default, which the workspace's empty `rustfmt.toml` leaves
# in place. Import lists are laid out to it so `cargo fmt` finds nothing to change;
# member declarations are wrapped a little earlier, because rustfmt leaves a macro
# body it cannot parse — which this DSL's is — exactly as written.
RUSTFMT_WIDTH = 100
WIDTH = 96


# --------------------------------------------------------------------------------
# Reading the spec.
# --------------------------------------------------------------------------------


def normalize(text: str) -> str:
    """One line, no leading `::` on a path."""
    text = " ".join(text.split())
    return re.sub(r"(?<![A-Za-z0-9_]):{2}", "", text)


def split_top(text: str) -> list[str]:
    """Split on commas outside `<>`, `()` and `[]`."""
    parts: list[str] = []
    depth = 0
    current = ""
    for char in text:
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return parts


def parse_generic(text: str) -> tuple[str, list[str]]:
    """`Promise<Option<T>>` → `("Promise", ["Option<T>"])`."""
    open_index = text.find("<")
    if open_index < 0 or not text.endswith(">"):
        return text, []
    return text[:open_index], split_top(text[open_index + 1 : -1])


def is_dictionary(entry: dict) -> bool:
    """Whether a spec entry is a WebIDL dictionary rather than an interface.

    A dictionary is a plain object built field by field, and web-sys gives it a
    `new` returning `Self` plus a deprecated chaining setter per field. An interface
    with a JavaScript constructor — `GPUValidationError` is one — has a `new` too,
    but it can throw, so its `new` returns `Result`. That is the difference, and it
    is read out of the spec rather than guessed from the name.
    """
    return any(
        member["kind"] in ("constructor", "builder")
        and member["ret"] in ("Self", "&mut Self")
        for member in entry["members"]
    )


class Surface:
    """`surface.json`, with the entries the crate does not emit already dropped."""

    def __init__(self, spec: dict):
        self.all_interfaces = set(spec["interfaces"])
        self.all_enums = set(spec["enums"])
        self.enums = {
            name: entry for name, entry in spec["enums"].items() if entry["used"]
        }
        self.namespaces = {
            name: dict(
                entry,
                constants=[c for c in entry["constants"] if c["used"]],
            )
            for name, entry in spec.get("namespaces", {}).items()
            if entry["used"]
        }
        self.interfaces: dict[str, dict] = {}
        self.dictionaries: set[str] = set()
        for name, entry in spec["interfaces"].items():
            if not entry["used"]:
                continue
            if is_dictionary(entry):
                self.dictionaries.add(name)
            members = [
                self.repair(name, member)
                for member in entry["members"]
                # A `builder` is web-sys' deprecated chaining setter; the plain
                # setter beside it says the same thing and is what is called.
                if member["used"] and member["kind"] != "builder"
            ]
            # `accessors` keeps every getter and setter, reached or not: a
            # dictionary's JavaScript property spellings are read off them, and a
            # constructor may fill in a field whose own accessors nothing calls.
            accessors = [
                member
                for member in entry["members"]
                if member["kind"] in ("getter", "setter")
            ]
            self.interfaces[name] = dict(entry, members=members, accessors=accessors)

    @staticmethod
    def repair(interface: str, member: dict) -> dict:
        repair = REPAIRS.get((interface, member["rust"]))
        return dict(member, **repair) if repair else member


# --------------------------------------------------------------------------------
# The spec's Rust signatures, spelled as this crate's types.
# --------------------------------------------------------------------------------


class Types:
    """Maps one spec type to one crate type, recording what it referred to.

    The references are what the generated files' `use` lines are built from: a
    module imports exactly the generated types it names and no more.
    """

    def __init__(self, known: set[str]):
        self.known = known
        self.referenced: set[str] = set()

    def argument(self, text: str) -> str:
        """An argument's type, which keeps its `&`."""
        text = normalize(text)
        if text.startswith("&"):
            inner = text[1:].strip()
            head, args = parse_generic(inner)
            if head == "js_sys::Array":
                return f"&[{self.value(args[0])}]"
            if inner.startswith("[") and inner.endswith("]"):
                return f"&[{self.value(inner[1:-1])}]"
            return f"&{self.value(inner)}"
        head, args = parse_generic(text)
        if head == "js_sys::Array":
            return f"&[{self.value(args[0])}]"
        if head == "Option":
            # The reference is inside the `Option`, so recurse through `argument`.
            return f"Option<{self.argument(args[0])}>"
        return self.value(text)

    def value(self, text: str) -> str:
        """A type in a position that owns it: a return, or inside a container."""
        text = normalize(text)
        head, args = parse_generic(text)

        # A Rust `Option` in the spec is web-sys' optional *argument*, not WebIDL
        # nullability; `js_sys::JsOption` is the nullable JavaScript slot and is
        # handled with the other generic types below.
        if head == "Option":
            return f"Option<{self.argument(args[0])}>"
        if head == "js_sys::Promise":
            return f"crate::Promise<{self.value(args[0])}>" if args else "crate::Promise"
        # An array arrives as a whole sequence — `rt::array_items` collects one — so
        # it is a `Vec` on the Rust side. An iterator is not: it is walked a step at
        # a time, and keeps its JavaScript identity.
        if head == "js_sys::Array":
            return f"alloc::vec::Vec<{self.value(args[0])}>"
        if head in GENERIC_CRATE_TYPES and args:
            inner = ", ".join(self.value(argument) for argument in args)
            return f"{GENERIC_CRATE_TYPES[head]}<{inner}>"
        if head in ("alloc::string::String", "String"):
            return "alloc::string::String"
        if head in CRATE_TYPES:
            # The generic parameter, if any, is web-sys describing the contents; one
            # JavaScript object either way.
            return CRATE_TYPES[head]
        if head in DOM_TYPES:
            return f"crate::{head}"
        if head in PRIMITIVES:
            return head
        if head in self.known:
            self.referenced.add(head)
            return head
        raise SystemExit(f"tools/generate.py: no mapping for the type `{text}`")


# --------------------------------------------------------------------------------
# Emitting.
# --------------------------------------------------------------------------------


def ancestry(extends: list[str], types: Types) -> str:
    """The `js_type!` ancestry list: parent first, `crate::JsValue` last."""
    chain = [types.value(parent) for parent in extends]
    return "[" + ", ".join([*chain, "crate::JsValue"]) + "]"


def member_line(kind: str, name: str, arguments: list[str], tail: str) -> list[str]:
    """One DSL member, broken across lines when it does not fit on one.

    Where it breaks is cosmetic — a macro matches tokens, not layout. The wrapped
    form ends its argument list with a comma, which the DSL's `$(,)?` accepts and
    which is how the hand-written declarations in `src/webgpu/mod.rs` are laid out.
    """
    single = f"    {kind} {name}({', '.join(arguments)}){tail}"
    if len(single) <= WIDTH:
        return [single]
    if not arguments:
        # Nothing to wrap: the name alone is long, so the return and the JavaScript
        # name go on the next line.
        return [f"    {kind} {name}()", f"        {tail.strip()}"]
    lines = [f"    {kind} {name}("]
    lines += [f"        {argument}," for argument in arguments]
    if len(f"    ){tail}") <= WIDTH:
        lines.append(f"    ){tail}")
    else:
        lines += ["    )", f"        {tail.strip()}"]
    return lines


def member_doc(kind: str, member: dict, name: str, dictionary: bool) -> str:
    """The doc comment for one member, in the wording web-sys generates.

    web-sys' own text is the wording because it is what a reader porting from
    `web_sys` will recognise, and because deriving it from the spec keeps it
    honest: nothing here describes behaviour the declaration does not state.
    """
    js = member["js"]
    if kind == "constructor":
        return f"    /// Construct a new `{name}`."
    if kind == "getter":
        verb = "Get the" if dictionary else "Getter for the"
        return f"    /// {verb} `{js}` field of this object."
    if kind == "setter":
        verb = "Change the" if dictionary else "Setter for the"
        return f"    /// {verb} `{js}` field of this object."
    return f"    /// The `{js}()` method."


def derived_class(name: str) -> str:
    """The JavaScript class name the DSL assumes when a declaration omits it.

    `dsl::Operation::operation_name` restores the `Gpu` prefix to `GPU` and leaves
    the rest of the Rust name alone, so a declaration only has to state its class
    where the spec disagrees with that.
    """
    return f"GPU{name[3:]}" if name.startswith("Gpu") else name


def mdn_doc(js_class: str) -> list[str]:
    """The MDN link for an interface, as a reference when inline would be too long."""
    inline = f"    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/{js_class})"
    if len(inline) <= RUSTFMT_WIDTH:
        return [inline]
    return [
        "    /// [MDN Documentation][mdn]",
        "    ///",
        f"    /// [mdn]: https://developer.mozilla.org/en-US/docs/Web/API/{js_class}",
    ]


def sort_key(member: dict) -> tuple[int, str]:
    """Constructors first, then getters, setters and methods, each by name."""
    order = {"constructor": 0, "getter": 1, "setter": 2, "method": 3}
    return order.get(member["kind"], 4), member["rust"]


def property_names(entry: dict) -> dict[str, str]:
    """A dictionary's field name in Rust → its property name in JavaScript.

    A dictionary constructor takes the required fields as arguments named after the
    fields, but only the accessors carry the JavaScript spelling, so the mapping is
    read off them.
    """
    names: dict[str, str] = {}
    for member in entry["accessors"]:
        names.setdefault(member["rust"], member["js"])
    return names


def property_of(argument: str, names: dict[str, str]) -> str:
    """The JavaScript property a constructor argument fills in."""
    # `type_` is web-sys escaping the keyword `type`.
    field = argument.rstrip("_")
    for candidate in (f"set_{field}", f"get_{field}"):
        if candidate in names:
            return names[candidate]
    # `set_size_f64` and `set_size` name one property; either spelling will do.
    prefixed = sorted(js for rust, js in names.items() if rust.startswith(f"set_{field}"))
    if prefixed:
        return prefixed[0]
    raise SystemExit(
        f"tools/generate.py: no property found for the constructor argument `{argument}`"
    )


def emit_type(name: str, entry: dict, surface: Surface, types: Types) -> list[str]:
    """One type: its `js_type!`, then the members declared on it."""
    js_class = entry["js_class"]
    dictionary = name in surface.dictionaries
    lines: list[str] = ["js_type! {"]

    if dictionary:
        lines += [
            f"    /// The WebGPU `{js_class}` dictionary.",
            "    ///",
            "    /// A dictionary is a plain object with no class of its own, so the cast",
            "    /// tests for an object rather than for a constructor.",
            f"    {name}: {ancestry(entry['extends'], types)},",
            "    instanceof(value) { value.is_object() },",
        ]
    else:
        lines += [
            f"    /// The `{js_class}` interface.",
            "    ///",
            *mdn_doc(js_class),
            f"    {name}: {ancestry(entry['extends'], types)},",
            f'    instanceof(value) {{ crate::napi::rt::instance_of(value, c"{js_class}") }},',
        ]
    lines += ["}"]

    members = sorted(entry["members"], key=sort_key)
    if not members:
        return lines

    macro = "webgpu_dictionary" if dictionary else "webgpu_members"
    body: list[str] = []
    names = property_names(entry) if dictionary else {}
    for member in members:
        kind = member["kind"]
        catch = " catch" if member["catch"] else ""
        arguments = member["args"]

        if kind == "constructor":
            if not dictionary:
                raise SystemExit(
                    f"tools/generate.py: `{name}.{member['rust']}` is a constructor on an"
                    " interface, which the DSL has no form for"
                )
            spelled = [
                f"{argument['name']}: {types.argument(argument['type'])}"
                f' as "{property_of(argument["name"], names)}"'
                for argument in arguments
            ]
            body.append(member_doc(kind, member, name, dictionary))
            body += member_line("new", member["rust"], spelled, ";")
            continue

        spelled = [
            f"{argument['name']}: {types.argument(argument['type'])}"
            for argument in arguments
        ]
        # `catch` is what turns the return type into a `Result`, so the declaration
        # names what the call resolves to and the spec's `Result<_, JsValue>` is
        # unwrapped here.
        head, generics = parse_generic(normalize(member["ret"]))
        returned = types.value(generics[0] if head == "Result" else member["ret"])
        result = "" if returned == "()" else f" -> {returned}"
        tail = f'{result} as "{member["js"]}"{catch};'
        body.append(member_doc(kind, member, name, dictionary))
        body += member_line(kind, member["rust"], spelled, tail)

    # The class name is stated only where the DSL's `Gpu` → `GPU` rule would get it
    # wrong, which is what `webgpu_members!` documents. The spec is the authority
    # for both sides of that comparison.
    header = f"    {name};"
    if js_class != derived_class(name):
        header = f'    {name} as "{js_class}";'
    declaration = lines + ["", f"{macro}! {{", header, *body, "}"]

    # A dictionary whose constructor takes nothing is a default value in the
    # ordinary sense, and clippy's `new_without_default` says so. `webgpu_members!`
    # cannot emit the trait impl itself — its members expand inside an `impl` block
    # — so the call is stated here, where the argument count is known.
    empty = [
        member
        for member in members
        if member["kind"] == "constructor" and not member["args"]
    ]
    for member in empty:
        declaration += ["", f'webgpu_default!({name}, {member["rust"]});']
    return declaration


def emit_enum(name: str, entry: dict) -> list[str]:
    """One string enum: its Rust spelling against its JavaScript spelling.

    The spec records no JavaScript name for an enum, because a string enum has no
    JavaScript object to name; web-sys lowered `GPU` to `Gpu` deriving the Rust
    name, and `derived_class` raises it back.
    """
    js_name = derived_class(name)
    lines = [
        "webgpu_enum! {",
        f"    /// The `{js_name}` enumeration.",
        f'    {name} as "{js_name}";',
    ]
    lines += [
        f'    {variant["rust"]} = "{variant["js"]}",' for variant in entry["variants"]
    ]
    lines.append("}")
    return lines


def emit_namespace(name: str, entry: dict) -> list[str]:
    """One WebIDL namespace: a module of constants.

    `GPUMapMode.READ` is a JavaScript property read in principle and a compile-time
    constant in practice — the values are fixed by the specification — so this
    emits them as `const`s, exactly as web-sys does.
    """
    js_name = derived_class("".join(part.title() for part in name.split("_")))
    lines = [
        f"/// The `{js_name}` namespace.",
        "///",
        f"/// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/{js_name})",
        f"pub mod {name} {{",
    ]
    for index, constant in enumerate(entry["constants"]):
        if index:
            lines.append("")
        lines += [
            f'    /// The `{js_name}.{constant["name"]}` const.',
            f'    pub const {constant["name"]}: {constant["type"]} = {constant["value"]};',
        ]
    lines.append("}")
    return lines


def imports(module: str, referenced: set[str], home: dict[str, str]) -> list[str]:
    """`use` lines for the generated types this module names but does not declare."""
    by_module: dict[str, list[str]] = {}
    for name in sorted(referenced):
        other = home[name]
        if other != module:
            by_module.setdefault(other, []).append(name)
    lines = []
    for other in sorted(by_module):
        names = by_module[other]
        if len(names) == 1:
            lines.append(f"use super::{other}::{names[0]};")
            continue
        single = f"use super::{other}::{{{', '.join(names)}}};"
        if len(single) <= RUSTFMT_WIDTH:
            lines.append(single)
            continue
        # rustfmt's default `Mixed` import layout: a list that does not fit on one
        # line becomes a block whose lines are filled to `max_width`. Emitting it
        # that way keeps `cargo fmt` from rewriting a generated file.
        lines.append(f"use super::{other}::{{")
        row = ""
        for name in names:
            candidate = f"{row} {name}," if row else f"    {name},"
            if len(candidate) > RUSTFMT_WIDTH:
                lines.append(row)
                row = f"    {name},"
            else:
                row = candidate
        lines += [row, "};"]
    return lines


def render(module: str, doc: str, body: list[str], import_lines: list[str]) -> str:
    parts = [HEADER.format(doc=doc)]
    if import_lines:
        parts.append("\n".join(import_lines) + "\n")
    parts.append("\n".join(body) + "\n")
    return "\n".join(parts)


def main() -> int:
    surface = Surface(json.loads(SPEC.read_text()))

    home: dict[str, str] = {}
    for module, (_, names) in MODULES.items():
        for name in names:
            home[name] = module
    for name in surface.enums:
        home[name] = ENUM_MODULE
    misfiled = sorted(name for name in surface.interfaces if name not in home)
    for name in misfiled:
        home[name] = FALLBACK_MODULE

    contents: dict[str, str] = {}
    member_count = 0

    modules = list(MODULES.items())
    if misfiled:
        modules.append((FALLBACK_MODULE, (FALLBACK_MODULE_DOC, misfiled)))

    # Every type the spec declares, not only the ones being emitted: a member
    # may name a type whose own entry is not `used`, and that has to become an
    # unresolved import for `tools/shake.py` to notice rather than a crash here.
    # A type the tables have no mapping for at all is still a hard error.
    known_types = surface.all_interfaces | surface.all_enums
    for module, (doc, names) in modules:
        types = Types(known_types)
        body: list[str] = []
        for name in sorted(names):
            entry = surface.interfaces.get(name)
            if entry is None:
                # Named by the table but no longer in the surface: the spec shrank.
                continue
            if body:
                body.append("")
            body += emit_type(name, entry, surface, types)
            member_count += len(entry["members"])
        if not body:
            continue
        declared = {name for name in names if name in surface.interfaces}
        contents[module] = render(
            module, doc, body, imports(module, types.referenced - declared, home)
        )

    body = []
    for name in sorted(surface.enums):
        if body:
            body.append("")
        body += emit_enum(name, surface.enums[name])
    contents[ENUM_MODULE] = render(ENUM_MODULE, ENUM_MODULE_DOC, body, [])

    if surface.namespaces:
        body = []
        for name in sorted(surface.namespaces):
            if body:
                body.append("")
            body += emit_namespace(name, surface.namespaces[name])
        contents[NAMESPACE_MODULE] = render(
            NAMESPACE_MODULE, NAMESPACE_MODULE_DOC, body, []
        )

    mod_body = [f"mod {module};" for module in sorted(contents)]
    mod_body.append("")
    mod_body += [f"pub use {module}::*;" for module in sorted(contents)]
    contents["mod"] = render(
        "mod",
        "The generated WebGPU bindings.\n//!\n//! One file per subject; every"
        " declaration in them is re-exported here, so a\n//! caller names"
        " `webgpu::GpuDevice` rather than the file it happens to be in.",
        mod_body,
        [],
    )

    OUT.mkdir(parents=True, exist_ok=True)
    for module, text in sorted(contents.items()):
        (OUT / f"{module}.rs").write_text(text)

    dictionaries = len(surface.dictionaries)
    interfaces = len(surface.interfaces) - dictionaries
    variants = sum(len(entry["variants"]) for entry in surface.enums.values())
    print(f"files       : {len(contents):4}  in {OUT.relative_to(CRATE)}")
    print(f"interfaces  : {interfaces:4}")
    print(f"dictionaries: {dictionaries:4}")
    print(f"members     : {member_count:4}")
    print(f"enums       : {len(surface.enums):4}  ({variants} variants)")
    constants = sum(len(entry["constants"]) for entry in surface.namespaces.values())
    print(f"namespaces  : {len(surface.namespaces):4}  ({constants} constants)")
    if misfiled:
        print(f"\nunfiled, landed in {FALLBACK_MODULE}.rs: {', '.join(misfiled)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
