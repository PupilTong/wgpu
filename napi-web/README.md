# wgpu-napi-web

The wasm-bindgen family — `wasm-bindgen`, `js-sys`, `web-sys`,
`wasm-bindgen-futures` — re-implemented on Node-API, so that `wgpu`'s WebGPU
backend works on `wasm32-wasip1` and `wasm32-wasip1-threads`.

## Why a WASI build needs this

`wgpu`'s WebGPU backend is a mapping onto JavaScript's WebGPU objects, written
against wasm-bindgen. Those crates *build* for every `wasm32` target, but they only
*work* on `wasm32-unknown-unknown`: each binding is an import from a placeholder
module that `wasm-bindgen-cli` resolves when it generates the JS glue, and that glue
is only produced for `wasm32-unknown-unknown`.

On WASI, wasm-bindgen takes its non-web path — every binding compiles to a Rust
function whose body is `panic!("function not implemented on non-wasm32 targets")`
(`wasm-bindgen/src/lib.rs`, the `externs!` macro). The module builds, links and
instantiates; the first JavaScript operation aborts it. So `cargo check` passing
proves nothing about a WASI target, and neither does a successful load. (Between
0.2.115 and [wasm-bindgen#5175](https://github.com/wasm-bindgen/wasm-bindgen/pull/5175)
the same build instead emitted unresolved `__wbindgen_placeholder__` imports and
could not be instantiated at all — a louder failure for the same reason.)

A napi-rs addon reaches JavaScript by another route: the module is loaded by
`@napi-rs/wasm-runtime`, emnapi implements Node-API against the host's JavaScript,
and Rust calls it through `napi-sys`. This crate keeps the API surface `wgpu` is
written against and re-points every binding at that route — a property read, a
method call, a construction — against the very same `navigator.gpu` objects.

Enable it with `wgpu`'s `napi-web` feature. It has no effect on any other target.

```toml
[target.'cfg(target_os = "wasi")'.dependencies]
wgpu = { version = "30", default-features = false, features = ["napi-web", "wgsl"] }
```

## Installing the environment

Node-API operations need an `napi_env`, and `wgpu`'s API has nowhere to carry one,
so the addon installs it per thread before the first `wgpu` call:

```rust
#[napi::module_init]
fn init(env: napi::Env) {
    // SAFETY: `env` is live for the lifetime of the module on this thread.
    unsafe { wgpu_napi_web::install(env.raw()) };
}
```

Every thread that touches `wgpu` needs its own `install`, because a `napi_env` — and
every JavaScript value reached through it — belongs to one thread. Rendering from a
thread without one panics with that message rather than corrupting anything.

## Building the addon

This crate has no build script and needs no emnapi to `cargo check` or `cargo
clippy`. Linking the final `cdylib` does, and that is `napi-build`'s job in the
addon, not this crate's:

```bash
EMNAPI_LINK_DIR=<dir with libemnapi-napi-rs-mt.a / libemnapi-basic-napi-rs.a> \
  cargo build --release --target wasm32-wasip1-threads
```

`napi-build`'s WASI setup emits the linker arguments (`--import-memory`,
`--import-undefined`, `--export-table`, `crt1-reactor.o`, a 64 MiB stack) and picks
the emnapi archive matching the threading model: `emnapi-napi-rs-mt` for
`-threads`, `emnapi-basic-napi-rs` otherwise. Without `EMNAPI_LINK_DIR` its build
script fails outright, so add `napi-build` only to the crate that actually links.

Verify a build really is free of wasm-bindgen by inspecting the module's imports —
they should be Node-API and WASI only, with no `__wbindgen*`:

```bash
wasm-tools print target/wasm32-wasip1-threads/release/<addon>.wasm | grep '(import'
```

## Scope

This is not a general-purpose wasm-bindgen replacement. It covers what `wgpu`'s
backend uses, and its conversions follow wasm-bindgen's generated bindings —
unchecked reinterpretation of whatever JavaScript returned — rather than validating
types at the boundary. Three differences are worth knowing:

- **Closures outlive their handle.** `napi_create_function` gives JavaScript a real
  function that owns its Rust state through a finalizer, so dropping a `Closure`
  cannot invalidate a callback JavaScript still holds. `Closure::forget` therefore
  has nothing to leak.
- **A thrown exception from a non-`catch` binding panics.** wasm-bindgen lets it
  escape through the import boundary; Node-API leaves it pending on the
  environment, where it would poison every later call, so it is taken and reported.
  Under `panic = "abort"` — every WASI target Rust ships — that ends the module,
  which is what an uncaught `throw` through wasm-bindgen does too.
- **`js_sys::Uint8Array::view` copies.** wasm-bindgen can hand JavaScript a window
  onto wasm memory; a Node-API typed array is backed by its own `ArrayBuffer`, so
  writes are staged and copied. See the `js_sys` module docs for exactly where.

## Layout

| Path | What it is |
| --- | --- |
| `src/env.rs` | the per-thread `napi_env`, handle scopes, status → `Result` |
| `src/value.rs` | `JsValue` (Rust-side primitives, or an `Rc`-counted `napi_ref`), `JsCast` |
| `src/convert.rs` | `AsJs` / `FromJs`, the one ABI across the boundary |
| `src/rt.rs` | the Node-API operations the `#[wasm_bindgen]` attribute lowers to |
| `src/js_sys/` | the JavaScript built-ins, including the typed containers |
| `src/web_sys.rs` | the DOM types WebGPU surface creation needs |
| `src/closure.rs` | Rust functions callable from JavaScript |
| `src/futures.rs` | `JsFuture` and `spawn_local`, driven by the JavaScript event loop |
| `../napi-web-macro/` | the `#[wasm_bindgen]` attribute, lowering bindings onto `src/rt.rs` |
