//! The one place the crate names its JavaScript interop layer.
//!
//! Every `js_sys` / `web_sys` / `wasm_bindgen` / `wasm_bindgen_futures` path in
//! this crate is written as `crate::js::…` instead of naming those crates
//! directly, so that a single `cfg` here selects which implementation the whole
//! crate compiles against:
//!
//! * by default, the real [`wasm_bindgen`] family — the only thing that works on
//!   `wasm32-unknown-unknown`, where the loader is wasm-bindgen's generated JS
//!   glue;
//! * with the `napi-web` feature on `wasm32-wasip1(-threads)`, the
//!   [`wgpu_napi_web`] shim, which reaches the same JavaScript objects through
//!   Node-API (napi-rs / emnapi) instead. wasm-bindgen cannot be used there at
//!   all: its imports are satisfied by glue that only exists for
//!   `wasm32-unknown-unknown`, so a WASI module built against it links but
//!   cannot be instantiated.
//!
//! The generated bindings under [`crate::backend::webgpu::webgpu_sys`] are
//! re-vendored from `web-sys` by `cargo xtask vendor-web-sys`, which rewrites
//! their crate paths to point here — keep that rewrite in sync with this module.

// `fragile-send-sync-non-atomic-wasm` claims `Send`/`Sync` on the argument that a
// wasm binary without atomics is single-threaded. That argument does not hold here:
// `wasm32-wasip1-threads` has real threads yet reports no `atomics` target feature,
// so the feature would silently mark thread-affine Node-API handles as shareable.
// A Node-API value belongs to the `napi_env` of one thread, so this is refused
// rather than left to fail at runtime.
#[cfg(all(napi_web, send_sync))]
compile_error!(
    "wgpu: `fragile-send-sync-non-atomic-wasm` cannot be combined with `napi-web` — \
     Node-API values are bound to the thread that owns their environment. Keep `wgpu` \
     objects on one thread instead."
);

#[cfg(not(napi_web))]
pub use {::js_sys, ::wasm_bindgen, ::wasm_bindgen_futures, ::web_sys};

#[cfg(napi_web)]
pub use ::wgpu_napi_web::{js_sys, wasm_bindgen, wasm_bindgen_futures, web_sys};
