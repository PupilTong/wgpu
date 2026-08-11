//! WebGPU, bound into Rust over Node-API, for `wasm32-wasip1` and
//! `wasm32-wasip1-threads`.
//!
//! # Why this exists
//!
//! WebGPU in Rust normally arrives through `web-sys`, whose bindings are
//! wasm-bindgen imports resolved by JS glue that `wasm-bindgen-cli` only emits for
//! `wasm32-unknown-unknown`. On WASI that glue does not exist: wasm-bindgen
//! compiles every binding to a stub whose body is
//! `panic!("function not implemented on non-wasm32 targets")`, so such a build
//! loads and then aborts on its first JavaScript call.
//!
//! A napi-rs addon reaches JavaScript by another route entirely — the module is
//! loaded by `@napi-rs/wasm-runtime`, emnapi implements Node-API against the host's
//! JavaScript, and Rust calls it through `napi-sys`. This crate binds WebGPU over
//! that route: each interface is a handle to the real JavaScript object, and each
//! member is the property read, property write, method call or construction that
//! WebGPU's IDL says it is.
//!
//! There is no wasm-bindgen, `js-sys` or `web-sys` here, and nothing here imitates
//! them.
//!
//! # Scope
//!
//! The bindings cover what a WebGPU consumer actually calls, derived from
//! `wgpu`'s backend rather than guessed: 94 interfaces, 437 members, 32 string
//! enums and one namespace, out of the 1156 members `web-sys` generates.
//! `tools/extract_surface.py` recomputes that set from the IDL-generated bindings,
//! so the scope can be re-derived rather than maintained by hand.
//!
//! Alongside them are the JavaScript language types WebGPU's IDL names — [`Object`],
//! [`Promise`], [`JsString`], [`JsOption`], [`JsIterator`], [`ArrayBuffer`] and the
//! typed arrays — and the DOM types its external-image sources and surface creation
//! name. Those are this crate's own; nothing here imitates `js-sys` or `web-sys`.
//!
//! # Using it
//!
//! Node-API operations need an `napi_env`, and WebGPU's API has nowhere to carry
//! one, so the addon installs it per thread before its first call:
//!
//! ```ignore
//! #[napi::module_init]
//! fn init(env: napi::Env) {
//!     // SAFETY: `env` is live for the lifetime of the module on this thread.
//!     unsafe { napi_rs_webgpu::install(env.raw()) };
//! }
//! ```
//!
//! The `GPU` itself is passed in rather than discovered, because a host need not
//! put it on a global: node-webgpu's `create([])` hands back a `GPU` and installs
//! no `navigator`. [`install_gpu`] takes that object, with [`adopt_js_value`]
//! turning the raw `napi_value` into one of this crate's handles:
//!
//! ```ignore
//! #[napi]
//! fn install_webgpu(env: napi::Env, gpu: napi::JsUnknown) {
//!     // SAFETY: `gpu` is live in the handle scope Node opened for this call.
//!     unsafe { napi_rs_webgpu::install_gpu(napi_rs_webgpu::adopt_js_value(env.raw(), gpu.raw())) };
//! }
//! ```
//!
//! When a host does expose `navigator.gpu` — every browser — [`gpu`] finds it
//! there instead, and nothing needs to be installed.
//!
//! Every thread that touches WebGPU needs its own [`install`], because a
//! `napi_env`, and every JavaScript value reached through it, belongs to one
//! thread. The handles here are `!Send` accordingly, so that is enforced rather
//! than documented.

extern crate alloc;

#[macro_use]
mod dsl;

mod builtins;
mod dom;
mod entry;
mod napi;
mod support;
mod typed_array;
mod webgpu;

pub use builtins::{global, JsIter, JsIterator, JsOption, JsString, Number, Reflect, Undefined};
pub use dom::*;
pub use entry::{adopt_js_value, gpu, install_gpu};
pub use napi::env::{install, is_installed, uninstall};
pub use napi::value::{JsCast, JsError, JsValue};
pub use support::{Error, Object, Promise};
pub use typed_array::{ArrayBuffer, Uint32Array, Uint8Array};
pub use webgpu::*;

/// Awaiting JavaScript promises, and running Rust futures on the JavaScript event
/// loop.
pub mod futures {
    pub use crate::napi::futures::{spawn_local, JsFuture};
}

/// Rust functions JavaScript can call.
pub mod callback {
    pub use crate::napi::closure::{Closure, ScopedClosure};
}
