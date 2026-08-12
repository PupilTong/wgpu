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
//! This binding uses Emnapi 2's `emnapi_sync_memory` pointer ABI to flush
//! Wasm-side ArrayBuffer staging allocations before JavaScript reads them. Hosts
//! must therefore load it with Emnapi 2.x (the acceptance harness pins
//! `2.0.0-alpha.3`); Emnapi 1.x has an incompatible by-value ABI.
//!
//! There is no wasm-bindgen, `js-sys` or `web-sys` here, and nothing here imitates
//! them.
//!
//! # Scope
//!
//! The bindings cover what `wgpu` actually calls and nothing else: 88 interfaces,
//! 364 members, 32 string enums and one namespace, out of the 1156 members
//! `web-sys` generates. That set is not a judgement — `tools/extract_surface.py`
//! derives the whole surface from the IDL-generated bindings and `tools/shake.py`
//! then empties it and adds back exactly what the compiler asks for, so removing
//! any of what is left breaks a build. Both are commands, so a `wgpu` that reaches
//! further is a re-run away from having its bindings back.
//!
//! Alongside them are the JavaScript language types WebGPU's IDL names — [`Object`],
//! [`Promise`], [`JsString`], [`JsOption`], [`JsIterator`], [`ArrayBuffer`] and the
//! typed arrays — and the DOM types its external-image sources and surface creation
//! name. Those are this crate's own; nothing here imitates `js-sys` or `web-sys`.
//!
//! # Using it
//!
//! Node-API operations need an `napi_env`, and WebGPU's API has nowhere to carry
//! one, so the addon installs it per thread before its first call. The `GPU` is
//! passed in rather than discovered for the same kind of reason: a host need not
//! put it on a global, and node-webgpu's `create([])` hands one back while
//! installing no `navigator`. [`adopt_js_value`] turns the raw `napi_value` into
//! one of this crate's handles:
//!
//! ```ignore
//! #[napi]
//! fn install_webgpu(env: napi::Env, gpu: napi::Unknown) {
//!     // SAFETY: `env` is this thread's environment, live for as long as the
//!     // module is loaded, and `gpu` is live in the handle scope Node opened
//!     // for this call.
//!     unsafe {
//!         napi_rs_webgpu::install(env.raw());
//!         napi_rs_webgpu::install_gpu(napi_rs_webgpu::adopt_js_value(env.raw(), gpu.raw()));
//!     }
//! }
//! ```
//!
//! Both happen in a `#[napi]` function rather than at module load because
//! napi-rs' `#[module_init]` is a static constructor: it runs before Node-API
//! exists and is handed no environment, so the first `#[napi]` call is the
//! earliest point one is available.
//!
//! When a host does expose `navigator.gpu` — every browser — [`gpu`] finds it
//! there instead, and only [`install`] is needed.
//!
//! `harness/` is a worked example of all of this: an addon built for
//! `wasm32-wasip1-threads` that renders a triangle through node-webgpu and checks
//! the pixels.
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

pub use builtins::{JsIter, JsIterator, JsOption, JsString, Number, Reflect, Undefined};
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
