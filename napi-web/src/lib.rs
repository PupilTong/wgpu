//! The wasm-bindgen family, re-implemented on Node-API, for `wgpu`'s WebGPU
//! backend on `wasm32-wasip1(-threads)`.
//!
//! # Why this exists
//!
//! `wgpu`'s WebGPU backend is a thin mapping onto the JavaScript WebGPU objects,
//! written against `wasm-bindgen`, `js-sys` and `web-sys`. Those crates build for
//! any `wasm32` target, but they only *work* on `wasm32-unknown-unknown`: every
//! binding is a wasm import from a placeholder module that `wasm-bindgen-cli`
//! resolves when it emits the JS glue, and that glue is only produced for
//! `wasm32-unknown-unknown`.
//!
//! On WASI, wasm-bindgen compiles each binding to a stub whose body is
//! `panic!("function not implemented on non-wasm32 targets")`. Such a build links
//! and instantiates, and then aborts on its first JavaScript operation — a
//! failure that neither `cargo check` nor loading the module reveals.
//!
//! A napi-rs addon reaches JavaScript by a different route: the module is loaded
//! by `@napi-rs/wasm-runtime`, emnapi implements Node-API against the host's
//! JavaScript, and the Rust side calls it through `napi-sys`. This crate supplies
//! the wasm-bindgen *surface* that `wgpu` is written against, with each binding
//! lowered onto that route — a property read, a method call, a construction —
//! against the very same `navigator.gpu` objects.
//!
//! # What is here
//!
//! * [`wasm_bindgen`] — [`JsValue`](value::JsValue), [`JsCast`](value::JsCast) and
//!   the `#[wasm_bindgen]` attribute, which lowers `extern "C"` declarations to
//!   Node-API calls instead of wasm imports.
//! * [`js_sys`] — the JavaScript built-ins the backend uses, including the typed
//!   containers (`Array<T>`, `Object<T>`, `Promise<T>`, `JsOption<T>`) that modern
//!   `web-sys` bindings are written in terms of.
//! * [`web_sys`] — the handful of DOM types WebGPU surface creation needs.
//! * [`wasm_bindgen_futures`] — `JsFuture` and `spawn_local`, driven by JavaScript
//!   promise callbacks rather than by wasm-bindgen's executor.
//!
//! # Using it from an addon
//!
//! Node-API operations need an `napi_env`, and `wgpu`'s API has nowhere to put
//! one, so the addon installs it once per thread before the first `wgpu` call:
//!
//! ```ignore
//! #[napi::module_init]
//! fn init(env: napi::Env) {
//!     // SAFETY: `env` is live for the lifetime of the module on this thread.
//!     unsafe { wgpu_napi_web::install(env.raw()) };
//! }
//! ```
//!
//! Everything else is `wgpu` as usual. See [`install`] for the per-thread rules.
//!
//! # What is not here
//!
//! This is not a general-purpose wasm-bindgen replacement. It covers the surface
//! `wgpu` uses, and its conversions follow wasm-bindgen's generated bindings —
//! unchecked reinterpretation of whatever JavaScript returned — rather than
//! validating types at the boundary.

extern crate alloc;

mod closure;
mod convert;
mod env;
mod futures;
mod rt;
mod value;

pub mod js_sys;
pub mod web_sys;

pub use env::{install, is_installed, uninstall};

/// Stand-in for the `wasm-bindgen` crate.
pub mod wasm_bindgen {
    pub use crate::convert::{AsJs, FromJs};
    pub use crate::value::{JsCast, JsError, JsGeneric, JsValue, Promising, UnwrapThrowExt};
    pub use wgpu_napi_web_macro::wasm_bindgen;

    /// Stand-in for `wasm_bindgen::closure`.
    pub mod closure {
        pub use crate::closure::{Closure, ScopedClosure};
    }

    /// Stand-in for `wasm_bindgen::sys`, the fundamental JavaScript types used as
    /// generic parameters.
    pub mod sys {
        pub use crate::js_sys::{JsOption, Null, Undefined};
        pub use crate::value::Promising;
    }

    /// Stand-in for `wasm_bindgen::prelude`.
    pub mod prelude {
        pub use crate::closure::Closure;
        pub use crate::value::{JsCast, JsError, JsValue, UnwrapThrowExt};
        pub use wgpu_napi_web_macro::wasm_bindgen;
    }
}

/// Stand-in for the `wasm-bindgen-futures` crate.
pub mod wasm_bindgen_futures {
    pub use crate::futures::{spawn_local, JsFuture};
}

#[doc(hidden)]
pub mod __rt {
    //! What the `#[wasm_bindgen]` attribute expands to. Not a stable surface.
    pub use crate::convert::{AsJs, FromJs};
    pub use crate::rt::{
        array_from, array_items, array_length, call, call_method, cast, construct, describe, error,
        get, get_dynamic, get_index, global, global_this, instance_of, new_object, set,
        set_dynamic, set_index, unwrap_js,
    };
    pub use crate::value::{JsCast, JsGeneric, JsValue, Promising};
}
