//! The one place this crate names its JavaScript interop layer.
//!
//! [`ExternalImageSource`](crate::ExternalImageSource) wraps live JS objects, so
//! it has to agree with whichever interop layer `wgpu` itself compiled against:
//! the real `web-sys` family by default, or the Node-API stand-in from
//! [`wgpu_napi_web`] under the `napi-web` feature (WASI only, where wasm-bindgen
//! has no working loader). See `wgpu/src/js.rs` for the full reasoning.

// The condition matches `wgpu`'s `napi_web` cfg alias, including its WASI test:
// the shim is only a dependency for WASI targets, so `napi-web` enabled for any
// other target must keep resolving to the real crates.
#[cfg(not(all(target_os = "wasi", feature = "napi-web")))]
pub use {::js_sys, ::web_sys};

#[cfg(all(target_os = "wasi", feature = "napi-web"))]
pub use ::wgpu_napi_web::{js_sys, web_sys};
