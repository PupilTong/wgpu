//! The Node-API plumbing every WebGPU binding in this crate is built on.
//!
//! Nothing here is WebGPU-specific and nothing here imitates another crate's API:
//! it is the machinery a Rust caller needs to hold and use JavaScript values
//! through Node-API — an environment per thread, a handle that survives past the
//! scope that produced it, conversions in both directions, and the four operations
//! a binding lowers to (read a property, write a property, call a method,
//! construct).
//!
//! * [`env`] — the per-thread `napi_env`, handle scopes, status → `Result`.
//! * [`value`] — [`JsValue`](value::JsValue): a primitive held in Rust, or an
//!   `Rc`-counted `napi_ref` for anything with identity.
//! * [`convert`] — [`AsJs`](convert::AsJs) / [`FromJs`](convert::FromJs), the one
//!   ABI across the boundary.
//! * [`rt`] — the operations themselves, keyed by C string literals so a call
//!   needs no allocation for its own name.
//! * [`closure`] — Rust functions JavaScript can call, freed by a finalizer.
//! * [`futures`] — awaiting a JavaScript promise, and running Rust futures on the
//!   JavaScript event loop.

pub(crate) mod closure;
pub(crate) mod convert;
pub(crate) mod env;
pub(crate) mod futures;
pub(crate) mod rt;
pub(crate) mod value;
