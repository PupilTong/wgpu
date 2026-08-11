//! Where the `GPU` object comes from.
//!
//! [`crate::install`] gives this crate a thread's `napi_env`, which is enough to
//! reach JavaScript. It is not enough to reach WebGPU: the entry point to the API
//! is a `GPU` object, and a host is free to put it anywhere.
//!
//! On the web it is `navigator.gpu`, and [`gpu`] finds it there. Under Node it
//! usually is not: the `webgpu` npm package returns a `GPU` from `create([])` and
//! installs no `navigator` at all, so nothing on the global names it. That is why
//! [`install_gpu`] exists — the object is passed in rather than discovered, and
//! discovery is only the fallback.
//!
//! One slot per thread, like the environment, because the `GPU` is a JavaScript
//! value and a JavaScript value belongs to the `napi_env` of one thread.

use core::cell::RefCell;

use napi_sys as sys;

use crate::napi::rt;
use crate::napi::value::JsValue;

thread_local! {
    /// The `GPU` [`install_gpu`] was given, if it was called on this thread.
    static GPU: RefCell<Option<JsValue>> = const { RefCell::new(None) };
}

/// Takes a Node-API value into this crate's [`JsValue`].
///
/// The bridge an addon needs to hand anything to this crate from the raw ABI: a
/// napi-rs `JsUnknown` (or any `napi_value`) becomes a value this crate can hold
/// past the call that produced it. Objects are kept behind a `napi_ref`;
/// primitives are copied into Rust and need no reference at all.
///
/// Its one caller in practice is [`install_gpu`], which is why it lives here.
///
/// # Safety
///
/// `env` must be the calling thread's live `napi_env` and `value` must be a valid
/// `napi_value` in that environment's *current* handle scope — inside a `#[napi]`
/// function or a Node-API callback, that is the scope the runtime already opened.
#[must_use]
pub unsafe fn adopt_js_value(env: sys::napi_env, value: sys::napi_value) -> JsValue {
    // SAFETY: the caller guarantees a live `env` and a `value` in its current
    // handle scope, which is exactly `from_napi`'s requirement.
    unsafe { JsValue::from_napi(env, value) }
}

/// Installs the `GPU` object this thread reaches WebGPU through.
///
/// Call it once per thread, after [`crate::install`], with whatever the host's
/// bootstrap produced — for the `webgpu` npm package, the return value of
/// `create([])`:
///
/// ```ignore
/// #[napi]
/// fn install_webgpu(env: napi::Env, gpu: napi::JsUnknown) {
///     // SAFETY: `env` is this thread's environment, and `gpu` is a live value
///     // in the handle scope Node opened for this call.
///     unsafe {
///         napi_rs_webgpu::install(env.raw());
///         let gpu = napi_rs_webgpu::adopt_js_value(env.raw(), gpu.raw());
///         napi_rs_webgpu::install_gpu(gpu);
///     }
/// }
/// ```
///
/// Installing `undefined` or `null` clears the slot, which puts [`gpu`] back on
/// its `navigator.gpu` fallback. Installing a second object replaces the first;
/// WebGPU objects already made from the old one keep working, because each holds
/// its own handle.
///
/// # Safety
///
/// `gpu` must belong to the Node-API environment installed on this thread, and
/// must stay valid for as long as it stays installed. This crate holds the value
/// past the call that supplied it and can check neither condition: a handle
/// carries no type, and only the host knows when its environment is torn down.
/// Releasing the handle after that point is undefined, which is what makes this
/// `unsafe` — a handle from *another* thread's environment is merely refused, by
/// every call that tries to use it.
pub unsafe fn install_gpu(gpu: JsValue) {
    let installed = (!gpu.is_undefined() && !gpu.is_null()).then_some(gpu);
    GPU.with(|slot| *slot.borrow_mut() = installed);
}

/// This thread's `GPU`, or `None` when WebGPU is not reachable from it.
///
/// In order: the object [`install_gpu`] was given, then `globalThis.navigator.gpu`
/// — where every browser puts it, and where a Node host that emulates `navigator`
/// puts it too. A `navigator` without a `gpu`, or no `navigator` at all, is
/// `None`.
///
/// The fallback is read afresh each time rather than cached, so a host that
/// defines `navigator.gpu` late still gets found.
///
/// No `instanceof GPU` is performed on either path. A host that hands over a
/// `GPU` knows what it handed over, and node-webgpu's class is not on the global
/// for the check to find; on the web, `navigator.gpu` is a `GPU` by definition.
/// That matches the rest of the crate, where a value returned by the API that
/// declares it is taken at its word.
///
/// # Panics
///
/// If nothing was installed by [`install_gpu`] and [`crate::install`] was never
/// called on this thread: reading a global needs an environment, and a thread
/// without one has no JavaScript to search.
//
// TODO: return `Option<Gpu>` once `crate::webgpu` re-exports the generated `Gpu`
// — it is declared in `webgpu/generated/adapter.rs` but not yet reachable from
// `webgpu/mod.rs`. The change is `Gpu::from(value)` at both returns and nothing
// else: `Gpu` is a transparent newtype over exactly the handle returned here.
#[must_use]
pub fn gpu() -> Option<JsValue> {
    if let Some(installed) = GPU.with(|slot| slot.borrow().clone()) {
        return Some(installed);
    }
    let navigator = rt::global(c"navigator").ok()?;
    if navigator.is_undefined() || navigator.is_null() {
        return None;
    }
    let gpu = rt::get(&navigator, c"gpu").ok()?;
    (!gpu.is_undefined() && !gpu.is_null()).then_some(gpu)
}
