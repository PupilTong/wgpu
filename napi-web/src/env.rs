//! The Node-API environment every call in this crate goes through.
//!
//! wasm-bindgen gets its JavaScript context for free: its imports are bound at
//! instantiation, so any Rust code can call JS at any time. Node-API instead
//! hands out an opaque `napi_env` at module initialisation, and every call needs
//! it. `wgpu`'s API takes no environment parameter, so the addon installs one
//! here once and this crate reads it from a thread-local afterwards.
//!
//! One `napi_env` per thread is the Node-API model — Node's main thread, each
//! `worker_threads` worker, and under `@napi-rs/wasm-runtime` each spawned wasm
//! thread. A thread that never called [`install`] cannot touch JavaScript, which
//! is exactly the truth about that thread.

use core::cell::Cell;
use core::ptr;

use napi_sys as sys;

use crate::value::JsValue;

thread_local! {
    static ENV: Cell<sys::napi_env> = const { Cell::new(ptr::null_mut()) };
}

/// Installs the Node-API environment for the current thread.
///
/// Call this once from the addon, as early as an `napi_env` exists — from
/// `#[napi::module_init]`, from a `#[napi]` function before the first `wgpu`
/// call, or from a raw `napi_register_wasm_v1` entry point. With napi-rs 3.x the
/// value comes from `napi::Env::raw`; a raw Node-API embedder passes the
/// `napi_env` it was handed directly.
///
/// Installing a second, different environment on the same thread replaces the
/// first: the JavaScript values already held by live `wgpu` objects belong to
/// the old environment and calling into them afterwards is undefined, so only do
/// this after those objects are gone.
///
/// # Safety
///
/// `env` must be a live `napi_env` for the calling thread, and must stay live for
/// as long as this crate is used from that thread (i.e. until [`uninstall`], or
/// for the lifetime of the addon).
pub unsafe fn install(env: sys::napi_env) {
    assert!(
        !env.is_null(),
        "wgpu-napi-web: install() needs a live napi_env, got null"
    );
    ENV.with(|slot| slot.set(env));
}

/// Forgets the environment installed for this thread.
///
/// Call this from a Node-API cleanup hook, after every `wgpu` object made through
/// this shim has been dropped. JavaScript handles dropped afterwards can no
/// longer release their Node-API reference and are simply left to the garbage
/// collector, which is the correct outcome during environment teardown.
pub fn uninstall() {
    ENV.with(|slot| slot.set(ptr::null_mut()));
}

/// Whether this thread has an environment installed, i.e. whether it can reach
/// JavaScript at all.
pub fn is_installed() -> bool {
    ENV.with(|slot| !slot.get().is_null())
}

/// The environment for this thread, or `None` if [`install`] was never called.
pub(crate) fn try_env() -> Option<sys::napi_env> {
    ENV.with(|slot| {
        let env = slot.get();
        (!env.is_null()).then_some(env)
    })
}

/// The environment for this thread.
///
/// # Panics
///
/// If no environment was installed. That is a wiring mistake in the addon rather
/// than a runtime condition worth reporting through a `Result`: without an
/// environment there is no JavaScript to talk to, so no `wgpu` call can proceed.
pub(crate) fn env() -> sys::napi_env {
    try_env().expect(
        "wgpu-napi-web: no Node-API environment installed on this thread — \
         call wgpu_napi_web::install(env) from the addon's module init (or from a \
         #[napi] function) before using wgpu, and once per thread that uses it",
    )
}

/// Runs `f` inside a Node-API handle scope.
///
/// Every `napi_value` produced inside the scope is released when it closes, which
/// is why `f` may only return values that no longer borrow the scope: a
/// [`JsValue`] holds its own `napi_ref`, Rust data holds nothing. Without this,
/// the intermediate values of a render — every property key, every descriptor —
/// would accumulate in whichever scope the addon happened to be called in.
pub(crate) fn scope<R>(f: impl FnOnce(sys::napi_env) -> Result<R, JsValue>) -> Result<R, JsValue> {
    let env = env();
    let mut handle_scope = ptr::null_mut();
    // SAFETY: `env` is a live environment for this thread.
    let status = unsafe { sys::napi_open_handle_scope(env, &mut handle_scope) };
    if status != sys::Status::napi_ok {
        return Err(JsValue::from_str(
            "wgpu-napi-web: napi_open_handle_scope failed",
        ));
    }
    let result = f(env);
    // SAFETY: `handle_scope` is the scope just opened on this env, and scopes are
    // closed in reverse order of opening because `f` can only nest.
    let close = unsafe { sys::napi_close_handle_scope(env, handle_scope) };
    debug_assert_eq!(
        close,
        sys::Status::napi_ok,
        "wgpu-napi-web: napi_close_handle_scope failed"
    );
    result
}

/// Turns a Node-API status into a `Result`, attaching the pending JavaScript
/// exception when there is one.
///
/// `operation` names the call site, because a bare status code says nothing about
/// what was being attempted.
pub(crate) fn check(status: sys::napi_status, operation: &str) -> Result<(), JsValue> {
    if status == sys::Status::napi_ok {
        return Ok(());
    }
    if status == sys::Status::napi_pending_exception {
        if let Some(exception) = take_exception() {
            return Err(exception);
        }
    }
    Err(JsValue::from_string(alloc::format!(
        "wgpu-napi-web: {operation} failed with Node-API status {status}"
    )))
}

/// Clears and returns the pending JavaScript exception, if any.
///
/// A pending exception poisons every later Node-API call on the environment, so
/// it has to be taken at the point of failure rather than left for whoever calls
/// next.
pub(crate) fn take_exception() -> Option<JsValue> {
    let env = try_env()?;
    let mut pending = false;
    // SAFETY: `env` is live; `pending` is a valid out pointer.
    unsafe {
        if sys::napi_is_exception_pending(env, &mut pending) != sys::Status::napi_ok || !pending {
            return None;
        }
        let mut value = ptr::null_mut();
        if sys::napi_get_and_clear_last_exception(env, &mut value) != sys::Status::napi_ok {
            return None;
        }
        Some(JsValue::from_napi(env, value))
    }
}
