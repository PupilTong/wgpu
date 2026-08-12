//! Rust functions callable from JavaScript.
//!
//! wasm-bindgen's `Closure` puts the Rust closure in a table and hands JavaScript
//! a shim that indexes into it, with the `Closure` owning the entry: dropping it
//! invalidates the JS side. Node-API works the other way round — `napi_create_function`
//! makes a real JS function carrying an opaque data pointer, and a finalizer
//! attached to that function frees the data when the *garbage collector* is done
//! with it.
//!
//! That inverts one detail of the wasm-bindgen contract, in the safe direction: a
//! [`Closure`] here can be dropped while JavaScript still holds the function, and
//! the function keeps working until JS lets go. [`Closure::forget`] therefore has
//! nothing to leak and is a no-op — code that calls it (as `wgpu` does for the
//! device-lost and uncaptured-error handlers) is correct either way, and code that
//! drops early no longer breaks.

use alloc::boxed::Box;
use core::ffi::c_void;
use core::marker::PhantomData;
use core::ptr;

use napi_sys as sys;

use crate::napi::convert::{AsJs, FromJs};
use crate::napi::env;
use crate::napi::value::{JsCast, JsError, JsValue};

/// A Rust function that JavaScript can call, with a lifetime bound on what it
/// borrows.
///
/// `T` is the `dyn FnMut(..) -> ..` shape being exposed, matching
/// `wasm_bindgen::closure::ScopedClosure`.
pub struct ScopedClosure<'a, T: ?Sized> {
    function: JsValue,
    // `Box<T>` rather than `T` because `T` is unsized.
    _closure: PhantomData<Box<T>>,
    _lifetime: PhantomData<&'a ()>,
}

/// A `'static` [`ScopedClosure`], which is the only kind this shim can build:
/// JavaScript decides when the function dies, so it must not borrow.
pub type Closure<T> = ScopedClosure<'static, T>;

impl<T: ?Sized> core::fmt::Debug for ScopedClosure<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Closure").finish_non_exhaustive()
    }
}

impl<T: ?Sized> AsRef<JsValue> for ScopedClosure<'_, T> {
    fn as_ref(&self) -> &JsValue {
        &self.function
    }
}

impl<T: ?Sized> From<ScopedClosure<'_, T>> for JsValue {
    fn from(closure: ScopedClosure<'_, T>) -> Self {
        closure.function
    }
}

impl<T: ?Sized> ScopedClosure<'_, T> {
    /// The JavaScript function for this closure.
    pub fn as_js_value(&self) -> &JsValue {
        &self.function
    }

    /// Kept for parity with wasm-bindgen, where it leaks the closure so JavaScript
    /// can keep calling it. Here the JS function already owns its Rust state, so
    /// there is nothing to leak: this only drops the Rust-side handle.
    pub fn forget(self) {
        drop(self);
    }
}

impl<T: ?Sized + IntoJsFunction> Closure<T> {
    /// Exposes a boxed `FnMut` to JavaScript.
    pub fn wrap(closure: Box<T>) -> Self {
        Self {
            function: closure.into_js_function(),
            _closure: PhantomData,
            _lifetime: PhantomData,
        }
    }
}

impl<T: ?Sized> Closure<T> {
    /// Exposes a `FnOnce` to JavaScript.
    ///
    /// Calling it a second time panics, as it does with wasm-bindgen: the JS side
    /// has no way to express "already consumed".
    pub fn once<F: IntoJsFunctionOnce<T>>(closure: F) -> Self {
        Self {
            function: closure.into_js_function_once(),
            _closure: PhantomData,
            _lifetime: PhantomData,
        }
    }
}

/// The `dyn FnMut(..) -> ..` shapes that can be handed to JavaScript.
///
/// Implemented for the arities `wgpu` uses. Each impl monomorphises a trampoline
/// that converts the JavaScript arguments and the Rust return value.
pub trait IntoJsFunction {
    /// Creates the JavaScript function, transferring ownership of the closure to it.
    fn into_js_function(self: Box<Self>) -> JsValue;
}

/// Turns a `FnOnce` into the JavaScript function for the `dyn FnMut` shape `T`.
pub trait IntoJsFunctionOnce<T: ?Sized> {
    /// Creates the JavaScript function, transferring ownership of the closure to it.
    fn into_js_function_once(self) -> JsValue;
}

/// What a Rust closure may return to JavaScript.
///
/// A `Result` is how the generated bindings spell "this callback may throw"; `Err`
/// becomes a real JavaScript exception raised from the callback.
pub trait ClosureReturn {
    /// The value to return to JavaScript, or the value to throw.
    fn into_js_result(self) -> Result<JsValue, JsValue>;
}

impl ClosureReturn for () {
    fn into_js_result(self) -> Result<JsValue, JsValue> {
        Ok(JsValue::UNDEFINED)
    }
}

impl ClosureReturn for JsValue {
    fn into_js_result(self) -> Result<JsValue, JsValue> {
        Ok(self)
    }
}

impl<T: AsJs, E: Into<JsValue>> ClosureReturn for Result<T, E> {
    fn into_js_result(self) -> Result<JsValue, JsValue> {
        match self {
            Ok(value) => Ok(value.as_js()),
            Err(error) => Err(error.into()),
        }
    }
}

impl ClosureReturn for JsError {
    fn into_js_result(self) -> Result<JsValue, JsValue> {
        Ok(self.into())
    }
}

macro_rules! closure_impls {
    ($( ($arity:literal $(, $argument:ident as $binding:ident)*) ),* $(,)?) => {
        $(
            impl<$($argument,)* R> IntoJsFunction for dyn FnMut($($argument),*) -> R
            where
                $($argument: FromJs + 'static,)*
                R: ClosureReturn + 'static,
            {
                fn into_js_function(self: Box<Self>) -> JsValue {
                    /// Reads the JavaScript arguments, runs the closure, and returns or
                    /// throws its result.
                    unsafe extern "C" fn trampoline<$($argument,)* R>(
                        env: sys::napi_env,
                        info: sys::napi_callback_info,
                    ) -> sys::napi_value
                    where
                        $($argument: FromJs + 'static,)*
                        R: ClosureReturn + 'static,
                    {
                        #[allow(
                            clippy::zero_repeat_side_effects,
                            reason = "the zero-argument arity produces an empty array, and \
                                      the repeated expression is a null pointer — the lint \
                                      does not see through the inline const"
                        )]
                        let mut arguments = [const { ptr::null_mut() }; $arity];
                        let mut argc = $arity;
                        let mut data = ptr::null_mut();
                        // SAFETY: `info` is the callback info Node-API just handed us;
                        // `arguments` has room for `argc` values.
                        let status = sys::napi_get_cb_info(
                            env,
                            info,
                            &mut argc,
                            arguments.as_mut_ptr(),
                            ptr::null_mut(),
                            &mut data,
                        );
                        if status != sys::Status::napi_ok || data.is_null() {
                            log::error!("napi-rs-webgpu: callback invoked without its closure");
                            return ptr::null_mut();
                        }
                        // Arguments JavaScript did not pass are `undefined`, which is what
                        // a null `napi_value` converts to, so a short call is not an error.
                        #[allow(
                            unused_mut,
                            unused_variables,
                            reason = "a zero-argument closure reads no arguments"
                        )]
                        let mut arguments = arguments.into_iter();
                        // SAFETY: `data` is the `Box<Box<dyn FnMut…>>` leaked in
                        // `into_js_function` for this very function object, and the
                        // finalizer that frees it cannot run while the function is
                        // executing.
                        let closure = &mut *data.cast::<Box<dyn FnMut($($argument),*) -> R>>();
                        let result = closure(
                            $({
                                let raw = arguments.next().unwrap_or(ptr::null_mut());
                                // SAFETY: `raw` is valid in this callback's handle scope.
                                <$argument as FromJs>::from_js(JsValue::from_napi(env, raw))
                            }),*
                        );
                        match result.into_js_result() {
                            Ok(value) => match value.to_napi(env) {
                                Ok(raw) => raw,
                                Err(error) => {
                                    log::error!("napi-rs-webgpu: callback result unusable: {error}");
                                    ptr::null_mut()
                                }
                            },
                            Err(error) => {
                                throw(env, &error);
                                ptr::null_mut()
                            }
                        }
                    }

                    // SAFETY: the trampoline matches this closure's type parameters, and
                    // the finalizer frees exactly the box created here.
                    unsafe {
                        create_function(
                            Box::into_raw(Box::new(self)).cast::<c_void>(),
                            Some(trampoline::<$($argument,)* R>),
                            Some(finalize::<Box<dyn FnMut($($argument),*) -> R>>),
                        )
                    }
                }
            }

            impl<F, $($argument,)* R> IntoJsFunctionOnce<dyn FnMut($($argument),*) -> R> for F
            where
                F: FnOnce($($argument),*) -> R + 'static,
                $($argument: FromJs + 'static,)*
                R: ClosureReturn + 'static,
            {
                fn into_js_function_once(self) -> JsValue {
                    let mut closure = Some(self);
                    let boxed: Box<dyn FnMut($($argument),*) -> R> = Box::new(
                        move |$($binding: $argument),*| {
                            let closure = closure
                                .take()
                                .expect("napi-rs-webgpu: a `Closure::once` was called twice");
                            closure($($binding),*)
                        },
                    );
                    boxed.into_js_function()
                }
            }
        )*
    };
}

closure_impls!(
    (0),
    (1, A0 as a0),
    (2, A0 as a0, A1 as a1),
    (3, A0 as a0, A1 as a1, A2 as a2),
);

/// Creates a JavaScript function that calls `trampoline` with `data`, freeing
/// `data` through `finalizer` once JavaScript has collected the function.
///
/// # Safety
///
/// `data` must be a pointer that `finalizer` can free exactly once, and
/// `trampoline` must be the trampoline matching its type.
unsafe fn create_function(
    data: *mut c_void,
    trampoline: sys::napi_callback,
    finalizer: sys::napi_finalize,
) -> JsValue {
    let created = env::scope(|env| {
        let mut function = ptr::null_mut();
        env::check(
            sys::napi_create_function(
                env,
                c"wgpu_rust_closure".as_ptr(),
                17,
                trampoline,
                data,
                &mut function,
            ),
            "napi_create_function",
        )?;
        // Without this the closure would leak: nothing else ever frees `data`.
        env::check(
            sys::napi_add_finalizer(
                env,
                function,
                data,
                finalizer,
                ptr::null_mut(),
                ptr::null_mut(),
            ),
            "napi_add_finalizer",
        )?;
        Ok(JsValue::from_napi(env, function))
    });
    crate::napi::rt::unwrap_js(created, "creating a JavaScript function for a Rust closure")
}

/// Frees the closure a function was created with.
///
/// # Safety
///
/// Node-API calls this once per function, with the `data` pointer that function
/// was created with; that pointer is a `Box<T>` leaked by `into_js_function`.
unsafe extern "C" fn finalize<T>(_env: sys::napi_env, data: *mut c_void, _hint: *mut c_void) {
    if data.is_null() {
        return;
    }
    drop(Box::from_raw(data.cast::<T>()));
}

/// Raises `error` as a JavaScript exception from the current callback.
///
/// # Safety
///
/// Must be called from inside a Node-API callback on `env`, and the callback must
/// return immediately afterwards.
unsafe fn throw(env: sys::napi_env, error: &JsValue) {
    let raw = match error.to_napi(env) {
        Ok(raw) => raw,
        Err(_) => return,
    };
    let status = sys::napi_throw(env, raw);
    debug_assert_eq!(
        status,
        sys::Status::napi_ok,
        "napi-rs-webgpu: napi_throw failed"
    );
}

/// Creates a JavaScript function calling a plain Rust function, with no state to
/// free. Used for the microtask drain in [`crate::napi::futures`].
pub(crate) fn stateless_function(trampoline: sys::napi_callback) -> JsValue {
    // SAFETY: a null data pointer with no finalizer owns nothing, so there is
    // nothing to free and the trampoline must not read `data`.
    unsafe {
        let created = env::scope(|env| {
            let mut function = ptr::null_mut();
            env::check(
                sys::napi_create_function(
                    env,
                    c"wgpu_rust_task".as_ptr(),
                    14,
                    trampoline,
                    ptr::null_mut(),
                    &mut function,
                ),
                "napi_create_function",
            )?;
            Ok(JsValue::from_napi(env, function))
        });
        crate::napi::rt::unwrap_js(created, "creating a JavaScript function for the task queue")
    }
}

/// Casts a closure's function to a bindings type, as `wgpu` does when installing
/// an event handler (`closure.as_ref().unchecked_ref()`).
impl<T: ?Sized> ScopedClosure<'_, T> {
    /// The function, viewed as `U`.
    pub fn unchecked_ref<U: JsCast>(&self) -> &U {
        U::unchecked_from_js_ref(&self.function)
    }
}
