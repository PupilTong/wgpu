//! The operations `#[wasm_bindgen]` lowers to.
//!
//! wasm-bindgen turns each declaration in an `extern "C"` block into a dedicated
//! wasm import, resolved by generated JS glue. There is no glue here, so each
//! declaration becomes a Node-API property read, property write, method call or
//! construction against the live object — the same JavaScript operations the glue
//! would have performed, just late-bound by name.
//!
//! Property names arrive as C string literals (`c"requestDevice"`) so the calls
//! below need no allocation: Node-API's `*_named_property` entry points want a
//! NUL-terminated name, and the macro has the name at compile time.

use alloc::vec::Vec;
use core::ffi::CStr;
use core::ptr;

use napi_sys as sys;

use crate::napi::env;
use crate::napi::value::{JsCast, JsValue};

/// Reads `target[name]`.
pub fn get(target: &JsValue, name: &CStr) -> Result<JsValue, JsValue> {
    env::scope(|env| {
        // SAFETY: inside a handle scope on `env`; `name` is NUL-terminated.
        unsafe {
            let object = target.to_napi(env)?;
            let mut out = ptr::null_mut();
            env::check(
                sys::napi_get_named_property(env, object, name.as_ptr(), &mut out),
                "napi_get_named_property",
            )?;
            Ok(JsValue::from_napi(env, out))
        }
    })
}

/// Writes `target[name] = value`.
pub fn set(target: &JsValue, name: &CStr, value: &JsValue) -> Result<(), JsValue> {
    env::scope(|env| {
        // SAFETY: inside a handle scope on `env`; `name` is NUL-terminated.
        unsafe {
            let object = target.to_napi(env)?;
            let value = value.to_napi(env)?;
            env::check(
                sys::napi_set_named_property(env, object, name.as_ptr(), value),
                "napi_set_named_property",
            )
        }
    })
}

/// Calls `target[name](..args)`.
pub fn call_method(target: &JsValue, name: &CStr, args: &[JsValue]) -> Result<JsValue, JsValue> {
    env::scope(|env| {
        // SAFETY: inside a handle scope on `env`; `name` is NUL-terminated and
        // `arguments` is a contiguous run of `argc` live values.
        unsafe {
            let receiver = target.to_napi(env)?;
            let mut function = ptr::null_mut();
            env::check(
                sys::napi_get_named_property(env, receiver, name.as_ptr(), &mut function),
                "napi_get_named_property",
            )?;
            let arguments = to_napi_all(env, args)?;
            let mut out = ptr::null_mut();
            env::check(
                sys::napi_call_function(
                    env,
                    receiver,
                    function,
                    arguments.len(),
                    arguments.as_ptr(),
                    &mut out,
                ),
                "napi_call_function",
            )?;
            Ok(JsValue::from_napi(env, out))
        }
    })
}

/// Calls `function.call(this, ..args)`.
pub fn call(function: &JsValue, this: &JsValue, args: &[JsValue]) -> Result<JsValue, JsValue> {
    env::scope(|env| {
        // SAFETY: inside a handle scope on `env`; `arguments` is a contiguous run
        // of `argc` live values.
        unsafe {
            let function = function.to_napi(env)?;
            let receiver = this.to_napi(env)?;
            let arguments = to_napi_all(env, args)?;
            let mut out = ptr::null_mut();
            env::check(
                sys::napi_call_function(
                    env,
                    receiver,
                    function,
                    arguments.len(),
                    arguments.as_ptr(),
                    &mut out,
                ),
                "napi_call_function",
            )?;
            Ok(JsValue::from_napi(env, out))
        }
    })
}

/// Calls `new globalThis[class](..args)`.
pub fn construct(class: &CStr, args: &[JsValue]) -> Result<JsValue, JsValue> {
    env::scope(|env| {
        // SAFETY: inside a handle scope on `env`; `class` is NUL-terminated.
        unsafe {
            let constructor = global_property(env, class)?;
            let arguments = to_napi_all(env, args)?;
            let mut out = ptr::null_mut();
            env::check(
                sys::napi_new_instance(
                    env,
                    constructor,
                    arguments.len(),
                    arguments.as_ptr(),
                    &mut out,
                ),
                "napi_new_instance",
            )?;
            Ok(JsValue::from_napi(env, out))
        }
    })
}

/// `globalThis[name]`.
pub fn global(name: &CStr) -> Result<JsValue, JsValue> {
    env::scope(|env| {
        // SAFETY: inside a handle scope on `env`; `name` is NUL-terminated.
        unsafe {
            let value = global_property(env, name)?;
            Ok(JsValue::from_napi(env, value))
        }
    })
}

/// `value instanceof globalThis[class]`.
///
/// `false` when the class is not defined at all, which is how a feature-detection
/// cast (`dyn_into::<GpuAdapter>()` in a browser without WebGPU) should behave.
pub fn instance_of(value: &JsValue, class: &CStr) -> bool {
    env::scope(|env| {
        // SAFETY: inside a handle scope on `env`; `class` is NUL-terminated.
        unsafe {
            let constructor = global_property(env, class)?;
            let mut kind = 0;
            env::check(sys::napi_typeof(env, constructor, &mut kind), "napi_typeof")?;
            if kind != sys::ValueType::napi_function {
                return Ok(false);
            }
            let value = value.to_napi(env)?;
            let mut result = false;
            env::check(
                sys::napi_instanceof(env, value, constructor, &mut result),
                "napi_instanceof",
            )?;
            Ok(result)
        }
    })
    .unwrap_or(false)
}

/// A new, empty `{}`.
pub fn new_object() -> JsValue {
    unwrap_js(
        env::scope(|env| {
            // SAFETY: inside a handle scope on `env`.
            unsafe {
                let mut out = ptr::null_mut();
                env::check(sys::napi_create_object(env, &mut out), "napi_create_object")?;
                Ok(JsValue::from_napi(env, out))
            }
        }),
        "creating a JavaScript object",
    )
}

/// A new array holding `items`.
pub fn array_from(items: &[JsValue]) -> JsValue {
    unwrap_js(
        env::scope(|env| {
            // SAFETY: inside a handle scope on `env`; indices are within the length
            // the array was created with.
            unsafe {
                let mut array = ptr::null_mut();
                env::check(
                    sys::napi_create_array_with_length(env, items.len(), &mut array),
                    "napi_create_array_with_length",
                )?;
                for (index, item) in items.iter().enumerate() {
                    let value = item.to_napi(env)?;
                    env::check(
                        sys::napi_set_element(env, array, index as u32, value),
                        "napi_set_element",
                    )?;
                }
                Ok(JsValue::from_napi(env, array))
            }
        }),
        "creating a JavaScript array",
    )
}

/// The elements of an array-like value, or an empty vector if it is not one.
pub fn array_items(value: &JsValue) -> Vec<JsValue> {
    env::scope(|env| {
        // SAFETY: inside a handle scope on `env`.
        unsafe {
            let array = value.to_napi(env)?;
            let mut is_array = false;
            env::check(
                sys::napi_is_array(env, array, &mut is_array),
                "napi_is_array",
            )?;
            if !is_array {
                return Ok(Vec::new());
            }
            let mut length = 0;
            env::check(
                sys::napi_get_array_length(env, array, &mut length),
                "napi_get_array_length",
            )?;
            let mut items = Vec::with_capacity(length as usize);
            for index in 0..length {
                let mut element = ptr::null_mut();
                env::check(
                    sys::napi_get_element(env, array, index, &mut element),
                    "napi_get_element",
                )?;
                items.push(JsValue::from_napi(env, element));
            }
            Ok(items)
        }
    })
    .unwrap_or_default()
}

/// Reads `target[index]`.
pub fn get_index(target: &JsValue, index: u32) -> Result<JsValue, JsValue> {
    env::scope(|env| {
        // SAFETY: inside a handle scope on `env`.
        unsafe {
            let object = target.to_napi(env)?;
            let mut out = ptr::null_mut();
            env::check(
                sys::napi_get_element(env, object, index, &mut out),
                "napi_get_element",
            )?;
            Ok(JsValue::from_napi(env, out))
        }
    })
}

/// A JavaScript `Error` with this message.
pub fn error(message: &str) -> JsValue {
    unwrap_js(
        env::scope(|env| {
            // SAFETY: inside a handle scope on `env`.
            unsafe {
                let mut text = ptr::null_mut();
                env::check(
                    sys::napi_create_string_utf8(
                        env,
                        message.as_ptr().cast(),
                        message.len() as isize,
                        &mut text,
                    ),
                    "napi_create_string_utf8",
                )?;
                let mut out = ptr::null_mut();
                env::check(
                    sys::napi_create_error(env, ptr::null_mut(), text, &mut out),
                    "napi_create_error",
                )?;
                Ok(JsValue::from_napi(env, out))
            }
        }),
        "creating a JavaScript Error",
    )
}

/// The result of an operation declared without `catch`.
///
/// wasm-bindgen lets the JavaScript exception escape through the import boundary,
/// which unwinds or traps the module. Node-API instead leaves it pending on the
/// environment, where it would poison every later call, so the exception is taken
/// and reported here. Under `panic = "abort"` — every WASI target Rust ships —
/// this ends the module, which is the same outcome by a clearer route.
#[track_caller]
pub fn unwrap_js<T>(result: Result<T, JsValue>, operation: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            log::error!("napi-rs-webgpu: {operation} threw: {error}");
            panic!("napi-rs-webgpu: {operation} threw: {error}");
        }
    }
}

/// Materialises every argument in the current handle scope.
///
/// # Safety
///
/// The returned values must not escape the current handle scope.
unsafe fn to_napi_all(
    env: sys::napi_env,
    args: &[JsValue],
) -> Result<Vec<sys::napi_value>, JsValue> {
    args.iter().map(|arg| arg.to_napi(env)).collect()
}

/// `globalThis[name]`, valid in the current handle scope.
///
/// # Safety
///
/// Must be called inside a handle scope on `env`, and the result must not escape it.
unsafe fn global_property(env: sys::napi_env, name: &CStr) -> Result<sys::napi_value, JsValue> {
    let mut global = ptr::null_mut();
    env::check(sys::napi_get_global(env, &mut global), "napi_get_global")?;
    let mut out = ptr::null_mut();
    env::check(
        sys::napi_get_named_property(env, global, name.as_ptr(), &mut out),
        "napi_get_named_property",
    )?;
    Ok(out)
}

/// Casts a JS value to a bindings type, as the generated code does after every
/// property read and call.
pub fn cast<T: JsCast>(value: JsValue) -> T {
    crate::napi::convert::cast(value)
}
