//! Moving values across the Rust/JavaScript boundary.
//!
//! wasm-bindgen has `IntoWasmAbi`/`FromWasmAbi` for this, generated per type and
//! specialised per ABI. Here there is one ABI — [`JsValue`] — so two traits are
//! enough: [`AsJs`] to hand a Rust value to JavaScript, [`FromJs`] to take one
//! back. `#[wasm_bindgen]` emits both for every type it declares, and the impls
//! below cover the primitives, slices and options that appear in the generated
//! signatures.
//!
//! [`FromJs`] is deliberately infallible: it mirrors wasm-bindgen's generated
//! bindings, which reinterpret whatever JavaScript returned as the declared type
//! without a check. A getter declared to return a number that returns a string
//! instead yields a default, not an error — same as the ABI it replaces.

use alloc::string::String;
use alloc::vec::Vec;

use crate::value::{JsCast, JsValue};

/// Produces the JavaScript value for `self`.
pub trait AsJs {
    /// The JavaScript form of this value.
    fn as_js(&self) -> JsValue;
}

/// Reinterprets a JavaScript value as `Self`.
pub trait FromJs {
    /// Converts `value`, falling back to a default if it is not the expected shape.
    fn from_js(value: JsValue) -> Self;
}

impl<T: AsJs + ?Sized> AsJs for &T {
    #[inline]
    fn as_js(&self) -> JsValue {
        T::as_js(self)
    }
}

impl<T: AsJs + ?Sized> AsJs for &mut T {
    #[inline]
    fn as_js(&self) -> JsValue {
        T::as_js(self)
    }
}

impl AsJs for JsValue {
    #[inline]
    fn as_js(&self) -> JsValue {
        self.clone()
    }
}

impl FromJs for JsValue {
    #[inline]
    fn from_js(value: JsValue) -> Self {
        value
    }
}

impl AsJs for bool {
    #[inline]
    fn as_js(&self) -> JsValue {
        JsValue::from_bool(*self)
    }
}

impl FromJs for bool {
    #[inline]
    fn from_js(value: JsValue) -> Self {
        value.as_bool().unwrap_or_else(|| value.is_truthy())
    }
}

macro_rules! number_conversions {
    ($($ty:ty),* $(,)?) => {
        $(
            impl AsJs for $ty {
                #[inline]
                #[allow(
                    clippy::cast_lossless,
                    clippy::cast_precision_loss,
                    reason = "JavaScript numbers are f64; the generated bindings only \
                              carry values that WebGPU already limits to that range"
                )]
                fn as_js(&self) -> JsValue {
                    JsValue::from_f64(*self as f64)
                }
            }

            impl FromJs for $ty {
                #[inline]
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_possible_wrap,
                    reason = "matches wasm-bindgen's ABI, which truncates the JS number \
                              to the declared Rust width"
                )]
                fn from_js(value: JsValue) -> Self {
                    value.as_f64().unwrap_or(0.0) as $ty
                }
            }
        )*
    };
}

number_conversions!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);

impl AsJs for str {
    #[inline]
    fn as_js(&self) -> JsValue {
        JsValue::from_str(self)
    }
}

impl AsJs for String {
    #[inline]
    fn as_js(&self) -> JsValue {
        JsValue::from_str(self)
    }
}

impl FromJs for String {
    #[inline]
    fn from_js(value: JsValue) -> Self {
        value.as_string().unwrap_or_default()
    }
}

impl AsJs for () {
    #[inline]
    fn as_js(&self) -> JsValue {
        JsValue::UNDEFINED
    }
}

impl FromJs for () {
    #[inline]
    fn from_js(_value: JsValue) -> Self {}
}

/// `None` becomes `undefined`, matching how wasm-bindgen passes optional
/// arguments and how WebGPU dictionaries treat absent fields.
impl<T: AsJs> AsJs for Option<T> {
    #[inline]
    fn as_js(&self) -> JsValue {
        match self {
            Some(value) => value.as_js(),
            None => JsValue::UNDEFINED,
        }
    }
}

/// `undefined` and `null` both become `None`; wasm-bindgen's optional getters
/// accept either.
impl<T: FromJs> FromJs for Option<T> {
    #[inline]
    fn from_js(value: JsValue) -> Self {
        if value.is_undefined() || value.is_null() {
            None
        } else {
            Some(T::from_js(value))
        }
    }
}

/// Slices become JavaScript arrays, which is what the generated dictionary
/// setters (`set_entries(&[GpuBindGroupEntry])`) expect.
impl<T: AsJs> AsJs for [T] {
    fn as_js(&self) -> JsValue {
        let items: Vec<JsValue> = self.iter().map(AsJs::as_js).collect();
        crate::rt::array_from(&items)
    }
}

impl<T: AsJs, const N: usize> AsJs for [T; N] {
    #[inline]
    fn as_js(&self) -> JsValue {
        self.as_slice().as_js()
    }
}

impl<T: AsJs> AsJs for Vec<T> {
    #[inline]
    fn as_js(&self) -> JsValue {
        self.as_slice().as_js()
    }
}

impl<T: FromJs> FromJs for Vec<T> {
    fn from_js(value: JsValue) -> Self {
        crate::rt::array_items(&value)
            .into_iter()
            .map(T::from_js)
            .collect()
    }
}

/// Convenience for the generated code: casting a JS value to a bindings type is
/// always the unchecked reinterpretation, since JavaScript already guaranteed the
/// shape by returning it from the API that declares it.
pub(crate) fn cast<T: JsCast>(value: JsValue) -> T {
    T::unchecked_from_js(value)
}
