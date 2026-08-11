//! [`JsValue`] and the casting traits built on it.
//!
//! wasm-bindgen's `JsValue` is an index into a JS-side table that the generated
//! glue keeps alive. Node-API has no such table: a `napi_value` is only valid
//! inside the handle scope that produced it, and outliving a scope requires an
//! explicit `napi_ref`. So this `JsValue` is one of two things:
//!
//! * a primitive (`undefined`, `null`, boolean, number, string) held **in Rust**,
//!   which needs no environment to exist, no reference to stay alive, and no JS
//!   round trip to inspect; or
//! * a reference-counted `napi_ref` to a JS object, function or symbol, released
//!   when the last clone drops.
//!
//! Keeping primitives on the Rust side is what makes `JsValue::from_str("label")`
//! as cheap as it looks, and it is why property keys — created by the thousand
//! per frame — never touch the JS heap until the call that uses them.

use alloc::rc::Rc;
use alloc::string::String;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ptr;

use napi_sys as sys;

use crate::napi::env;

/// A JavaScript value.
#[derive(Clone)]
#[repr(transparent)]
pub struct JsValue(pub(crate) Repr);

#[derive(Clone)]
pub(crate) enum Repr {
    Undefined,
    Null,
    Bool(bool),
    Number(f64),
    /// Strings live in Rust: they are immutable in JS anyway, so a copy is
    /// indistinguishable from a reference, and it keeps keys environment-free.
    Str(Rc<str>),
    /// Objects, functions and symbols, which have identity and so must stay in JS.
    Handle(Rc<Handle>),
}

/// An owned `napi_ref`, released when the last [`JsValue`] clone drops.
pub(crate) struct Handle {
    env: sys::napi_env,
    reference: sys::napi_ref,
}

impl Handle {
    /// The value behind this reference, valid in the current handle scope.
    ///
    /// # Safety
    ///
    /// The returned `napi_value` must not escape the enclosing handle scope.
    unsafe fn value(&self) -> Result<sys::napi_value, JsValue> {
        let mut out = ptr::null_mut();
        env::check(
            sys::napi_get_reference_value(self.env, self.reference, &mut out),
            "napi_get_reference_value",
        )?;
        if out.is_null() {
            // A live reference whose value is gone means the environment is being
            // torn down; there is nothing useful to hand back.
            return Err(JsValue::from_str(
                "napi-rs-webgpu: JavaScript reference is no longer valid",
            ));
        }
        Ok(out)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // Only release the reference if the environment this handle belongs to is
        // still the installed one. After `uninstall` (environment teardown) the
        // reference cannot be deleted, and the garbage collector owns the value.
        if env::try_env() != Some(self.env) {
            return;
        }
        // SAFETY: `reference` was created by `napi_create_reference` on `self.env`,
        // is owned by this `Handle`, and is deleted exactly once.
        unsafe {
            let status = sys::napi_delete_reference(self.env, self.reference);
            debug_assert_eq!(
                status,
                sys::Status::napi_ok,
                "napi-rs-webgpu: napi_delete_reference failed"
            );
        }
    }
}

impl JsValue {
    /// The JavaScript `undefined` value.
    pub const UNDEFINED: Self = Self(Repr::Undefined);
    /// The JavaScript `null` value.
    pub const NULL: Self = Self(Repr::Null);
    /// JavaScript `true`.
    pub const TRUE: Self = Self(Repr::Bool(true));
    /// JavaScript `false`.
    pub const FALSE: Self = Self(Repr::Bool(false));

    /// `undefined`.
    #[inline]
    pub const fn undefined() -> Self {
        Self::UNDEFINED
    }

    /// `null`.
    #[inline]
    pub const fn null() -> Self {
        Self::NULL
    }

    /// A JavaScript string with the contents of `value`.
    #[inline]
    #[allow(
        clippy::should_implement_trait,
        reason = "the name mirrors `wasm_bindgen::JsValue::from_str`, which the \
                  bindings call by name; `FromStr` would also make it fallible"
    )]
    pub fn from_str(value: &str) -> Self {
        Self(Repr::Str(Rc::from(value)))
    }

    /// A JavaScript string taking ownership of `value`.
    #[inline]
    pub fn from_string(value: String) -> Self {
        Self(Repr::Str(Rc::from(value)))
    }

    /// A JavaScript number.
    #[inline]
    pub const fn from_f64(value: f64) -> Self {
        Self(Repr::Number(value))
    }

    /// A JavaScript boolean.
    #[inline]
    pub const fn from_bool(value: bool) -> Self {
        Self(Repr::Bool(value))
    }

    /// Whether this is `undefined`.
    #[inline]
    pub fn is_undefined(&self) -> bool {
        matches!(self.0, Repr::Undefined)
    }

    /// Whether this is `null`.
    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self.0, Repr::Null)
    }

    /// Whether this is a string.
    #[inline]
    pub fn is_string(&self) -> bool {
        matches!(self.0, Repr::Str(_))
    }

    /// Whether this is a JavaScript object (which excludes `null`, matching
    /// `wasm_bindgen::JsValue::is_object`).
    pub fn is_object(&self) -> bool {
        self.type_of() == Some(sys::ValueType::napi_object)
    }

    /// Whether this is a JavaScript function.
    pub fn is_function(&self) -> bool {
        self.type_of() == Some(sys::ValueType::napi_function)
    }

    /// Whether this is a symbol.
    pub fn is_symbol(&self) -> bool {
        self.type_of() == Some(sys::ValueType::napi_symbol)
    }

    /// Whether this value is truthy by JavaScript's rules.
    pub fn is_truthy(&self) -> bool {
        match &self.0 {
            Repr::Undefined | Repr::Null => false,
            Repr::Bool(value) => *value,
            Repr::Number(value) => *value != 0.0 && !value.is_nan(),
            Repr::Str(value) => !value.is_empty(),
            Repr::Handle(_) => true,
        }
    }

    /// Whether this value is falsy by JavaScript's rules.
    #[inline]
    pub fn is_falsy(&self) -> bool {
        !self.is_truthy()
    }

    /// The string contents, if this is a string.
    pub fn as_string(&self) -> Option<String> {
        match &self.0 {
            Repr::Str(value) => Some(String::from(&**value)),
            _ => None,
        }
    }

    /// The numeric value, if this is a number.
    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        match self.0 {
            Repr::Number(value) => Some(value),
            _ => None,
        }
    }

    /// The boolean value, if this is a boolean.
    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        match self.0 {
            Repr::Bool(value) => Some(value),
            _ => None,
        }
    }

    /// The `typeof` of this value as a Node-API value type, or `None` for a value
    /// that needs an environment we do not have.
    fn type_of(&self) -> Option<sys::napi_valuetype> {
        match &self.0 {
            Repr::Undefined => Some(sys::ValueType::napi_undefined),
            Repr::Null => Some(sys::ValueType::napi_null),
            Repr::Bool(_) => Some(sys::ValueType::napi_boolean),
            Repr::Number(_) => Some(sys::ValueType::napi_number),
            Repr::Str(_) => Some(sys::ValueType::napi_string),
            Repr::Handle(handle) => {
                let mut kind = 0;
                // SAFETY: the handle's env is live while the handle is.
                unsafe {
                    let value = handle.value().ok()?;
                    (sys::napi_typeof(handle.env, value, &mut kind) == sys::Status::napi_ok)
                        .then_some(kind)
                }
            }
        }
    }

    /// Adopts `value` from the current handle scope, taking a reference if it needs
    /// one to outlive the scope.
    ///
    /// # Safety
    ///
    /// `value` must be a valid `napi_value` in `env`'s current handle scope.
    pub(crate) unsafe fn from_napi(env: sys::napi_env, value: sys::napi_value) -> Self {
        if value.is_null() {
            return Self::UNDEFINED;
        }
        let mut kind = 0;
        if sys::napi_typeof(env, value, &mut kind) != sys::Status::napi_ok {
            return Self::UNDEFINED;
        }
        match kind {
            sys::ValueType::napi_undefined => Self::UNDEFINED,
            sys::ValueType::napi_null => Self::NULL,
            sys::ValueType::napi_boolean => {
                let mut out = false;
                if sys::napi_get_value_bool(env, value, &mut out) == sys::Status::napi_ok {
                    Self::from_bool(out)
                } else {
                    Self::UNDEFINED
                }
            }
            sys::ValueType::napi_number => {
                let mut out = 0.0;
                if sys::napi_get_value_double(env, value, &mut out) == sys::Status::napi_ok {
                    Self::from_f64(out)
                } else {
                    Self::UNDEFINED
                }
            }
            sys::ValueType::napi_string => read_string(env, value)
                .map(Self::from_string)
                .unwrap_or(Self::UNDEFINED),
            // Objects, functions, symbols and externals have identity, so they stay
            // in JavaScript behind a reference.
            _ => Self::reference(env, value).unwrap_or(Self::UNDEFINED),
        }
    }

    /// Takes a strong reference to a JS value so it outlives the current scope.
    ///
    /// # Safety
    ///
    /// `value` must be valid in `env`'s current handle scope.
    unsafe fn reference(env: sys::napi_env, value: sys::napi_value) -> Option<Self> {
        let mut reference = ptr::null_mut();
        if sys::napi_create_reference(env, value, 1, &mut reference) != sys::Status::napi_ok {
            return None;
        }
        Some(Self(Repr::Handle(Rc::new(Handle { env, reference }))))
    }

    /// Materialises this value in `env`'s current handle scope.
    ///
    /// # Safety
    ///
    /// The returned `napi_value` must not escape the current handle scope.
    pub(crate) unsafe fn to_napi(&self, env: sys::napi_env) -> Result<sys::napi_value, JsValue> {
        let mut out = ptr::null_mut();
        match &self.0 {
            Repr::Undefined => {
                env::check(sys::napi_get_undefined(env, &mut out), "napi_get_undefined")?;
            }
            Repr::Null => {
                env::check(sys::napi_get_null(env, &mut out), "napi_get_null")?;
            }
            Repr::Bool(value) => {
                env::check(
                    sys::napi_get_boolean(env, *value, &mut out),
                    "napi_get_boolean",
                )?;
            }
            Repr::Number(value) => {
                env::check(
                    sys::napi_create_double(env, *value, &mut out),
                    "napi_create_double",
                )?;
            }
            Repr::Str(value) => {
                env::check(
                    sys::napi_create_string_utf8(
                        env,
                        value.as_ptr().cast(),
                        value.len() as isize,
                        &mut out,
                    ),
                    "napi_create_string_utf8",
                )?;
            }
            Repr::Handle(handle) => {
                if handle.env != env {
                    return Err(JsValue::from_str(
                        "napi-rs-webgpu: a JavaScript value from another Node-API \
                         environment (thread) cannot be used here",
                    ));
                }
                out = handle.value()?;
            }
        }
        Ok(out)
    }
}

/// Reads a JS string into Rust.
///
/// # Safety
///
/// `value` must be a string valid in `env`'s current handle scope.
unsafe fn read_string(env: sys::napi_env, value: sys::napi_value) -> Option<String> {
    let mut len = 0;
    if sys::napi_get_value_string_utf8(env, value, ptr::null_mut(), 0, &mut len)
        != sys::Status::napi_ok
    {
        return None;
    }
    // `len` excludes the terminator that Node-API always writes.
    let mut bytes = alloc::vec![0u8; len + 1];
    let mut written = 0;
    if sys::napi_get_value_string_utf8(
        env,
        value,
        bytes.as_mut_ptr().cast(),
        bytes.len(),
        &mut written,
    ) != sys::Status::napi_ok
    {
        return None;
    }
    bytes.truncate(written);
    String::from_utf8(bytes).ok()
}

impl Default for JsValue {
    fn default() -> Self {
        Self::UNDEFINED
    }
}

impl fmt::Debug for JsValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Repr::Undefined => f.write_str("JsValue(undefined)"),
            Repr::Null => f.write_str("JsValue(null)"),
            Repr::Bool(value) => write!(f, "JsValue({value})"),
            Repr::Number(value) => write!(f, "JsValue({value})"),
            Repr::Str(value) => write!(f, "JsValue({value:?})"),
            Repr::Handle(_) => match self.coerce_to_string() {
                Some(text) => write!(f, "JsValue({text})"),
                None => f.write_str("JsValue(object)"),
            },
        }
    }
}

impl fmt::Display for JsValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Repr::Undefined => f.write_str("undefined"),
            Repr::Null => f.write_str("null"),
            Repr::Bool(value) => write!(f, "{value}"),
            Repr::Number(value) => write!(f, "{value}"),
            Repr::Str(value) => f.write_str(value),
            Repr::Handle(_) => match self.coerce_to_string() {
                Some(text) => f.write_str(&text),
                None => f.write_str("[object]"),
            },
        }
    }
}

impl JsValue {
    /// `String(this)`, for diagnostics. `None` if there is no environment, or if
    /// coercion itself throws (a `Symbol`, or a `toString` that fails).
    fn coerce_to_string(&self) -> Option<String> {
        let handle = match &self.0 {
            Repr::Handle(handle) => handle,
            _ => return None,
        };
        env::scope(|env| {
            // SAFETY: inside a handle scope on the value's own environment.
            unsafe {
                let value = handle.value()?;
                let mut out = ptr::null_mut();
                env::check(
                    sys::napi_coerce_to_string(env, value, &mut out),
                    "napi_coerce_to_string",
                )?;
                read_string(env, out).ok_or_else(|| {
                    JsValue::from_str("napi-rs-webgpu: could not read coerced string")
                })
            }
        })
        .ok()
        .or_else(|| {
            // A throwing coercion leaves an exception pending, which would poison
            // every later call on this environment.
            env::take_exception();
            None
        })
    }
}

impl PartialEq for JsValue {
    /// JavaScript `===` semantics, decided in Rust for primitives and by
    /// `napi_strict_equals` for anything with identity.
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Repr::Undefined, Repr::Undefined) | (Repr::Null, Repr::Null) => true,
            (Repr::Bool(a), Repr::Bool(b)) => a == b,
            (Repr::Number(a), Repr::Number(b)) => a == b,
            (Repr::Str(a), Repr::Str(b)) => a == b,
            (Repr::Handle(a), Repr::Handle(b)) => {
                if Rc::ptr_eq(a, b) {
                    return true;
                }
                if a.env != b.env {
                    return false;
                }
                env::scope(|env| {
                    // SAFETY: inside a handle scope on the shared environment.
                    unsafe {
                        let (left, right) = (a.value()?, b.value()?);
                        let mut equal = false;
                        env::check(
                            sys::napi_strict_equals(env, left, right, &mut equal),
                            "napi_strict_equals",
                        )?;
                        Ok(equal)
                    }
                })
                .unwrap_or(false)
            }
            _ => false,
        }
    }
}

impl Eq for JsValue {}

impl JsValue {
    /// A total order over values, so that generated types can derive `Ord` the way
    /// wasm-bindgen's do. Objects order by reference identity, which is stable
    /// within a run but carries no JavaScript meaning.
    fn ordering_key(&self) -> (u8, f64, Option<&str>, usize) {
        match &self.0 {
            Repr::Undefined => (0, 0.0, None, 0),
            Repr::Null => (1, 0.0, None, 0),
            Repr::Bool(value) => (2, f64::from(u8::from(*value)), None, 0),
            Repr::Number(value) => (3, *value, None, 0),
            Repr::Str(value) => (4, 0.0, Some(value), 0),
            Repr::Handle(handle) => (5, 0.0, None, handle.reference as usize),
        }
    }
}

impl PartialOrd for JsValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for JsValue {
    fn cmp(&self, other: &Self) -> Ordering {
        let (kind, number, text, address) = self.ordering_key();
        let (other_kind, other_number, other_text, other_address) = other.ordering_key();
        kind.cmp(&other_kind)
            .then_with(|| number.total_cmp(&other_number))
            .then_with(|| text.cmp(&other_text))
            .then_with(|| address.cmp(&other_address))
    }
}

impl Hash for JsValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let (kind, number, text, address) = self.ordering_key();
        kind.hash(state);
        number.to_bits().hash(state);
        text.hash(state);
        address.hash(state);
    }
}

impl AsRef<JsValue> for JsValue {
    #[inline]
    fn as_ref(&self) -> &JsValue {
        self
    }
}

macro_rules! from_number {
    ($($ty:ty),* $(,)?) => {
        $(
            impl From<$ty> for JsValue {
                #[inline]
                fn from(value: $ty) -> Self {
                    Self::from_f64(f64::from(value))
                }
            }
        )*
    };
}

from_number!(i8, i16, i32, u8, u16, u32, f32);

impl From<f64> for JsValue {
    #[inline]
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl From<bool> for JsValue {
    #[inline]
    fn from(value: bool) -> Self {
        Self::from_bool(value)
    }
}

impl From<&str> for JsValue {
    #[inline]
    fn from(value: &str) -> Self {
        Self::from_str(value)
    }
}

impl From<String> for JsValue {
    #[inline]
    fn from(value: String) -> Self {
        Self::from_string(value)
    }
}

impl From<&String> for JsValue {
    #[inline]
    fn from(value: &String) -> Self {
        Self::from_str(value)
    }
}

impl From<()> for JsValue {
    #[inline]
    fn from((): ()) -> Self {
        Self::UNDEFINED
    }
}

impl<T: Into<JsValue>> From<Option<T>> for JsValue {
    #[inline]
    fn from(value: Option<T>) -> Self {
        match value {
            Some(value) => value.into(),
            None => Self::UNDEFINED,
        }
    }
}

/// Casting between JavaScript types, mirroring `wasm_bindgen::JsCast`.
///
/// Every type generated by `#[wasm_bindgen]` here is `#[repr(transparent)]` over
/// [`JsValue`], which is what makes the reference casts sound.
pub trait JsCast: AsRef<JsValue> + Into<JsValue>
where
    Self: Sized,
{
    /// Whether `value` is an instance of this type, by `instanceof` or by
    /// `typeof` for the primitive wrappers.
    fn instanceof(value: &JsValue) -> bool;

    /// Whether `value` has this type. Defaults to [`JsCast::instanceof`], and is
    /// overridden for types that are recognised structurally.
    fn is_type_of(value: &JsValue) -> bool {
        Self::instanceof(value)
    }

    /// Reinterprets `value` as this type without checking.
    fn unchecked_from_js(value: JsValue) -> Self;

    /// Reinterprets `value` as this type without checking.
    fn unchecked_from_js_ref(value: &JsValue) -> &Self;

    /// Whether this value is an instance of `T`.
    fn has_type<T: JsCast>(&self) -> bool {
        T::is_type_of(self.as_ref())
    }

    /// Casts to `T`, returning `self` unchanged when the value is not a `T`.
    fn dyn_into<T: JsCast>(self) -> Result<T, Self> {
        if self.has_type::<T>() {
            Ok(self.unchecked_into())
        } else {
            Err(self)
        }
    }

    /// Casts to `&T`, or `None` when the value is not a `T`.
    fn dyn_ref<T: JsCast>(&self) -> Option<&T> {
        if self.has_type::<T>() {
            Some(self.unchecked_ref())
        } else {
            None
        }
    }

    /// Casts to `T` without checking.
    fn unchecked_into<T: JsCast>(self) -> T {
        T::unchecked_from_js(self.into())
    }

    /// Casts to `&T` without checking.
    fn unchecked_ref<T: JsCast>(&self) -> &T {
        T::unchecked_from_js_ref(self.as_ref())
    }

    /// Whether this value is an instance of `T`, by `instanceof`.
    fn is_instance_of<T: JsCast>(&self) -> bool {
        T::instanceof(self.as_ref())
    }
}

impl JsCast for JsValue {
    fn instanceof(_value: &JsValue) -> bool {
        true
    }

    fn unchecked_from_js(value: JsValue) -> Self {
        value
    }

    fn unchecked_from_js_ref(value: &JsValue) -> &Self {
        value
    }
}

/// A JavaScript `Error`, as produced by fallible Rust closures handed to JS.
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct JsError(JsValue);

impl JsError {
    /// A new `Error` with the given message.
    pub fn new(message: &str) -> Self {
        Self(crate::napi::rt::error(message))
    }

    /// The underlying value.
    pub fn into_value(self) -> JsValue {
        self.0
    }
}

impl From<JsError> for JsValue {
    fn from(error: JsError) -> Self {
        error.0
    }
}
