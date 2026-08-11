//! The three types every WebGPU binding needs that are not themselves WebGPU.
//!
//! WebGPU's IDL roots every interface and every dictionary at `object`, hands back
//! promises from four of its methods, and lets most of them throw. This module
//! supplies exactly those: [`Object`], [`Promise`] and [`Error`]. They are this
//! crate's own types — the point of the crate is that nothing here is a stand-in
//! for `js-sys`.

use alloc::string::String;
use core::fmt;
use core::marker::PhantomData;

use crate::napi::closure::Closure;
use crate::napi::convert::{AsJs, FromJs};
use crate::napi::value::{JsCast, JsError, JsValue};

/// A JavaScript object: what every WebGPU interface and dictionary extends.
///
/// Dictionaries are plain `{}` built by their constructor and filled in by their
/// setters; interfaces are host objects. Both are one JavaScript value, so both are
/// `#[repr(transparent)]` over [`JsValue`] — which is what makes the reference casts
/// in [`JsCast`] sound.
///
/// `T` names what the object's *values* are, for the one WebGPU member that is a
/// `record`: `GPUDeviceDescriptor.requiredLimits` is a
/// `record<DOMString, GPUSize64>`, so its setter takes an
/// `&Object<`[`Number`](crate::Number)`>`. It holds no data — a record and an
/// ordinary object are the same JavaScript value — and defaults to [`JsValue`],
/// which is every other use.
#[repr(transparent)]
pub struct Object<T = JsValue>(JsValue, PhantomData<T>);

impl Object {
    /// A new, empty object.
    #[must_use]
    pub fn new() -> Self {
        Self(crate::napi::rt::new_object(), PhantomData)
    }
}

impl<T> Object<T> {
    /// A new, empty object whose values are `T`.
    ///
    /// The same `{}` [`Object::new`] builds; the type parameter is what the caller
    /// intends to put in it.
    #[must_use]
    pub fn new_typed() -> Self {
        Self(crate::napi::rt::new_object(), PhantomData)
    }

    /// JavaScript's `Object.prototype.valueOf`, which for an ordinary object is the
    /// object itself.
    ///
    /// `wgpu` calls it to spell a `GPUError` as the plain object its error
    /// classifier takes.
    #[must_use]
    pub fn value_of(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl Default for Object {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for Object<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<T> PartialEq for Object<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Object<T> {}

impl<T> fmt::Debug for Object<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

/// The last link in every `Deref` chain in this crate.
///
/// Each generated type derefs to its parent and each parent to *its* parent, all
/// of which end at [`Object`]; this step is what carries that chain the rest of
/// the way to [`JsValue`], so a `&GpuBuffer`, a `&Uint8Array` or a
/// `&HtmlCanvasElement` coerces where a `&JsValue` is wanted. `js_sys::Object`
/// derefs to `wasm_bindgen::JsValue` for the same reason, and `wgpu` relies on it
/// in signatures it does not control — `Uint8Array::new_with_byte_offset_and_length`
/// and `Uint8Array::set` both take `&JsValue` and are handed an `&ArrayBuffer` and
/// an `&Uint8Array`.
impl<T> core::ops::Deref for Object<T> {
    type Target = JsValue;

    #[inline]
    fn deref(&self) -> &JsValue {
        &self.0
    }
}

impl<T> AsRef<JsValue> for Object<T> {
    #[inline]
    fn as_ref(&self) -> &JsValue {
        &self.0
    }
}

impl<T> From<Object<T>> for JsValue {
    #[inline]
    fn from(object: Object<T>) -> Self {
        object.0
    }
}

impl<T> From<JsValue> for Object<T> {
    #[inline]
    fn from(value: JsValue) -> Self {
        Self(value, PhantomData)
    }
}

impl<T> JsCast for Object<T> {
    fn instanceof(value: &JsValue) -> bool {
        // `typeof`, not `instanceof globalThis.Object`: an object with a null
        // prototype and an object from another realm are both objects and neither is
        // `instanceof Object`. Functions answer `typeof "function"` yet are objects.
        value.is_object() || value.is_function()
    }

    fn unchecked_from_js(value: JsValue) -> Self {
        Self(value, PhantomData)
    }

    fn unchecked_from_js_ref(value: &JsValue) -> &Self {
        // SAFETY: `Object<T>` is `#[repr(transparent)]` over `JsValue`; the phantom
        // parameter occupies no space.
        unsafe { &*core::ptr::from_ref(value).cast::<Self>() }
    }
}

impl<T> AsJs for Object<T> {
    #[inline]
    fn as_js(&self) -> JsValue {
        self.0.clone()
    }
}

impl<T> FromJs for Object<T> {
    #[inline]
    fn from_js(value: JsValue) -> Self {
        Self(value, PhantomData)
    }
}

/// A JavaScript promise resolving to `T`.
///
/// Awaiting one goes through [`crate::napi::futures::JsFuture`], which attaches
/// `then` callbacks; `T` records what the resolution is and carries no data.
#[repr(transparent)]
pub struct Promise<T = JsValue> {
    object: Object,
    resolution: PhantomData<T>,
}

impl<T> Promise<T> {
    /// `promise.then(onFulfilled)`.
    ///
    /// The alternative to awaiting: `wgpu` uses it where the continuation is a
    /// callback it was handed rather than the rest of an `async fn` — the
    /// device-lost handler, and the submitted-work-done notification.
    ///
    /// # Panics
    ///
    /// If `then` throws, which for a real promise it does not.
    pub fn then(&self, on_fulfilled: &Closure<dyn FnMut(T)>) -> Promise {
        self.attach(&[on_fulfilled.as_js_value().clone()])
    }

    /// `promise.then(onFulfilled, onRejected)`.
    ///
    /// Both callbacks return a `Result` because either may throw; `R` is what they
    /// resolve the next promise with, which no caller here reads.
    ///
    /// # Panics
    ///
    /// If `then` throws, which for a real promise it does not.
    pub fn then_with_reject<R>(
        &self,
        on_fulfilled: &Closure<dyn FnMut(T) -> Result<R, JsError>>,
        on_rejected: &Closure<dyn FnMut(JsValue) -> Result<R, JsError>>,
    ) -> Promise
    where
        R: AsJs,
    {
        self.attach(&[
            on_fulfilled.as_js_value().clone(),
            on_rejected.as_js_value().clone(),
        ])
    }

    /// The `then` call both forms make, differing only in how many callbacks they
    /// pass.
    fn attach(&self, callbacks: &[JsValue]) -> Promise {
        let attached = crate::napi::rt::call_method(self.object.as_ref(), c"then", callbacks);
        Promise::from(crate::napi::rt::unwrap_js(attached, "Promise.then"))
    }
}

impl<T> From<JsValue> for Promise<T> {
    #[inline]
    fn from(value: JsValue) -> Self {
        Self {
            object: Object::from(value),
            resolution: PhantomData,
        }
    }
}

impl<T> Clone for Promise<T> {
    fn clone(&self) -> Self {
        Self {
            object: self.object.clone(),
            resolution: PhantomData,
        }
    }
}

impl<T> fmt::Debug for Promise<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Promise").finish_non_exhaustive()
    }
}

impl<T> AsRef<JsValue> for Promise<T> {
    #[inline]
    fn as_ref(&self) -> &JsValue {
        self.object.as_ref()
    }
}

impl<T> From<Promise<T>> for JsValue {
    #[inline]
    fn from(promise: Promise<T>) -> Self {
        promise.object.into()
    }
}

impl<T> JsCast for Promise<T> {
    fn instanceof(value: &JsValue) -> bool {
        // A promise is whatever has a callable `then`, which is also what `await`
        // accepts; `instanceof Promise` would reject a thenable from another realm.
        crate::napi::rt::get(value, c"then")
            .map(|then| then.is_function())
            .unwrap_or(false)
    }

    fn unchecked_from_js(value: JsValue) -> Self {
        Self::from(value)
    }

    fn unchecked_from_js_ref(value: &JsValue) -> &Self {
        // SAFETY: `Promise<T>` is `#[repr(transparent)]` over `Object`, itself
        // transparent over `JsValue`; `PhantomData` occupies no space.
        unsafe { &*core::ptr::from_ref(value).cast::<Self>() }
    }
}

impl<T> AsJs for Promise<T> {
    #[inline]
    fn as_js(&self) -> JsValue {
        self.object.as_js()
    }
}

impl<T> FromJs for Promise<T> {
    #[inline]
    fn from_js(value: JsValue) -> Self {
        <Self as JsCast>::unchecked_from_js(value)
    }
}

/// What a WebGPU call that throws hands back.
///
/// WebGPU rejects and throws with `GPUError` subclasses, `DOMException`s and, for a
/// device loss, a `GPUDeviceLostInfo`. Any of them is a JavaScript value, so this
/// keeps the value rather than flattening it to a string, and reads a message out
/// only when someone asks.
#[derive(Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Error(JsValue);

impl Error {
    /// The value JavaScript threw or rejected with.
    #[must_use]
    pub fn value(&self) -> &JsValue {
        &self.0
    }

    /// Takes the thrown value.
    #[must_use]
    pub fn into_value(self) -> JsValue {
        self.0
    }

    /// The error's `message`, or its string form when it has none — a promise may
    /// reject with any value, including a primitive.
    #[must_use]
    pub fn message(&self) -> String {
        match crate::napi::rt::get(&self.0, c"message") {
            Ok(message) if message.is_string() => message.as_string().unwrap_or_default(),
            _ => alloc::format!("{}", self.0),
        }
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Error").field(&self.message()).finish()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message())
    }
}

impl From<JsValue> for Error {
    #[inline]
    fn from(value: JsValue) -> Self {
        Self(value)
    }
}

impl From<Error> for JsValue {
    #[inline]
    fn from(error: Error) -> Self {
        error.0
    }
}
