//! The JavaScript language types the WebGPU bindings name.
//!
//! WebGPU's IDL is not written only in terms of WebGPU: a `sequence<DOMString>`
//! argument is an array of JavaScript strings, a `setlike` interface hands back a
//! JavaScript iterator, a nullable member is a slot that may hold nothing, and a
//! `record<DOMString, double>` is a plain object whose values are numbers. This
//! module declares those, so that a generated signature can say what the IDL says.
//!
//! [`Reflect`] is here for the other reason: two WebGPU dictionary members
//! (`GPUCanvasConfiguration.colorSpace`, and the `constants` of a programmable
//! stage) are written with a key computed at run time rather than a key the
//! declaration knows, and property access by value is a JavaScript operation with
//! no WebGPU name.

use core::marker::PhantomData;

use crate::napi::convert::FromJs;
use crate::napi::rt;
use crate::napi::value::{JsCast, JsValue};
use crate::support::{Error, Object};

js_type! {
    /// A JavaScript string.
    ///
    /// The bindings name it where WebGPU's IDL says `DOMString` inside a container
    /// — `sequence<GPUTextureFormat>` reaches Rust as `&[JsString]`, and
    /// `WGSLLanguageFeatures.keys()` as an iterator of them. A `DOMString` on its
    /// own is a Rust `String` in argument and return position, because there is
    /// nothing to be gained from keeping it in JavaScript.
    JsString: [Object, JsValue],
    instanceof(value) { value.is_string() },
}

js_type! {
    /// A JavaScript number.
    ///
    /// Present as the value type of [`Object`]'s `record` form: the device
    /// descriptor's `requiredLimits` is a `record<DOMString, GPUSize64>`, which the
    /// bindings spell `&Object<Number>`.
    Number: [Object, JsValue],
    instanceof(value) { value.as_f64().is_some() },
}

js_type! {
    /// JavaScript `undefined`, as the resolution of a promise that resolves to
    /// nothing.
    ///
    /// WebGPU has two: `GPUQueue.onSubmittedWorkDone()` and `GPUBuffer.mapAsync()`,
    /// both `Promise<undefined>`. It is a type rather than `()` so that the
    /// callback a `then` takes has an argument to name.
    Undefined: [JsValue],
    instanceof(value) { value.is_undefined() },
}

js_type! {
    /// A slot that may hold a `T`, still in JavaScript.
    ///
    /// WebGPU's IDL has nullable types in three places this crate reaches:
    /// `Promise<GPUAdapter?>`, `Promise<GPUError?>`, and the four sequences whose
    /// holes are meaningful (`GPUFragmentState.targets`,
    /// `GPURenderPassDescriptor.colorAttachments`,
    /// `GPUVertexState.buffers`, `GPUPipelineLayoutDescriptor.bindGroupLayouts`).
    ///
    /// It is not a Rust `Option<T>`, because the value has not been inspected yet:
    /// a `Promise<GPUAdapter?>` resolves to *something*, and only
    /// [`into_option`](Self::into_option) asks whether that something is a
    /// `GPUAdapter`. Absent is `undefined`; WebIDL converts that to `null` at every
    /// nullable parameter, so a hole written this way arrives as the `null` the
    /// specification calls for.
    JsOption[T]: [JsValue],
    instanceof(value) { let _ = value; true },
}

js_type! {
    /// A JavaScript iterator whose values are `T`.
    ///
    /// The one `setlike` interface in this surface is `WGSLLanguageFeatures`, whose
    /// `keys()`, `values()` and `entries()` each return one of these.
    JsIterator[T]: [Object, JsValue],
    instanceof(value) {
        // The iterator protocol is a callable `next`, which is also all that
        // `for..of` requires of an iterator object. `instanceof` has nothing to
        // test against: the iterator prototypes are not on the global.
        rt::get(value, c"next")
            .map(|next| next.is_function())
            .unwrap_or(false)
    },
}

impl<T> JsOption<T>
where
    T: JsCast,
{
    /// An empty slot.
    #[must_use]
    pub fn new() -> Self {
        Self::from(JsValue::UNDEFINED)
    }

    /// A slot holding `value`.
    #[must_use]
    pub fn wrap(value: T) -> Self {
        Self::from(value.into())
    }

    /// A slot holding `option`'s value, if it has one.
    #[must_use]
    pub fn from_option(option: Option<T>) -> Self {
        match option {
            Some(value) => Self::wrap(value),
            None => Self::new(),
        }
    }

    /// Whether the slot is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.js().is_undefined()
    }

    /// The value, cloned out of the slot.
    #[must_use]
    pub fn as_option(&self) -> Option<T> {
        (!self.is_empty()).then(|| T::unchecked_from_js(self.js().clone()))
    }

    /// The value, taken out of the slot.
    #[must_use]
    pub fn into_option(self) -> Option<T> {
        (!self.is_empty()).then(|| T::unchecked_from_js(JsValue::from(self)))
    }

    /// The value.
    ///
    /// # Panics
    ///
    /// If the slot is empty.
    #[must_use]
    pub fn unwrap(self) -> T {
        self.expect("called `JsOption::unwrap()` on an empty value")
    }

    /// The value, panicking with `message` if the slot is empty.
    ///
    /// # Panics
    ///
    /// If the slot is empty.
    #[must_use]
    #[track_caller]
    pub fn expect(self, message: &str) -> T {
        match self.into_option() {
            Some(value) => value,
            None => panic!("{message}"),
        }
    }
}

impl<T: JsCast> Default for JsOption<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Walks the iterator, one `next()` call per step.
impl<T: FromJs> IntoIterator for JsIterator<T> {
    type Item = Result<T, Error>;
    type IntoIter = JsIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        JsIter {
            iterator: self.into(),
            exhausted: false,
            values: PhantomData,
        }
    }
}

/// The Rust iterator [`JsIterator`] turns into.
///
/// A step is `next()`, then `done` and `value` off the result object, which is the
/// iterator protocol written out. Once a step throws, or reports `done`, no further
/// call is made: an iterator that has finished or failed is not asked again.
#[derive(Debug)]
pub struct JsIter<T> {
    iterator: JsValue,
    exhausted: bool,
    values: PhantomData<T>,
}

impl<T: FromJs> Iterator for JsIter<T> {
    type Item = Result<T, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        let step = || -> Result<Option<T>, JsValue> {
            let step = rt::call_method(&self.iterator, c"next", &[])?;
            if rt::get(&step, c"done")?.is_truthy() {
                return Ok(None);
            }
            Ok(Some(T::from_js(rt::get(&step, c"value")?)))
        };
        match step() {
            Ok(Some(value)) => Some(Ok(value)),
            Ok(None) => {
                self.exhausted = true;
                None
            }
            Err(error) => {
                self.exhausted = true;
                Some(Err(Error::from(error)))
            }
        }
    }
}

/// Property access with a key that is a value.
///
/// The generated bindings never need this — every member they declare has a name
/// known at compile time, which is why the runtime's own `get` and `set` take a
/// `&CStr`. `wgpu` reaches for it twice: to set `GPUCanvasConfiguration.colorSpace`,
/// which the vendored WebGPU bindings do not declare, and to build the `constants`
/// record of a programmable stage, whose keys are the shader's override names.
///
/// It is a type with associated functions rather than free functions so that the
/// call reads as it does against `js-sys`, whose `Reflect` mirrors JavaScript's own
/// `Reflect` namespace.
#[derive(Debug)]
#[non_exhaustive]
pub struct Reflect;

impl Reflect {
    /// `target[key]`.
    ///
    /// # Errors
    ///
    /// If the read throws — a proxy trap, or a getter that raises.
    pub fn get(target: &JsValue, key: &JsValue) -> Result<JsValue, JsValue> {
        rt::get_value(target, key)
    }

    /// `target[key] = value`.
    ///
    /// Answers `true` when the write took effect, matching JavaScript's
    /// `Reflect.set`; a Node-API write that is refused rather than thrown reports
    /// success, so in practice this is `true` whenever it is `Ok`.
    ///
    /// # Errors
    ///
    /// If the write throws — a frozen object in strict mode, a proxy trap, or a
    /// setter that raises.
    pub fn set(target: &JsValue, key: &JsValue, value: &JsValue) -> Result<bool, JsValue> {
        rt::set_value(target, key, value).map(|()| true)
    }
}

/// `globalThis`.
///
/// `js_sys::global()` is how `wgpu` decides whether it is running on a browser's
/// main thread or in a dedicated worker, by looking for the `Window` and
/// `WorkerGlobalScope` constructors on it.
#[must_use]
pub fn global() -> Object {
    Object::from(rt::unwrap_js(rt::global_this(), "globalThis"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_slot_is_undefined() {
        let empty = JsOption::<JsString>::new();
        assert!(empty.is_empty());
        assert!(empty.as_option().is_none());
        assert!(empty.into_option().is_none());
    }

    #[test]
    fn a_filled_slot_keeps_its_value() {
        let slot = JsOption::wrap(JsString::from(JsValue::from_str("wgsl")));
        assert!(!slot.is_empty());
        assert_eq!(
            slot.into_option().and_then(|value| value.as_string()),
            Some(alloc::string::String::from("wgsl"))
        );
    }

    #[test]
    fn from_option_maps_both_cases() {
        assert!(JsOption::<Number>::from_option(None).is_empty());
        let filled = JsOption::from_option(Some(Number::from(JsValue::from_f64(4.0))));
        assert!(!filled.is_empty());
    }
}
