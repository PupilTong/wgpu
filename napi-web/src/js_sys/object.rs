//! `Object`, the parent of nearly every WebGPU type.

use crate::rt;
use crate::value::{JsCast, JsValue};

js_type! {
    /// The JavaScript `Object`.
    ///
    /// `T` names the type of the object's property *values*, which is how the
    /// generated bindings spell a JavaScript record: `Object<Number>` is the limits
    /// object `requestDevice` takes. The parameter carries no data — an object is
    /// one JavaScript value whatever its members are — and it does not affect the
    /// `Deref` target, which is [`JsValue`] because `Object` extends nothing.
    Object[T = JsValue]: [JsValue],
    instanceof(value) {
        // `typeof`, not `value instanceof globalThis.Object`: a `null`-prototype
        // object and an object from another realm are both objects, and neither is
        // an `instanceof Object`. Functions answer `typeof "function"` yet are
        // objects, so they count too.
        value.is_object() || value.is_function()
    },
    resolves_to Self,
}

impl Object {
    /// `new Object()`, i.e. `{}`.
    ///
    /// Untyped, as in `js-sys`: the generated dictionary constructors start from
    /// this and immediately reinterpret it as the dictionary type. Use
    /// [`Object::new_typed`] for a record whose value type is known.
    pub fn new() -> Self {
        rt::cast(rt::new_object())
    }
}

impl Default for Object {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Object<T> {
    /// `new Object()` as a record of `T`.
    pub fn new_typed() -> Self {
        <Self as JsCast>::unchecked_from_js(rt::new_object())
    }

    /// `this.valueOf()`.
    ///
    /// For a plain object this is the object itself; the WebGPU error types
    /// override it, which is why `wgpu` calls it on a `GPUError` before reading the
    /// message off the result.
    pub fn value_of(&self) -> Object {
        super::call_method(self.js(), c"valueOf", &[], "Object.valueOf")
    }
}
