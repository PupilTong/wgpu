//! The JavaScript primitive wrappers: `String`, `Number`, `Function`, and the
//! `undefined`/`null` marker types.

use alloc::string::String;

use crate::rt;
use crate::value::JsValue;

use super::Object;

js_type! {
    /// The JavaScript `String`.
    ///
    /// `wgpu` reaches this type by casting: every WebGPU enum value is a JavaScript
    /// string, and the typed dictionary setters take `&[JsString]`. Read the
    /// contents with `as_string`, inherited from [`JsValue`] through `Deref`.
    JsString: [Object, JsValue],
    instanceof(value) {
        // `typeof value === "string"`. A `new String(..)` wrapper object is not a
        // string by this test, which is what `js-sys` decided too.
        value.is_string()
    },
    resolves_to Self,
}

impl JsString {
    /// The string's `length` in UTF-16 code units.
    pub fn length(&self) -> u32 {
        super::get_property(self.js(), c"length", "String.length")
    }
}

impl From<&str> for JsString {
    fn from(value: &str) -> Self {
        Self::from_parent(rt::cast(JsValue::from_str(value)))
    }
}

impl From<String> for JsString {
    fn from(value: String) -> Self {
        Self::from_parent(rt::cast(JsValue::from_string(value)))
    }
}

impl core::fmt::Display for JsString {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self.js(), f)
    }
}

js_type! {
    /// The JavaScript `Number`.
    ///
    /// Used as a type parameter — `Object<Number>` is how the generated bindings
    /// spell the limits record — rather than as a value.
    Number: [Object, JsValue],
    instanceof(value) {
        // Every JavaScript number is an `f64`, so this is `typeof value === "number"`.
        value.as_f64().is_some()
    },
    resolves_to Self,
}

impl Number {
    /// The numeric value, i.e. `this.valueOf()`.
    ///
    /// Zero for a value that is not a number, matching the generated bindings'
    /// unchecked conversion.
    pub fn value_of(&self) -> f64 {
        self.js().as_f64().unwrap_or(0.0)
    }
}

impl From<f64> for Number {
    fn from(value: f64) -> Self {
        Self::from_parent(rt::cast(JsValue::from_f64(value)))
    }
}

js_type! {
    /// A JavaScript function.
    ///
    /// `T` is the JavaScript signature, written as a Rust function type: the
    /// vendored bindings declare `GPUSupportedFeatures.forEach`'s callback as
    /// `Function<fn(JsString) -> Undefined>`. It is phantom — the calls below are
    /// dynamic, as JavaScript's are — and `js-sys` additionally bounds it with a
    /// `JsFunction` trait that maps the signature onto typed `call`/`bind`
    /// arities. Nothing in `wgpu` uses those, so the bound is omitted and the calls
    /// below are the untyped ones.
    Function[T = fn() -> JsValue]: [Object, JsValue],
    instanceof(value) {
        // `typeof value === "function"`, which is true for every callable —
        // including classes and cross-realm functions, which `instanceof Function`
        // would miss.
        value.is_function()
    },
    resolves_to Self,
}

impl<T> Function<T> {
    /// `this.call(context)`.
    pub fn call0(&self, context: &JsValue) -> Result<JsValue, JsValue> {
        rt::call(self.js(), context, &[])
    }

    /// `this.call(context, argument1)`.
    pub fn call1(&self, context: &JsValue, argument1: &JsValue) -> Result<JsValue, JsValue> {
        rt::call(self.js(), context, core::slice::from_ref(argument1))
    }

    /// `this.call(context, argument1, argument2)`.
    pub fn call2(
        &self,
        context: &JsValue,
        argument1: &JsValue,
        argument2: &JsValue,
    ) -> Result<JsValue, JsValue> {
        rt::call(self.js(), context, &[argument1.clone(), argument2.clone()])
    }

    /// The declared parameter count, i.e. `this.length`.
    pub fn length(&self) -> u32 {
        super::get_property(self.js(), c"length", "Function.length")
    }
}

js_type! {
    /// The JavaScript `undefined` value as a type.
    ///
    /// Names "resolves to nothing" in the generated signatures:
    /// `Promise<Undefined>` is `GPUDevice.lost`'s sibling `onSubmittedWorkDone`.
    Undefined: [JsValue],
    instanceof(value) { value.is_undefined() },
    resolves_to Self,
}

impl Undefined {
    /// The `undefined` constant.
    pub const UNDEFINED: Self = Self(JsValue::UNDEFINED);
}

impl Default for Undefined {
    fn default() -> Self {
        Self::UNDEFINED
    }
}

impl core::fmt::Display for Undefined {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("undefined")
    }
}

js_type! {
    /// The JavaScript `null` value as a type.
    Null: [JsValue],
    instanceof(value) { value.is_null() },
    resolves_to Self,
}

impl Null {
    /// The `null` constant.
    pub const NULL: Self = Self(JsValue::NULL);
}

impl Default for Null {
    fn default() -> Self {
        Self::NULL
    }
}

impl core::fmt::Display for Null {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("null")
    }
}
