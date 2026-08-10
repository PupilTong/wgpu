//! `Promise`, and the `then` overloads that drive `JsFuture`.

use crate::closure::ScopedClosure;
use crate::rt;
use crate::value::{JsError, JsValue};

use super::Object;

js_type! {
    /// A JavaScript `Promise` resolving to `T`.
    ///
    /// `T` is what awaiting it produces, which is why [`crate::value::Promising`]
    /// maps this type to `T` and every other type to itself: a
    /// `Promise<JsOption<GpuAdapter>>` awaits to a `JsOption<GpuAdapter>`, and
    /// `JsFuture` reads the resolution type from there.
    Promise[T = JsValue]: [Object, JsValue],
    instanceof(value) { rt::instance_of(value, c"Promise") },
    resolves_to T,
}

impl<T: 'static> Promise<T> {
    /// `this.then(resolve)`.
    ///
    /// The returned promise resolves to whatever `resolve` returns, which this
    /// signature does not track — `JsValue` — because nothing in `wgpu` uses it.
    pub fn then(&self, resolve: &ScopedClosure<'_, dyn FnMut(T)>) -> Promise<JsValue> {
        self.call_then(&[callback(resolve)])
    }

    /// `this.then(resolve, reject)`.
    ///
    /// The pair that turns a promise into a Rust future: exactly one of the two
    /// callbacks runs, and each reports back through the state they share.
    /// `R` is the callback's own return value, discarded here for the same reason
    /// [`Promise::then`]'s is.
    pub fn then_with_reject<R: 'static>(
        &self,
        resolve: &ScopedClosure<'_, dyn FnMut(T) -> Result<R, JsError>>,
        reject: &ScopedClosure<'_, dyn FnMut(JsValue) -> Result<R, JsError>>,
    ) -> Promise<JsValue> {
        self.call_then(&[callback(resolve), callback(reject)])
    }

    /// `this.then(resolve, reject)` with both callbacks taking the untyped value.
    ///
    /// `js-sys` declares this one only on `Promise<JsValue>`; it is available for
    /// every `T` here because the callbacks ignore the resolution type anyway.
    pub fn then2(
        &self,
        resolve: &ScopedClosure<'_, dyn FnMut(JsValue)>,
        reject: &ScopedClosure<'_, dyn FnMut(JsValue)>,
    ) -> Promise<JsValue> {
        self.call_then(&[callback(resolve), callback(reject)])
    }

    /// `this.then(..arguments)`.
    ///
    /// Declared without `catch` in `js-sys`, so a throwing `then` — which means the
    /// value is not a thenable at all — ends the module rather than being reported.
    fn call_then(&self, arguments: &[JsValue]) -> Promise<JsValue> {
        rt::cast(rt::unwrap_js(
            rt::call_method(self.js(), c"then", arguments),
            "Promise.then",
        ))
    }
}

/// A closure's JavaScript function, ready to pass as an argument.
///
/// [`ScopedClosure`] owns the function; cloning the value only adds a reference, so
/// the closure keeps working for as long as JavaScript holds onto it.
fn callback<T: ?Sized>(closure: &ScopedClosure<'_, T>) -> JsValue {
    <ScopedClosure<'_, T> as AsRef<JsValue>>::as_ref(closure).clone()
}
