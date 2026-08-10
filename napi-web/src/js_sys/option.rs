//! `JsOption`, a value that JavaScript may have left `undefined`.

use crate::value::{JsCast, JsValue};

use super::Undefined;

js_type! {
    /// A JavaScript value that is either a `T` or `undefined`.
    ///
    /// This is not Rust's `Option`: the value stays in JavaScript, and only
    /// [`JsOption::is_empty`], [`JsOption::as_option`] or [`JsOption::into_option`]
    /// look at it. The WebGPU bindings use it wherever the IDL declares a nullable
    /// sequence member — `Array<JsOption<GpuColorTargetState>>` is a fragment
    /// state's `targets`, whose holes are `undefined` entries.
    ///
    /// **Only `undefined` is absent.** JavaScript `null` is a present value, which
    /// is what makes this different from [`Option`] round-tripping through
    /// [`crate::convert::FromJs`], where both become `None`.
    ///
    /// `js-sys` leaves the parameter unbounded; here it is [`JsCast`] so that
    /// [`JsCast::instanceof`] can ask `T` whether the present value is really a `T`.
    /// Every use in the WebGPU bindings satisfies that.
    JsOption[T: JsCast]: [JsValue],
    instanceof(value) { value.is_undefined() || <T as JsCast>::instanceof(value) },
    resolves_to Self,
}

impl<T: JsCast> JsOption<T> {
    /// An absent value, i.e. `undefined`.
    pub fn new() -> Self {
        Self::from_parent(JsValue::UNDEFINED)
    }

    /// A present value.
    pub fn wrap(value: T) -> Self {
        Self::from_parent(value.into())
    }

    /// A present value for `Some`, `undefined` for `None`.
    pub fn from_option(value: Option<T>) -> Self {
        match value {
            Some(value) => Self::wrap(value),
            None => Self::new(),
        }
    }

    /// Whether the value is absent, i.e. `undefined`. `null` is *not* absent.
    pub fn is_empty(&self) -> bool {
        self.js().is_undefined()
    }

    /// The value, cloned, or `None` if it is absent.
    pub fn as_option(&self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            Some(<T as JsCast>::unchecked_from_js(self.js().clone()))
        }
    }

    /// The value, or `None` if it is absent.
    pub fn into_option(self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            Some(<T as JsCast>::unchecked_from_js(self.0))
        }
    }

    /// The value.
    ///
    /// # Panics
    ///
    /// If the value is absent.
    pub fn unwrap(self) -> T {
        self.expect("called `JsOption::unwrap()` on an empty value")
    }

    /// The value, panicking with `message` if it is absent.
    ///
    /// # Panics
    ///
    /// If the value is absent.
    pub fn expect(self, message: &str) -> T {
        match self.into_option() {
            Some(value) => value,
            None => panic!("{message}"),
        }
    }

    /// The value, or `T`'s default if it is absent.
    pub fn unwrap_or_default(self) -> T
    where
        T: Default,
    {
        self.into_option().unwrap_or_default()
    }

    /// The value, or the result of `f` if it is absent.
    pub fn unwrap_or_else<F: FnOnce() -> T>(self, f: F) -> T {
        self.into_option().unwrap_or_else(f)
    }
}

impl<T: JsCast> Default for JsOption<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: JsCast> From<Undefined> for JsOption<T> {
    fn from(_undefined: Undefined) -> Self {
        Self::new()
    }
}

impl<T: JsCast> From<Option<T>> for JsOption<T> {
    fn from(value: Option<T>) -> Self {
        Self::from_option(value)
    }
}
