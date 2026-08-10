//! The JavaScript iterator protocol, and the Rust iterators that drive it.

use crate::rt;
use crate::value::{JsCast, JsValue};

use super::Object;

js_type! {
    /// Any object following the JavaScript iterator protocol — anything with a
    /// `next()` returning `{ value, done }`.
    ///
    /// `GPUSupportedFeatures.keys()` and `.values()` return these. Iterate one from
    /// Rust with `IntoIterator`, which yields `Result<T, JsValue>` because `next()`
    /// may throw.
    Iterator[T = JsValue]: [JsValue],
    instanceof(value) { looks_like_iterator(value) },
    resolves_to Self,
}

js_type! {
    /// One step of a JavaScript iterator: the `{ value, done }` object `next()`
    /// returns.
    IteratorNext[T = JsValue]: [Object, JsValue],
    instanceof(value) { value.is_object() },
    resolves_to Self,
}

/// Whether `value` behaves like an iterator.
///
/// The protocol is structural — an iterator is any object with a callable `next` —
/// so this is the same duck test `js-sys` applies rather than a class check.
fn looks_like_iterator(value: &JsValue) -> bool {
    value.is_object() && rt::get(value, c"next").is_ok_and(|next| next.is_function())
}

impl<T: JsCast> Iterator<T> {
    /// `this.next()`.
    ///
    /// Declared with `catch` in `js-sys`: a `next` that throws, or that returns a
    /// non-object, is a `TypeError` the caller sees rather than a panic.
    pub fn next(&self) -> Result<IteratorNext<T>, JsValue> {
        rt::call_method(self.js(), c"next", &[]).map(rt::cast)
    }
}

impl<T: JsCast> IteratorNext<T> {
    /// Whether the sequence is exhausted.
    pub fn done(&self) -> bool {
        super::get_property(self.js(), c"done", "IteratorNext.done")
    }

    /// The value for this step. Meaningless once [`IteratorNext::done`] is true.
    pub fn value(&self) -> T {
        rt::cast(rt::unwrap_js(
            rt::get(self.js(), c"value"),
            "IteratorNext.value",
        ))
    }
}

/// How far a Rust iterator has got through a JavaScript one.
///
/// A JavaScript iterator holds its own position, so the only Rust-side state is
/// whether the end (or an error) has already been seen — after which `next` must
/// not be called again.
struct IterState {
    done: bool,
}

impl IterState {
    fn new() -> Self {
        Self { done: false }
    }

    fn next<T: JsCast>(&mut self, js: &Iterator<T>) -> Option<Result<T, JsValue>> {
        if self.done {
            return None;
        }
        let step = match js.next() {
            Ok(step) => step,
            Err(error) => {
                self.done = true;
                return Some(Err(error));
            }
        };
        if step.done() {
            self.done = true;
            None
        } else {
            Some(Ok(step.value()))
        }
    }
}

/// Iterates a borrowed [`Iterator`]. Created by `IntoIterator`.
pub struct Iter<'a, T = JsValue> {
    js: &'a Iterator<T>,
    state: IterState,
}

impl<T> core::fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Iter")
            .field("done", &self.state.done)
            .finish_non_exhaustive()
    }
}

/// Iterates an owned [`Iterator`]. Created by `IntoIterator`.
pub struct IntoIter<T = JsValue> {
    js: Iterator<T>,
    state: IterState,
}

impl<T> core::fmt::Debug for IntoIter<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IntoIter")
            .field("done", &self.state.done)
            .finish_non_exhaustive()
    }
}

impl<T: JsCast> core::iter::Iterator for Iter<'_, T> {
    type Item = Result<T, JsValue>;

    fn next(&mut self) -> Option<Self::Item> {
        self.state.next(self.js)
    }
}

impl<T: JsCast> core::iter::Iterator for IntoIter<T> {
    type Item = Result<T, JsValue>;

    fn next(&mut self) -> Option<Self::Item> {
        self.state.next(&self.js)
    }
}

impl<'a, T: JsCast> core::iter::IntoIterator for &'a Iterator<T> {
    type Item = Result<T, JsValue>;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            js: self,
            state: IterState::new(),
        }
    }
}

impl<T: JsCast> core::iter::IntoIterator for Iterator<T> {
    type Item = Result<T, JsValue>;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            js: self,
            state: IterState::new(),
        }
    }
}
