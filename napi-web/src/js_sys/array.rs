//! `Array`, `ArrayTuple` and their Rust iterators.

use alloc::vec::Vec;
use core::ops::Range;

use napi_sys as sys;

use crate::env;
use crate::rt;
use crate::value::{JsCast, JsValue};

use super::Object;

js_type! {
    /// The JavaScript `Array`, whose elements are declared to be `T`.
    Array[T = JsValue]: [Object, JsValue],
    instanceof(value) { is_array(value) },
    resolves_to Self,
}

js_type! {
    /// A JavaScript `Array` used as a fixed-arity tuple, whose Rust type parameter
    /// is the tuple of its element types.
    ///
    /// `GPUSupportedFeatures.entries()` yields these as `ArrayTuple<(JsString,
    /// JsString)>`. `js-sys` bounds the parameter with its `JsTuple` trait and
    /// provides accessors for every arity up to eight; the accessors here cover the
    /// pair, which is the only arity the WebGPU bindings produce.
    ArrayTuple[T = (JsValue,)]: [Object, JsValue],
    instanceof(value) { is_array(value) },
    resolves_to Self,
}

/// `Array.isArray(value)`.
///
/// [`crate::rt`] performs this test inside `array_items`/`array_length` but does
/// not expose it, and both array types need it for [`JsCast::instanceof`].
fn is_array(value: &JsValue) -> bool {
    // Only a handle to a JavaScript object can be an array, and deciding that needs
    // no environment, so primitives never open a handle scope.
    if !value.is_object() {
        return false;
    }
    env::scope(|env| {
        // SAFETY: inside a handle scope on `env`.
        unsafe {
            let value = value.to_napi(env)?;
            let mut result = false;
            env::check(sys::napi_is_array(env, value, &mut result), "napi_is_array")?;
            Ok(result)
        }
    })
    .unwrap_or(false)
}

impl Array {
    /// `new Array()`, i.e. `[]`.
    ///
    /// Untyped, as in `js-sys`. Use [`Array::new_typed`] when the element type is
    /// known.
    pub fn new() -> Self {
        rt::cast(rt::array_from(&[]))
    }
}

impl Default for Array {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: JsCast> Array<T> {
    /// `new Array()` with `T` as the element type.
    pub fn new_typed() -> Self {
        <Self as JsCast>::unchecked_from_js(rt::array_from(&[]))
    }

    /// The array's `length`.
    pub fn length(&self) -> u32 {
        rt::array_length(self.js())
    }

    /// `this[index]`.
    ///
    /// Out of range yields `undefined` reinterpreted as `T`, which is what the
    /// wasm-bindgen binding does.
    pub fn get(&self, index: u32) -> T {
        rt::cast(rt::unwrap_js(
            rt::get_index(self.js(), index),
            "Array element read",
        ))
    }

    /// `this[index] = value`.
    pub fn set(&self, index: u32, value: T) {
        rt::unwrap_js(
            rt::set_index(self.js(), index, element(&value)),
            "Array element write",
        );
    }

    /// `this.push(value)`, returning the new length.
    pub fn push(&self, value: &T) -> u32 {
        super::call_method(self.js(), c"push", &[element(value).clone()], "Array.push")
    }

    /// The elements, copied into a Rust vector.
    pub fn to_vec(&self) -> Vec<T> {
        rt::array_items(self.js())
            .into_iter()
            .map(<T as JsCast>::unchecked_from_js)
            .collect()
    }

    /// Iterates the elements, reading each one when it is reached.
    pub fn iter(&self) -> ArrayIter<'_, T> {
        ArrayIter {
            range: 0..self.length(),
            array: self,
        }
    }
}

impl<T: JsCast> From<&[T]> for Array<T> {
    fn from(items: &[T]) -> Self {
        let values: Vec<JsValue> = items.iter().map(|item| element(item).clone()).collect();
        <Self as JsCast>::unchecked_from_js(rt::array_from(&values))
    }
}

/// An element as the plain [`JsValue`] the [`crate::rt`] operations take.
///
/// `JsCast` guarantees `AsRef<JsValue>` and nothing more, so this names that impl
/// rather than relying on method resolution against whatever else `T` implements.
#[inline]
fn element<T: JsCast>(value: &T) -> &JsValue {
    <T as AsRef<JsValue>>::as_ref(value)
}

/// Iterates a borrowed [`Array`]. Created by [`Array::iter`].
pub struct ArrayIter<'a, T = JsValue> {
    range: Range<u32>,
    array: &'a Array<T>,
}

impl<T> core::fmt::Debug for ArrayIter<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ArrayIter")
            .field("range", &self.range)
            .finish_non_exhaustive()
    }
}

impl<T: JsCast> core::iter::Iterator for ArrayIter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.range.next()?;
        Some(self.array.get(index))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl<T: JsCast> core::iter::ExactSizeIterator for ArrayIter<'_, T> {}

/// Iterates an owned [`Array`]. Created by `IntoIterator`.
pub struct ArrayIntoIter<T = JsValue> {
    range: Range<u32>,
    array: Array<T>,
}

impl<T> core::fmt::Debug for ArrayIntoIter<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ArrayIntoIter")
            .field("range", &self.range)
            .finish_non_exhaustive()
    }
}

impl<T: JsCast> core::iter::Iterator for ArrayIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.range.next()?;
        Some(self.array.get(index))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl<T: JsCast> core::iter::ExactSizeIterator for ArrayIntoIter<T> {}

impl<T: JsCast> core::iter::IntoIterator for Array<T> {
    type Item = T;
    type IntoIter = ArrayIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        ArrayIntoIter {
            range: 0..self.length(),
            array: self,
        }
    }
}

impl<'a, T: JsCast> core::iter::IntoIterator for &'a Array<T> {
    type Item = T;
    type IntoIter = ArrayIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T1: JsCast, T2: JsCast> ArrayTuple<(T1, T2)> {
    /// `this[0]`.
    pub fn get0(&self) -> T1 {
        rt::cast(rt::unwrap_js(
            rt::get_index(self.js(), 0),
            "ArrayTuple element read",
        ))
    }

    /// `this[1]`.
    pub fn get1(&self) -> T2 {
        rt::cast(rt::unwrap_js(
            rt::get_index(self.js(), 1),
            "ArrayTuple element read",
        ))
    }

    /// Both elements, as a Rust tuple.
    pub fn into_tuple(self) -> (T1, T2) {
        (self.get0(), self.get1())
    }
}
