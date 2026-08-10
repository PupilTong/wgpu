//! The JavaScript built-ins `wgpu`'s WebGPU backend uses, standing in for `js-sys`.
//!
//! `js-sys` declares its types with `#[wasm_bindgen]`, so each of its methods is a
//! wasm import. The types here are the same types — same names, same generic
//! parameters, same method signatures — with each operation lowered onto the
//! Node-API calls in `crate::rt` instead: a property read, a property write, a
//! method call, a construction.
//!
//! # Scope
//!
//! Only what the vendored WebGPU bindings and `wgpu`'s backend reference. The
//! bindings were generated from a post-generics `web-sys`, so the containers are
//! typed — [`Array<T>`], [`Object<T>`], [`Promise<T>`], [`JsOption<T>`],
//! [`Iterator<T>`] — and those parameters have to be spelled the same way here or
//! the generated signatures do not parse. They are phantom: one JavaScript value is
//! one JavaScript value whatever its declared element type, so `T` only records
//! what the API promises the contents are.
//!
//! # How each type is built
//!
//! `js_type!` emits the whole boilerplate for one type: a `#[repr(transparent)]`
//! newtype over its JavaScript *parent* ([`Object`] for most, [`JsValue`] for the
//! ones that extend nothing), then [`Deref`](core::ops::Deref), [`AsRef`] and
//! [`From`] for every ancestor, [`JsCast`], [`AsJs`](crate::wasm_bindgen::AsJs),
//! [`FromJs`],
//! [`Promising`](crate::wasm_bindgen::Promising),
//! `Clone`, `Debug`, `PartialEq` and `Eq`. The transparent representation is what
//! makes [`JsCast::unchecked_from_js_ref`] — reinterpreting a `&JsValue` as
//! `&Self` — sound, and it is why every layer down to `JsValue` must stay
//! transparent.
//!
//! `PartialEq`/`Eq` are emitted for every type, including the ones `js-sys` leaves
//! without them: the generated WebGPU types all carry
//! `#[derive(Debug, Clone, PartialEq, Eq)]` and are newtypes over [`Object`], so
//! those derives only compile if `Object` has them. Equality is JavaScript `===`,
//! which for objects is reference identity.

use core::ffi::CStr;

use crate::convert::FromJs;
use crate::rt;
use crate::value::{JsCast, JsValue};

/// Emits the impls that let a type be used as one of its non-parent ancestors:
/// `&Uint8Array` where a `&JsValue` is wanted, and `Uint8Array` into `JsValue`.
///
/// Separate from [`js_type_core!`] because `macro_rules!` cannot expand the generic
/// parameter list inside a repetition over the ancestors; one invocation per
/// ancestor keeps each list at its own repetition depth.
macro_rules! js_upcast {
    (
        $name:ident $([$($generic:ident $(: $generic_bound:path)?),+])?,
        $ancestor:ty
    ) => {
        impl $(<$($generic $(: $generic_bound)?),+>)? AsRef<$ancestor>
            for $name $(<$($generic),+>)?
        {
            #[inline]
            fn as_ref(&self) -> &$ancestor {
                self.0.as_ref()
            }
        }

        impl $(<$($generic $(: $generic_bound)?),+>)? From<$name $(<$($generic),+>)?>
            for $ancestor
        {
            #[inline]
            fn from(value: $name $(<$($generic),+>)?) -> Self {
                value.0.into()
            }
        }
    };
}

/// Emits everything about a JavaScript type that only involves its parent.
///
/// Invoked through [`js_type!`], which adds the ancestors above the parent.
macro_rules! js_type_core {
    (
        $(#[$doc:meta])*
        $name:ident
            $([$($generic:ident $(: $generic_bound:path)? $(= $generic_default:ty)?),+])?
            : $parent:ty,
        instanceof($value:ident) $test:block,
        resolves_to $resolution:ty,
    ) => {
        $(#[$doc])*
        #[repr(transparent)]
        pub struct $name $(<$($generic $(: $generic_bound)? $(= $generic_default)?),+>)? (
            $parent,
            // The generic parameters name the JavaScript type of the contents. There
            // is one JavaScript value either way, so they hold no data.
            $($(core::marker::PhantomData<$generic>,)+)?
        );

        impl $(<$($generic $(: $generic_bound)?),+>)? $name $(<$($generic),+>)? {
            /// Wraps a parent value, which is the entire representation.
            #[inline]
            fn from_parent(parent: $parent) -> Self {
                Self(parent, $($(core::marker::PhantomData::<$generic>,)+)?)
            }

            /// This value as a plain [`JsValue`], which every [`crate::rt`]
            /// operation takes.
            #[inline]
            fn js(&self) -> &$crate::value::JsValue {
                <Self as AsRef<$crate::value::JsValue>>::as_ref(self)
            }
        }

        impl $(<$($generic $(: $generic_bound)?),+>)? core::ops::Deref
            for $name $(<$($generic),+>)?
        {
            type Target = $parent;

            #[inline]
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl $(<$($generic $(: $generic_bound)?),+>)? AsRef<$parent>
            for $name $(<$($generic),+>)?
        {
            #[inline]
            fn as_ref(&self) -> &$parent {
                &self.0
            }
        }

        impl $(<$($generic $(: $generic_bound)?),+>)? From<$name $(<$($generic),+>)?>
            for $parent
        {
            #[inline]
            fn from(value: $name $(<$($generic),+>)?) -> Self {
                value.0
            }
        }

        /// The unchecked conversion wasm-bindgen also generates for every imported
        /// type: whatever the value is, it is now spelled as this type. The chain of
        /// these down to [`JsValue`] is what lets a generated type's ancestor
        /// conversions be written as `value.into()`.
        impl $(<$($generic $(: $generic_bound)?),+>)? From<$crate::value::JsValue>
            for $name $(<$($generic),+>)?
        {
            #[inline]
            fn from(value: $crate::value::JsValue) -> Self {
                Self::from_parent(value.into())
            }
        }

        impl $(<$($generic $(: $generic_bound)?),+>)? $crate::value::JsCast for $name $(<$($generic),+>)? {
            fn instanceof($value: &$crate::value::JsValue) -> bool $test

            #[inline]
            fn unchecked_from_js(value: $crate::value::JsValue) -> Self {
                Self::from_parent(<$parent as $crate::value::JsCast>::unchecked_from_js(value))
            }

            #[inline]
            fn unchecked_from_js_ref(value: &$crate::value::JsValue) -> &Self {
                // SAFETY: `Self` is `#[repr(transparent)]` over its parent, which is
                // transparent over its own parent, down to `JsValue` — so `Self` and
                // `JsValue` have the same layout, and the phantom parameters add no
                // fields. This is the reinterpretation wasm-bindgen's generated
                // `unchecked_ref` performs.
                unsafe { &*core::ptr::from_ref(value).cast::<Self>() }
            }
        }

        impl $(<$($generic $(: $generic_bound)?),+>)? $crate::convert::AsJs for $name $(<$($generic),+>)? {
            #[inline]
            fn as_js(&self) -> $crate::value::JsValue {
                self.js().clone()
            }
        }

        impl $(<$($generic $(: $generic_bound)?),+>)? $crate::convert::FromJs for $name $(<$($generic),+>)? {
            #[inline]
            fn from_js(value: $crate::value::JsValue) -> Self {
                <Self as $crate::value::JsCast>::unchecked_from_js(value)
            }
        }

        impl $(<$($generic $(: $generic_bound)?),+>)? $crate::value::Promising for $name $(<$($generic),+>)? {
            type Resolution = $resolution;
        }

        impl $(<$($generic $(: $generic_bound)?),+>)? Clone for $name $(<$($generic),+>)? {
            #[inline]
            fn clone(&self) -> Self {
                Self::from_parent(self.0.clone())
            }
        }

        impl $(<$($generic $(: $generic_bound)?),+>)? core::fmt::Debug
            for $name $(<$($generic),+>)?
        {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_tuple(stringify!($name)).field(self.js()).finish()
            }
        }

        impl $(<$($generic $(: $generic_bound)?),+>)? PartialEq for $name $(<$($generic),+>)? {
            /// JavaScript `===`, which for objects is reference identity.
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.js() == other.js()
            }
        }

        impl $(<$($generic $(: $generic_bound)?),+>)? Eq for $name $(<$($generic),+>)? {}
    };
}

/// Emits one JavaScript type: the transparent newtype and every conversion the
/// generated bindings and `wgpu` expect on it.
///
/// The invocation reads as the type's JavaScript identity:
///
/// ```ignore
/// js_type! {
///     /// The JavaScript `Uint8Array`.
///     Uint8Array: [Object, JsValue],
///     instanceof(value) { is_typedarray(value, TypedarrayType::uint8_array) },
///     resolves_to Self,
/// }
/// ```
///
/// * the optional `[..]` after the name lists the generic parameters, each with an
///   optional bound and an optional default, exactly as `js-sys` spells them;
/// * the `[..]` after the colon is the ancestry, nearest first. The first entry is
///   the parent: it becomes the newtype's field and its `Deref` target. There is one
///   arm per depth, because that is what keeps the two lists from having to nest;
/// * `instanceof` is the body of [`JsCast::instanceof`] — the truthful runtime test
///   for this type, not a class-name lookup, because several of these built-ins are
///   recognised structurally (`typeof`, `Array.isArray`) rather than by class;
/// * `resolves_to` is [`Promising::Resolution`]: `Self` for everything except
///   [`Promise<T>`], which resolves to `T`.
macro_rules! js_type {
    // Extends nothing, so `JsValue` is the parent and there is nothing above it.
    (
        $(#[$doc:meta])*
        $name:ident
            $([$($generic:ident $(: $generic_bound:path)? $(= $generic_default:ty)?),+ $(,)?])?
            : [$parent:ty],
        instanceof($value:ident) $test:block,
        resolves_to $resolution:ty,
    ) => {
        js_type_core! {
            $(#[$doc])*
            $name $([$($generic $(: $generic_bound)? $(= $generic_default)?),+])? : $parent,
            instanceof($value) $test,
            resolves_to $resolution,
        }
    };

    // Extends one type, which extends `JsValue`.
    (
        $(#[$doc:meta])*
        $name:ident
            $([$($generic:ident $(: $generic_bound:path)? $(= $generic_default:ty)?),+ $(,)?])?
            : [$parent:ty, $ancestor:ty],
        instanceof($value:ident) $test:block,
        resolves_to $resolution:ty,
    ) => {
        js_type_core! {
            $(#[$doc])*
            $name $([$($generic $(: $generic_bound)? $(= $generic_default)?),+])? : $parent,
            instanceof($value) $test,
            resolves_to $resolution,
        }

        js_upcast!($name $([$($generic $(: $generic_bound)?),+])?, $ancestor);
    };
}

mod array;
mod iterator;
mod object;
mod option;
mod primitive;
mod promise;
mod typed_array;

pub use self::array::{Array, ArrayIntoIter, ArrayIter, ArrayTuple};
pub use self::iterator::{IntoIter, Iter, Iterator, IteratorNext};
pub use self::object::Object;
pub use self::option::JsOption;
pub use self::primitive::{Function, JsString, Null, Number, Undefined};
pub use self::promise::Promise;
pub use self::typed_array::{ArrayBuffer, Uint32Array, Uint8Array};

/// `globalThis`.
///
/// `js-sys` tries `self`, `window` and `global` in turn because its glue cannot ask
/// the engine directly. Node-API can: `napi_get_global` *is* the realm's global
/// object, so there is nothing to search.
pub fn global() -> Object {
    rt::cast(rt::unwrap_js(rt::global_this(), "reading globalThis"))
}

/// Stand-in for the `Reflect` namespace.
///
/// Only the two operations `wgpu` reaches for: setting a dictionary member that has
/// no generated setter (`constants`, `colorSpace`), and building the limits object
/// member by member.
#[allow(non_snake_case)]
pub mod Reflect {
    use super::{rt, JsValue};

    /// `Reflect.get(target, key)`.
    pub fn get(target: &JsValue, key: &JsValue) -> Result<JsValue, JsValue> {
        rt::get_dynamic(target, key)
    }

    /// `Reflect.set(target, property_key, value)`.
    ///
    /// The `bool` is `Reflect.set`'s own report of whether the write took effect.
    /// `napi_set_property` distinguishes only success from a thrown exception, so
    /// this is `Ok(true)` whenever nothing was thrown: the case `Reflect.set`
    /// reports as `false` — a write refused without an exception, as when a
    /// non-strict assignment hits a read-only property — is not observable through
    /// Node-API.
    pub fn set(target: &JsValue, property_key: &JsValue, value: &JsValue) -> Result<bool, JsValue> {
        rt::set_dynamic(target, property_key, value).map(|()| true)
    }
}

/// Stand-in for `js_sys::futures`, where `JsFuture` and `spawn_local` are defined
/// and from which `wasm_bindgen_futures` re-exports them.
pub mod futures {
    pub use crate::futures::{spawn_local, JsFuture};
}

/// Reads a property declared without `catch`, converting the result.
///
/// Every getter in this module is this call. `operation` names the property so a
/// throwing getter reports which one threw.
fn get_property<T: FromJs>(target: &JsValue, name: &CStr, operation: &str) -> T {
    T::from_js(rt::unwrap_js(rt::get(target, name), operation))
}

/// Calls a method declared without `catch`, converting the result.
fn call_method<T: FromJs>(
    target: &JsValue,
    name: &CStr,
    arguments: &[JsValue],
    operation: &str,
) -> T {
    T::from_js(rt::unwrap_js(
        rt::call_method(target, name, arguments),
        operation,
    ))
}

/// `new globalThis[class](..arguments)` for a constructor declared without `catch`.
fn construct<T: JsCast>(class: &CStr, arguments: &[JsValue], operation: &str) -> T {
    rt::cast(rt::unwrap_js(rt::construct(class, arguments), operation))
}
