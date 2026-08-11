//! The declaration machinery every binding in this crate is written with.
//!
//! WebGPU has 94 interfaces, 55 dictionaries and 32 string enums in the surface this
//! crate covers, and each one is the same shape: a `#[repr(transparent)]` handle
//! over its parent, then property reads, property writes, method calls and
//! constructions keyed by JavaScript name. Writing that out 437 times by hand would
//! be 437 chances to mistype a name, so it is declared once here and applied by
//! macro. `tools/extract_surface.py` derives the declarations themselves from
//! WebGPU's IDL, via the bindings web-sys generates from it.
//!
//! The `instanceof` body is always supplied by the caller rather than assumed:
//! several of these types are recognised structurally (`typeof`, a callable `then`)
//! and a class-name lookup would answer wrongly for an object from another realm.

use alloc::string::String;
use core::ffi::CStr;

use crate::napi::convert::{AsJs, FromJs};
use crate::napi::rt;
use crate::napi::value::{JsCast, JsValue};
use crate::support::{Error, Object};

#[macro_export]
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
/// Invoked through [`crate::js_type!`], which adds the ancestors above the parent.
#[macro_export]
macro_rules! js_type_core {
    (
        $(#[$doc:meta])*
        $name:ident
            $([$($generic:ident $(: $generic_bound:path)? $(= $generic_default:ty)?),+])?
            : $parent:ty,
        instanceof($value:ident) $test:block,
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

            /// This value as a plain [`crate::JsValue`], which every `napi::rt`
            /// operation takes.
            #[inline]
            fn js(&self) -> &$crate::napi::value::JsValue {
                <Self as AsRef<$crate::napi::value::JsValue>>::as_ref(self)
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
        /// these down to [`crate::JsValue`] is what lets a generated type's ancestor
        /// conversions be written as `value.into()`.
        impl $(<$($generic $(: $generic_bound)?),+>)? From<$crate::napi::value::JsValue>
            for $name $(<$($generic),+>)?
        {
            #[inline]
            fn from(value: $crate::napi::value::JsValue) -> Self {
                Self::from_parent(value.into())
            }
        }

        impl $(<$($generic $(: $generic_bound)?),+>)? $crate::napi::value::JsCast for $name $(<$($generic),+>)? {
            fn instanceof($value: &$crate::napi::value::JsValue) -> bool $test

            #[inline]
            fn unchecked_from_js(value: $crate::napi::value::JsValue) -> Self {
                Self::from_parent(<$parent as $crate::napi::value::JsCast>::unchecked_from_js(value))
            }

            #[inline]
            fn unchecked_from_js_ref(value: &$crate::napi::value::JsValue) -> &Self {
                // SAFETY: `Self` is `#[repr(transparent)]` over its parent, which is
                // transparent over its own parent, down to `JsValue` — so `Self` and
                // `JsValue` have the same layout, and the phantom parameters add no
                // fields. This is the reinterpretation wasm-bindgen's generated
                // `unchecked_ref` performs.
                unsafe { &*core::ptr::from_ref(value).cast::<Self>() }
            }
        }

        impl $(<$($generic $(: $generic_bound)?),+>)? $crate::napi::convert::AsJs for $name $(<$($generic),+>)? {
            #[inline]
            fn as_js(&self) -> $crate::napi::value::JsValue {
                self.js().clone()
            }
        }

        impl $(<$($generic $(: $generic_bound)?),+>)? $crate::napi::convert::FromJs for $name $(<$($generic),+>)? {
            #[inline]
            fn from_js(value: $crate::napi::value::JsValue) -> Self {
                <Self as $crate::napi::value::JsCast>::unchecked_from_js(value)
            }
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
/// }
/// ```
///
/// * the optional `[..]` after the name lists the generic parameters, each with an
///   optional bound and an optional default, exactly as `js-sys` spells them;
/// * the `[..]` after the colon is the ancestry, nearest first. The first entry is
///   the parent: it becomes the newtype's field and its `Deref` target, and every
///   entry after it gets the `AsRef`/`From` pair that `web-sys` provides — up the
///   whole chain, which for `GPUUncapturedErrorEvent` is `Event`, `Object`,
///   `JsValue`;
/// * `instanceof` is the body of [`JsCast::instanceof`] — the truthful runtime test
///   for this type, not a class-name lookup, because several of these built-ins are
///   recognised structurally (`typeof`, `Array.isArray`) rather than by class;
#[macro_export]
macro_rules! js_type {
    (
        $(#[$doc:meta])*
        $name:ident
            $([$($generic:ident $(: $generic_bound:path)? $(= $generic_default:ty)?),+ $(,)?])?
            : [$parent:ty $(, $ancestor:ty)* $(,)?],
        instanceof($value:ident) $test:block,
    ) => {
        js_type_core! {
            $(#[$doc])*
            $name $([$($generic $(: $generic_bound)? $(= $generic_default)?),+])? : $parent,
            instanceof($value) $test,
        }

        // The parent is already handled by `js_type_core!`; everything above it
        // needs the conversion pair.
        js_upcasts! {
            $name $([$($generic $(: $generic_bound)?),+])?; $($ancestor),*
        }
    };
}

/// Emits the conversions for each of a type's ancestors above its parent.
///
/// One ancestor per step rather than one repetition over all of them: the generic
/// parameters belong to the type and the ancestors do not, so a repetition holding
/// both would have to agree on a count they have no reason to share. Recursing
/// keeps each of the two lists in a group of its own.
#[macro_export]
macro_rules! js_upcasts {
    (
        $name:ident $([$($generic:ident $(: $generic_bound:path)?),+])?;
    ) => {};

    (
        $name:ident $([$($generic:ident $(: $generic_bound:path)?),+])?;
        $ancestor:ty $(, $rest:ty)*
    ) => {
        js_upcast!($name $([$($generic $(: $generic_bound)?),+])?, $ancestor);

        js_upcasts! {
            $name $([$($generic $(: $generic_bound)?),+])?; $($rest),*
        }
    };
}

/// Declares the members of one JavaScript interface.
///
/// The invocation is the type's Rust name, then one line per member, and it emits
/// a single `impl` block:
///
/// ```ignore
/// webgpu_members! {
///     GpuDevice;
///     /// The queue this device submits to.
///     getter queue() -> GpuQueue as "queue";
///     setter set_label(value: &str) as "label";
///     method destroy() as "destroy";
///     method create_buffer(descriptor: &GpuBufferDescriptor) -> GpuBuffer
///         as "createBuffer" catch;
/// }
/// ```
///
/// * `getter` reads the named property, `setter` writes it, `method` calls it;
/// * the string after `as` is the JavaScript name, which is what the call is keyed
///   by — the Rust name is free to differ, and does wherever `web-sys` disambiguates
///   an overload (`get_mapped_range_with_f64_and_f64`);
/// * a missing `-> T` means the member yields `()`;
/// * `catch` makes the return type `Result<T, Error>`. Without it a throw panics,
///   naming the operation the way WebGPU's IDL spells it: `GPUDevice.createBuffer`.
///   That is what wasm-bindgen does for a binding declared without `catch`, whose
///   exception escapes through the import boundary and ends the module;
/// * doc comments in front of a member are carried onto the generated function.
///
/// The panic name is built from the Rust type name, whose only difference from the
/// JavaScript class name is the capitalisation of the `Gpu` prefix. The four types
/// where that is not the whole difference (`GPUExtent3DDict`, `GPUOrigin2DDict`,
/// `GPUOrigin3DDict`, `WGSLLanguageFeatures`) state the class name in the header
/// instead: `webgpu_members! { GpuExtent3dDict as "GPUExtent3DDict"; .. }`.
#[macro_export]
macro_rules! webgpu_members {
    (
        $name:ident as $class:expr;
        $(
            $(#[$doc:meta])*
            $kind:ident $member:ident (
                $($argument:ident : $argument_type:ty $(as $argument_js:literal)?),* $(,)?
            )
            $(-> $result:ty)?
            $(as $js:literal $($catch:ident)?)?
            ;
        )*
    ) => {
        impl $name {
            $(
                $crate::webgpu_member! {
                    @kind $kind;
                    @class $class;
                    @member $member;
                    @args ($($argument : $argument_type $(as $argument_js)?),*);
                    @ret ($($result)?);
                    // The JavaScript name and `catch` travel together because a
                    // `ty` fragment may be followed by `as` and not by an
                    // identifier, so `-> T as "js" catch` can only be parsed with
                    // everything after the type in one optional group.
                    @tail ($(as $js $($catch)?)?);
                    @docs [$(#[$doc])*]
                }
            )*
        }
    };

    // No JavaScript class name given, so the Rust name stands in for it.
    ($name:ident; $($members:tt)*) => {
        $crate::webgpu_members! { $name as ::core::stringify!($name); $($members)* }
    };
}

/// Declares the members of one WebGPU dictionary.
///
/// A dictionary is a plain `{}` — there is no `GPUBufferDescriptor` class to
/// construct or to test with `instanceof` — so alongside the members
/// [`webgpu_members!`] accepts, this takes a `new` form that builds the object and
/// fills in the properties it is given:
///
/// ```ignore
/// webgpu_dictionary! {
///     GpuBufferDescriptor;
///     new new_with_f64(size: f64 as "size", usage: u32 as "usage");
///     setter set_label(val: &str) as "label";
///     setter set_mapped_at_creation(val: bool) as "mappedAtCreation";
/// }
/// ```
///
/// Each `new` argument carries the JavaScript property it is written to, and they
/// are written in the order they are listed. The `new` name is the Rust name only:
/// `web-sys` generates one constructor per required-member overload
/// (`new_with_f64`), and all of them build the same object.
#[macro_export]
macro_rules! webgpu_dictionary {
    ($($declaration:tt)*) => {
        $crate::webgpu_members! { $($declaration)* }
    };
}

/// Emits one member of an interface or dictionary.
///
/// Invoked by [`webgpu_members!`] once per declared member, with the parts it
/// parsed labelled, because a single repetition cannot branch on the member's
/// kind and a matcher can only branch on tokens it sees first.
#[macro_export]
#[doc(hidden)]
macro_rules! webgpu_member {
    // `target[name] = value`.
    (
        @kind setter;
        @class $class:expr;
        @member $member:ident;
        @args ($value:ident : $value_type:ty);
        @ret ();
        @tail (as $js:literal);
        @docs [$(#[$doc:meta])*]
    ) => {
        $(#[$doc])*
        pub fn $member(&self, $value: $value_type) {
            $crate::dsl::set_property(
                $crate::dsl::js_of(self),
                $crate::js_member_name!($js),
                &$crate::dsl::as_js(&$value),
                $crate::dsl::Operation::new($class, $js),
            );
        }
    };

    // A dictionary whose constructor takes nothing needs no property writes at
    // all. `webgpu_default!` gives these a `Default` as well; it is a separate
    // macro because a trait impl cannot be emitted from inside an `impl` block.
    (
        @kind new;
        @class $class:expr;
        @member $member:ident;
        @args ();
        @ret ();
        @tail ();
        @docs [$(#[$doc:meta])*]
    ) => {
        $(#[$doc])*
        #[must_use]
        pub fn $member() -> Self {
            $crate::dsl::new_dictionary()
        }
    };

    // `{ property: value, .. }`, which is all a WebGPU dictionary ever is.
    (
        @kind new;
        @class $class:expr;
        @member $member:ident;
        @args ($($argument:ident : $argument_type:ty as $argument_js:literal),*);
        @ret ();
        @tail ();
        @docs [$(#[$doc:meta])*]
    ) => {
        $(#[$doc])*
        #[must_use]
        pub fn $member($($argument: $argument_type),*) -> Self {
            let dictionary: Self = $crate::dsl::new_dictionary();
            $(
                $crate::dsl::set_property(
                    $crate::dsl::js_of(&dictionary),
                    $crate::js_member_name!($argument_js),
                    &$crate::dsl::as_js(&$argument),
                    $crate::dsl::Operation::new($class, $argument_js),
                );
            )*
            dictionary
        }
    };

    // `target[name](..arguments)`, discarding the result.
    (
        @kind method;
        @class $class:expr;
        @member $member:ident;
        @args ($($argument:ident : $argument_type:ty),*);
        @ret ();
        @tail (as $js:literal);
        @docs [$(#[$doc:meta])*]
    ) => {
        $(#[$doc])*
        pub fn $member(&self $(, $argument: $argument_type)*) {
            let (): () = $crate::dsl::call_method(
                $crate::dsl::js_of(self),
                $crate::js_member_name!($js),
                &[$($crate::dsl::as_js(&$argument)),*],
                $crate::dsl::Operation::new($class, $js),
            );
        }
    };

    // The same, where a throw is the caller's to handle.
    (
        @kind method;
        @class $class:expr;
        @member $member:ident;
        @args ($($argument:ident : $argument_type:ty),*);
        @ret ();
        @tail (as $js:literal catch);
        @docs [$(#[$doc:meta])*]
    ) => {
        $(#[$doc])*
        pub fn $member(
            &self $(, $argument: $argument_type)*
        ) -> ::core::result::Result<(), $crate::support::Error> {
            $crate::dsl::try_call_method(
                $crate::dsl::js_of(self),
                $crate::js_member_name!($js),
                &[$($crate::dsl::as_js(&$argument)),*],
            )
        }
    };

    // `target[name](..arguments)`, converting the result.
    (
        @kind method;
        @class $class:expr;
        @member $member:ident;
        @args ($($argument:ident : $argument_type:ty),*);
        @ret ($result:ty);
        @tail (as $js:literal);
        @docs [$(#[$doc:meta])*]
    ) => {
        $(#[$doc])*
        pub fn $member(&self $(, $argument: $argument_type)*) -> $result {
            $crate::dsl::call_method(
                $crate::dsl::js_of(self),
                $crate::js_member_name!($js),
                &[$($crate::dsl::as_js(&$argument)),*],
                $crate::dsl::Operation::new($class, $js),
            )
        }
    };

    // The same, where a throw is the caller's to handle.
    (
        @kind method;
        @class $class:expr;
        @member $member:ident;
        @args ($($argument:ident : $argument_type:ty),*);
        @ret ($result:ty);
        @tail (as $js:literal catch);
        @docs [$(#[$doc:meta])*]
    ) => {
        $(#[$doc])*
        pub fn $member(
            &self $(, $argument: $argument_type)*
        ) -> ::core::result::Result<$result, $crate::support::Error> {
            $crate::dsl::try_call_method(
                $crate::dsl::js_of(self),
                $crate::js_member_name!($js),
                &[$($crate::dsl::as_js(&$argument)),*],
            )
        }
    };

    // `target[name]`.
    (
        @kind getter;
        @class $class:expr;
        @member $member:ident;
        @args ();
        @ret ($result:ty);
        @tail (as $js:literal);
        @docs [$(#[$doc:meta])*]
    ) => {
        $(#[$doc])*
        pub fn $member(&self) -> $result {
            $crate::dsl::get_property(
                $crate::dsl::js_of(self),
                $crate::js_member_name!($js),
                $crate::dsl::Operation::new($class, $js),
            )
        }
    };

    // Nothing above matched, which is a mistyped declaration rather than a member
    // shape WebGPU has: only a method throws, and every other shape is fixed.
    (
        @kind $kind:ident;
        @class $class:expr;
        @member $member:ident;
        @args $args:tt;
        @ret $result:tt;
        @tail $tail:tt;
        @docs $docs:tt
    ) => {
        ::core::compile_error!(::core::concat!(
            "`",
            ::core::stringify!($kind),
            " ",
            ::core::stringify!($member),
            "` is not a member declaration this DSL has a shape for: a getter takes \
             no arguments and needs a `-> T`, a setter takes exactly one argument \
             and no `-> T`, a `new` needs an `as \"property\"` on every argument and \
             no `as` of its own, and only a method may be declared `catch`",
        ));
    };
}

/// Declares one WebGPU string enumeration.
///
/// ```ignore
/// webgpu_enum! {
///     GpuTextureFormat as "GPUTextureFormat";
///     Rgba8unorm = "rgba8unorm",
///     Bgra8unorm = "bgra8unorm",
/// }
/// ```
///
/// The value crossing the boundary is the string: these are plain fieldless Rust
/// enums with no [`JsCast`], because a JavaScript string is not an object and
/// `instanceof` has nothing to test.
///
/// An unrecognised string becomes the hidden `__Invalid` variant rather than a
/// panic. WebGPU gains enumerators — texture formats above all — faster than these
/// declarations are re-derived, and a device reporting one the bindings have not
/// heard of should not end the process; the variant is hidden, and the enums are
/// `#[non_exhaustive]`, so no caller can come to depend on it.
#[macro_export]
macro_rules! webgpu_enum {
    (
        $(#[$doc:meta])*
        $name:ident as $js_name:literal;
        $(
            $(#[$variant_doc:meta])*
            $variant:ident = $js:literal
        ),* $(,)?
    ) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum $name {
            $(
                $(#[$variant_doc])*
                $variant,
            )*
            /// A string this declaration does not list.
            #[doc(hidden)]
            __Invalid,
        }

        impl $name {
            #[doc = ::core::concat!(
                "The name this enumeration has in WebGPU's IDL: `", $js_name, "`."
            )]
            pub const JS_NAME: &'static str = $js_name;

            /// The JavaScript string for this enumerator.
            ///
            /// `__Invalid` has none, and answers with the empty string — which
            /// WebGPU rejects wherever the enumeration is accepted, so a value that
            /// came back unrecognised cannot be passed off as a valid one.
            #[must_use]
            pub fn as_js_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $js,)*
                    Self::__Invalid => "",
                }
            }

            /// The enumerator `value` names, or `__Invalid` if it names none.
            #[must_use]
            pub fn from_js_str(value: &str) -> Self {
                match value {
                    $($js => Self::$variant,)*
                    _ => Self::__Invalid,
                }
            }
        }

        impl $crate::napi::convert::AsJs for $name {
            #[inline]
            fn as_js(&self) -> $crate::napi::value::JsValue {
                $crate::napi::value::JsValue::from_str(self.as_js_str())
            }
        }

        impl $crate::napi::convert::FromJs for $name {
            #[inline]
            fn from_js(value: $crate::napi::value::JsValue) -> Self {
                match value.as_string() {
                    ::core::option::Option::Some(value) => Self::from_js_str(&value),
                    ::core::option::Option::None => Self::__Invalid,
                }
            }
        }

        impl ::core::convert::From<$name> for $crate::napi::value::JsValue {
            #[inline]
            fn from(value: $name) -> Self {
                Self::from_str(value.as_js_str())
            }
        }
    };
}

/// The JavaScript name of a member as a `&CStr`, resolved at compile time.
///
/// Node-API's `*_named_property` entry points want a NUL-terminated name, and the
/// declarations spell the JavaScript name as an ordinary string literal. `concat!`
/// and `CStr::from_bytes_with_nul` are both `const`, so this produces the same
/// static the `c"…"` literal would, with the check that the name holds no NUL of
/// its own — and, being a `const`, no call allocates for its own name.
#[macro_export]
#[doc(hidden)]
macro_rules! js_member_name {
    ($name:literal) => {{
        const NAME: &::core::ffi::CStr =
            match ::core::ffi::CStr::from_bytes_with_nul(::core::concat!($name, "\0").as_bytes()) {
                ::core::result::Result::Ok(name) => name,
                ::core::result::Result::Err(_) => {
                    ::core::panic!("a JavaScript member name cannot contain a NUL byte")
                }
            };
        NAME
    }};
}

/// What a panic calls the operation that threw.
///
/// The name is asked for only once a call has already failed, so building it may
/// allocate: the path that succeeds never touches it.
pub(crate) trait OperationName {
    /// The operation's name, for the panic message.
    fn operation_name(&self) -> String;
}

/// A member of a JavaScript class, named as WebGPU's IDL spells it.
///
/// The class and the member are kept apart rather than joined into one string
/// because both are compile-time constants and the joined form is only ever wanted
/// on the failing path.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Operation {
    class: &'static str,
    member: &'static str,
}

impl Operation {
    /// Names `class.member`. `class` is the binding's Rust name unless its
    /// declaration gave the JavaScript class name explicitly.
    #[inline]
    pub(crate) const fn new(class: &'static str, member: &'static str) -> Self {
        Self { class, member }
    }
}

impl OperationName for Operation {
    /// `GPUDevice.createBuffer`.
    ///
    /// WebGPU's IDL spells every class with an upper-case prefix that the generated
    /// Rust name spells in title case, so the prefix is restored here and the panic
    /// quotes the name the specification uses. A class whose JavaScript name differs
    /// from its Rust name by more than that states it in its declaration and arrives
    /// here already correct.
    fn operation_name(&self) -> String {
        let member = self.member;
        if let Some(rest) = self.class.strip_prefix("Gpu") {
            alloc::format!("GPU{rest}.{member}")
        } else if let Some(rest) = self.class.strip_prefix("Wgsl") {
            alloc::format!("WGSL{rest}.{member}")
        } else {
            alloc::format!("{}.{member}", self.class)
        }
    }
}

/// The hand-written bindings name their operation outright.
impl OperationName for &str {
    fn operation_name(&self) -> String {
        String::from(*self)
    }
}

/// Reads a property declared without `catch`, converting the result.
#[track_caller]
pub(crate) fn get_property<T: FromJs>(
    target: &JsValue,
    name: &CStr,
    operation: impl OperationName,
) -> T {
    T::from_js(unwrap_named(rt::get(target, name), &operation))
}

/// Writes a property declared without `catch`.
#[track_caller]
pub(crate) fn set_property(
    target: &JsValue,
    name: &CStr,
    value: &JsValue,
    operation: impl OperationName,
) {
    unwrap_named(rt::set(target, name, value), &operation);
}

/// Calls a method declared without `catch`, converting the result.
#[track_caller]
pub(crate) fn call_method<T: FromJs>(
    target: &JsValue,
    name: &CStr,
    arguments: &[JsValue],
    operation: impl OperationName,
) -> T {
    T::from_js(unwrap_named(
        rt::call_method(target, name, arguments),
        &operation,
    ))
}

/// `new globalThis[class](..arguments)` for a constructor declared without `catch`.
#[track_caller]
pub(crate) fn construct<T: JsCast>(
    class: &CStr,
    arguments: &[JsValue],
    operation: impl OperationName,
) -> T {
    rt::cast(unwrap_named(rt::construct(class, arguments), &operation))
}

/// Calls a method declared `catch`, handing the caller whatever it threw.
///
/// There is no property counterpart because WebGPU declares no accessor that
/// throws: the IDL's `[Throws]` sits on operations only, so a property read that
/// fails is a broken object rather than something a call site can act on.
pub(crate) fn try_call_method<T: FromJs>(
    target: &JsValue,
    name: &CStr,
    arguments: &[JsValue],
) -> Result<T, Error> {
    rt::call_method(target, name, arguments)
        .map(T::from_js)
        .map_err(Error::from)
}

/// A new `{}` as the dictionary type that declared it.
///
/// A WebGPU dictionary has no class of its own: it is an object literal whose
/// properties the API reads, so there is nothing to look up and nothing to
/// `instanceof`.
pub(crate) fn new_dictionary<T: JsCast>() -> T {
    T::unchecked_from_js(JsValue::from(Object::new()))
}

/// The untyped value behind a binding handle.
///
/// Each generated type has an `AsRef` impl per ancestor, so a bare `self.as_ref()`
/// in an expansion would be ambiguous; the bound here pins the one that matters.
#[inline]
pub(crate) fn js_of<T: AsRef<JsValue>>(value: &T) -> &JsValue {
    value.as_ref()
}

/// The JavaScript value for an argument, so an expansion need not name the trait.
#[inline]
pub(crate) fn as_js<T: AsJs + ?Sized>(value: &T) -> JsValue {
    value.as_js()
}

/// The result of an operation declared without `catch`.
///
/// Joins the operation's name only on the failing path, so a call that succeeds
/// pays nothing for the name it would have been reported under.
#[track_caller]
fn unwrap_named<T>(result: Result<T, JsValue>, operation: &dyn OperationName) -> T {
    match result {
        Ok(value) => value,
        Err(error) => rt::unwrap_js(Err(error), &operation.operation_name()),
    }
}

/// The parts of the machinery that are decided in Rust, and so can be checked
/// without a JavaScript engine to run against.
#[cfg(test)]
mod tests {
    use super::{Operation, OperationName};

    /// A panic quotes the class the way WebGPU's IDL spells it, from the Rust name
    /// the declaration gave.
    #[test]
    fn operation_names_use_the_idl_capitalisation() {
        assert_eq!(
            Operation::new("GpuDevice", "createBuffer").operation_name(),
            "GPUDevice.createBuffer"
        );
        assert_eq!(
            Operation::new("WgslLanguageFeatures", "size").operation_name(),
            "WGSLLanguageFeatures.size"
        );
    }

    /// A declaration that gave the JavaScript class name is left alone: that is how
    /// the four classes whose names differ by more than the prefix are spelled.
    #[test]
    fn operation_names_keep_an_explicit_class() {
        assert_eq!(
            Operation::new("GPUExtent3DDict", "depthOrArrayLayers").operation_name(),
            "GPUExtent3DDict.depthOrArrayLayers"
        );
    }

    /// The hand-written bindings name the whole operation themselves.
    #[test]
    fn a_plain_name_is_its_own_operation() {
        assert_eq!(
            "ArrayBuffer.byteLength".operation_name(),
            "ArrayBuffer.byteLength"
        );
    }

    /// A member name reaches Node-API NUL-terminated, built at compile time.
    #[test]
    fn member_names_are_c_strings() {
        assert_eq!(
            crate::js_member_name!("createBuffer").to_bytes(),
            b"createBuffer"
        );
    }
}

/// `Default` for a dictionary whose constructor takes no arguments.
///
/// 20 of the 64 WebGPU dictionary constructors take nothing, which makes them
/// default values in the ordinary sense. `webgpu_members!` cannot emit this itself:
/// its members expand inside an `impl` block, and a trait impl cannot nest there.
/// `tools/generate.py` emits a call to this next to each such dictionary.
#[macro_export]
macro_rules! webgpu_default {
    ($name:ident, $constructor:ident) => {
        impl ::core::default::Default for $name {
            #[inline]
            fn default() -> Self {
                Self::$constructor()
            }
        }
    };
}
