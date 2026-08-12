//! `ArrayBuffer` and the typed array views over it.
//!
//! On Wasm, Emnapi's memory-view extension gives [`Uint8Array::view`] the same
//! zero-copy contract as wasm-bindgen: JavaScript sees a temporary view directly
//! over this module's linear memory. `Uint8Array::from` remains the explicitly
//! owned-copy route.

use alloc::vec::Vec;
use core::ptr;

use napi_sys as sys;

use crate::napi::env;
use crate::napi::rt;
use crate::napi::value::JsValue;

use super::Object;

#[cfg(target_family = "wasm")]
#[link(wasm_import_module = "emnapi")]
unsafe extern "C" {
    /// Creates a JavaScript typed-array view directly over this module's linear
    /// memory. This extension is part of Emnapi 1.x as well as 2.x.
    fn emnapi_create_memory_view(
        env: sys::napi_env,
        view_type: EmnapiMemoryViewType,
        external_data: *mut core::ffi::c_void,
        byte_length: usize,
        finalize_cb: sys::napi_finalize,
        finalize_hint: *mut core::ffi::c_void,
        result: *mut sys::napi_value,
    ) -> sys::napi_status;
}

/// The C ABI represents `emnapi_memory_view_type` as an `int` on Wasm.
#[cfg(target_family = "wasm")]
type EmnapiMemoryViewType = core::ffi::c_int;

// Keep these separate from Node-API's `napi_typedarray_type`: the two enums
// currently assign the same values, but they are different public ABIs. These
// discriminants are `emnapi_uint8_array` and `emnapi_uint32_array` respectively
// in Emnapi 1.x's `emnapi.h`.
#[cfg(target_family = "wasm")]
const EMNAPI_UINT8_ARRAY: EmnapiMemoryViewType = 1;
#[cfg(target_family = "wasm")]
const EMNAPI_UINT32_ARRAY: EmnapiMemoryViewType = 6;

/// Creates a typed-array view over a live slice in this module's linear memory.
///
/// # Safety
///
/// `data..data + byte_length` must remain live for every JavaScript access to the
/// returned value. The caller must also uphold the aliasing and memory-growth
/// requirements documented on the typed array's `view` method.
#[cfg(target_family = "wasm")]
unsafe fn create_memory_view(
    view_type: EmnapiMemoryViewType,
    data: *const core::ffi::c_void,
    byte_length: usize,
) -> Result<JsValue, JsValue> {
    env::scope(|env| {
        // A zero-length Rust slice may carry a dangling but non-null pointer.
        // Emnapi treats null as the canonical empty view and never dereferences it.
        let external_data = if byte_length == 0 {
            ptr::null_mut()
        } else {
            data.cast_mut()
        };
        let mut value = ptr::null_mut();
        // SAFETY: the caller guarantees that the non-empty byte range is live in
        // this module's linear memory. `None` is intentional: Rust owns the bytes,
        // so JavaScript must neither free nor finalize them. All other pointers are
        // valid out-pointers or null as required by Emnapi's C ABI.
        unsafe {
            env::check(
                emnapi_create_memory_view(
                    env,
                    view_type,
                    external_data,
                    byte_length,
                    None,
                    ptr::null_mut(),
                    &mut value,
                ),
                "emnapi_create_memory_view",
            )?;
            if value.is_null() {
                return Err(JsValue::from_str(
                    "napi-rs-webgpu: emnapi_create_memory_view returned a null value",
                ));
            }
            Ok(JsValue::from_napi(env, value))
        }
    })
}

js_type! {
    /// The JavaScript `ArrayBuffer`.
    ///
    /// `wgpu` holds one of these for the region `GPUBuffer.getMappedRange` returned
    /// and builds [`Uint8Array`] views onto it; the bytes themselves stay in the
    /// browser until something copies them out.
    ///
    /// The `byteLength` getter is a property read rather than
    /// `napi_get_arraybuffer_info`, which would also hand back a pointer into the
    /// backing store that nothing here should hold.
    ArrayBuffer: [Object, JsValue],
    instanceof(value) { is_arraybuffer(value) },
}

impl ArrayBuffer {
    /// `new ArrayBuffer(length)`.
    pub fn new(length: u32) -> Self {
        crate::dsl::construct(c"ArrayBuffer", &[JsValue::from(length)], "new ArrayBuffer")
    }
}

/// Whether `value` is an `ArrayBuffer`.
///
/// `napi_is_arraybuffer` rather than `instanceof`, so a buffer from another realm —
/// which is what a browser-created buffer may well be — is recognised.
fn is_arraybuffer(value: &JsValue) -> bool {
    if !value.is_object() {
        return false;
    }
    env::scope(|env| {
        // SAFETY: inside a handle scope on `env`.
        unsafe {
            let value = value.to_napi(env)?;
            let mut result = false;
            env::check(
                sys::napi_is_arraybuffer(env, value, &mut result),
                "napi_is_arraybuffer",
            )?;
            Ok(result)
        }
    })
    .unwrap_or(false)
}

/// Emits one typed array type over `$element`, with `$kind` its Node-API tag.
///
/// The JavaScript class name is the Rust name, which is why `$class` and `$name`
/// look redundant: `$class` is the C string the constructor is looked up by.
macro_rules! typed_array {
    (
        $(#[$doc:meta])*
        $name:ident($element:ty, $class:expr, $kind:expr, $emnapi_kind:expr)
    ) => {
        js_type! {
            $(#[$doc])*
            $name: [Object, JsValue],
            instanceof(value) { is_typedarray(value, $kind) },
        }

        impl $name {
            /// `new` from anything the constructor accepts: a length, an
            /// `ArrayBuffer`, or an iterable of numbers.
            pub fn new(constructor_argument: &JsValue) -> Self {
                crate::dsl::construct(
                    $class,
                    core::slice::from_ref(constructor_argument),
                    concat!("new ", stringify!($name)),
                )
            }

            /// A view onto `buffer` of `length` elements starting at `byte_offset`.
            ///
            /// A real JavaScript view: no bytes are copied.
            pub fn new_with_byte_offset_and_length(
                buffer: &JsValue,
                byte_offset: u32,
                length: u32,
            ) -> Self {
                crate::dsl::construct(
                    $class,
                    &[
                        buffer.clone(),
                        JsValue::from(byte_offset),
                        JsValue::from(length),
                    ],
                    concat!("new ", stringify!($name)),
                )
            }

            /// The `ArrayBuffer` this view reads from.
            pub fn buffer(&self) -> ArrayBuffer {
                rt::cast(rt::unwrap_js(
                    rt::get(self.js(), c"buffer"),
                    concat!(stringify!($name), ".buffer"),
                ))
            }

            /// The number of elements in the view.
            pub fn length(&self) -> u32 {
                crate::dsl::get_property(
                    self.js(),
                    c"length",
                    concat!(stringify!($name), ".length"),
                )
            }

            /// `this.set(src, offset)`: stores `src`'s elements into this view.
            ///
            /// JavaScript's own `set`, so the store lands in whatever the view reads
            /// from — see [`Uint8Array`] for why this is not done through a data
            /// pointer.
            pub fn set(&self, src: &JsValue, offset: u32) {
                let _: () = crate::dsl::call_method(
                    self.js(),
                    c"set",
                    &[src.clone(), JsValue::from(offset)],
                    concat!(stringify!($name), ".set"),
                );
            }

            /// A JavaScript typed array viewing `rust` without copying on Wasm.
            ///
            /// # Safety
            ///
            /// On Wasm, the returned object aliases the slice but is disconnected
            /// from its lifetime. The slice must remain live and unmodified for all
            /// JavaScript reads; JavaScript must not mutate the view while Rust
            /// assumes a shared reference, and the view must not outlive a memory
            /// resize that invalidates its backing buffer.
            ///
            /// The non-Wasm compile-time fallback copies because there is no Wasm
            /// linear memory to view.
            pub unsafe fn view(rust: &[$element]) -> Self {
                #[cfg(target_family = "wasm")]
                {
                    let created = unsafe {
                        create_memory_view(
                            $emnapi_kind,
                            rust.as_ptr().cast::<core::ffi::c_void>(),
                            core::mem::size_of_val(rust),
                        )
                    };
                    rt::cast(rt::unwrap_js(
                        created,
                        concat!("creating a zero-copy JavaScript ", stringify!($name)),
                    ))
                }
                #[cfg(not(target_family = "wasm"))]
                {
                    Self::from_slice(rust)
                }
            }

            /// A JavaScript typed array holding a copy of `slice`.
            fn from_slice(slice: &[$element]) -> Self {
                #[cfg(target_family = "wasm")]
                {
                    // SAFETY: the temporary view is consumed synchronously by the
                    // JavaScript typed-array constructor while `slice` is live. The
                    // constructor creates a distinct backing store, so the returned
                    // array no longer aliases Rust memory.
                    let view = unsafe { Self::view(slice) };
                    Self::new(&view)
                }
                #[cfg(not(target_family = "wasm"))]
                {
                let byte_length = core::mem::size_of_val(slice);
                let created = env::scope(|env| {
                    // SAFETY: inside a handle scope on `env`.
                    // `napi_create_arraybuffer` reports `byte_length` writable bytes
                    // owned by the new buffer. `napi_create_typedarray` receives that
                    // same buffer with a zero offset and a matching element count.
                    unsafe {
                        let mut data = ptr::null_mut();
                        let mut buffer = ptr::null_mut();
                        env::check(
                            sys::napi_create_arraybuffer(
                                env,
                                byte_length,
                                &mut data,
                                &mut buffer,
                            ),
                            "napi_create_arraybuffer",
                        )?;
                        if byte_length != 0 && !data.is_null() {
                            ptr::copy_nonoverlapping(
                                slice.as_ptr().cast::<u8>(),
                                data.cast::<u8>(),
                                byte_length,
                            );
                        }
                        let mut array = ptr::null_mut();
                        env::check(
                            sys::napi_create_typedarray(
                                env,
                                $kind,
                                slice.len(),
                                buffer,
                                0,
                                &mut array,
                            ),
                            "napi_create_typedarray",
                        )?;
                        Ok(JsValue::from_napi(env, array))
                    }
                });
                rt::cast(rt::unwrap_js(
                    created,
                    concat!("creating a JavaScript ", stringify!($name)),
                ))
                }
            }

            /// This view's elements in a new vector.
            /// Reads the view's elements into `dst`, which must be the same length.
            ///
            /// Private because `to_vec` is the only caller: `wgpu` reads a mapped
            /// buffer by taking the whole thing, never into a buffer it already has.
            fn copy_to(&self, dst: &mut [$element]) {
                let length = self.length() as usize;
                assert_eq!(
                    length,
                    dst.len(),
                    concat!(stringify!($name), "::copy_to needs a destination of the \
                             same length as the view")
                );
                let copied = env::scope(|env| {
                    // SAFETY: inside a handle scope on `env`. `typedarray_elements`
                    // returns a pointer to the view's first element, aligned for that
                    // element type, only once it has confirmed the element type is
                    // `$kind`; the copy reads it as `$element` and never more elements
                    // than both sides hold.
                    unsafe {
                        let (data, available) = typedarray_elements(env, self.js(), $kind)?;
                        let count = available.min(dst.len());
                        if count != 0 && !data.is_null() {
                            ptr::copy_nonoverlapping(
                                data.cast::<$element>(),
                                dst.as_mut_ptr(),
                                count,
                            );
                        }
                        Ok(())
                    }
                });
                rt::unwrap_js(copied, concat!("reading a JavaScript ", stringify!($name)));
            }

            pub fn to_vec(&self) -> Vec<$element> {
                // Allocated before the handle scope opens: `copy_to` holds a raw
                // pointer into the runtime's memory, and nothing else should run
                // while it does.
                let mut out: Vec<$element> = alloc::vec![0; self.length() as usize];
                self.copy_to(&mut out);
                out
            }
        }

        impl From<&[$element]> for $name {
            fn from(slice: &[$element]) -> Self {
                Self::from_slice(slice)
            }
        }
    };
}

typed_array! {
    /// The JavaScript `Uint8Array`.
    ///
    /// This is how mapped buffer contents cross the boundary: `wgpu` keeps one over
    /// the `ArrayBuffer` from `GPUBuffer.getMappedRange` and copies through it in
    /// both directions.
    ///
    /// # Copy semantics
    ///
    /// On Wasm, [`Uint8Array::view`] calls Emnapi's
    /// `emnapi_create_memory_view`, which constructs the equivalent of
    /// `new Uint8Array(memory.buffer, ptr, len)`. No bytes move: JavaScript reads
    /// directly from the Rust slice. Its `unsafe` contract therefore matches the
    /// wasm-bindgen API it stands in for.
    ///
    /// `Uint8Array::from(slice)` is deliberately different. It feeds that
    /// temporary memory view to JavaScript's `Uint8Array` constructor, which makes
    /// a new backing store and copies the elements before the Rust borrow ends.
    ///
    /// Reading a JavaScript-owned view still copies: [`Uint8Array::to_vec`] asks
    /// `napi_get_typedarray_info` for the element pointer and copies out of it.
    ///
    /// A view *created in JavaScript* —
    /// [`Uint8Array::new_with_byte_offset_and_length`], which is
    /// `new Uint8Array(buffer, offset, length)` — is an ordinary zero-copy
    /// JavaScript view onto its `ArrayBuffer`, exactly as on the web.
    ///
    /// Writes never go through a data pointer: [`Uint8Array::set`] calls JavaScript's
    /// own `TypedArray.prototype.set`. A pointer obtained for a buffer that does not
    /// live in wasm memory may be a staging copy the runtime has no obligation to
    /// write back, so the JavaScript method is the only way to be sure the store
    /// lands.
    Uint8Array(
        u8,
        c"Uint8Array",
        sys::TypedarrayType::uint8_array,
        EMNAPI_UINT8_ARRAY
    )
}

typed_array! {
    /// The JavaScript `Uint32Array`.
    ///
    /// Declared by the generated bindings for the `setBindGroup` overload that takes
    /// a typed array of dynamic offsets. `wgpu` calls the `u32` slice overload
    /// instead, so nothing constructs one of these today.
    Uint32Array(
        u32,
        c"Uint32Array",
        sys::TypedarrayType::uint32_array,
        EMNAPI_UINT32_ARRAY
    )
}

/// Whether `value` is a typed array of `kind`.
///
/// `napi_get_typedarray_info` rather than `instanceof`, so a typed array from
/// another realm is recognised and a `Uint8Array` is not mistaken for a
/// `Uint32Array`.
fn is_typedarray(value: &JsValue, kind: sys::napi_typedarray_type) -> bool {
    if !value.is_object() {
        return false;
    }
    env::scope(|env| {
        // SAFETY: inside a handle scope on `env`, and the element query is only made
        // for a value `napi_is_typedarray` accepted. The pointer it returns is
        // discarded, so nothing outlives the scope.
        unsafe {
            let raw = value.to_napi(env)?;
            let mut is_typedarray = false;
            env::check(
                sys::napi_is_typedarray(env, raw, &mut is_typedarray),
                "napi_is_typedarray",
            )?;
            // A different element type makes the element query fail, which is the same
            // answer as "not this type".
            Ok(is_typedarray && typedarray_elements(env, value, kind).is_ok())
        }
    })
    .unwrap_or(false)
}

/// The first element of a typed array and how many elements follow it.
///
/// The pointer Node-API reports is already adjusted by the view's `byteOffset`, so
/// it addresses element zero of the view rather than of the underlying buffer.
///
/// # Safety
///
/// Must be called inside a handle scope on `env`. The returned pointer is only valid
/// inside that scope, and only for reading `kind`-sized elements.
unsafe fn typedarray_elements(
    env: sys::napi_env,
    value: &JsValue,
    kind: sys::napi_typedarray_type,
) -> Result<(*mut core::ffi::c_void, usize), JsValue> {
    let value = value.to_napi(env)?;
    let mut actual = 0;
    let mut length = 0;
    let mut data = ptr::null_mut();
    let mut buffer = ptr::null_mut();
    let mut byte_offset = 0;
    env::check(
        sys::napi_get_typedarray_info(
            env,
            value,
            &mut actual,
            &mut length,
            &mut data,
            &mut buffer,
            &mut byte_offset,
        ),
        "napi_get_typedarray_info",
    )?;
    if actual != kind {
        // Reading a `Uint32Array` as bytes (or the reverse) would misread every
        // element, so this is a hard error rather than a best-effort copy.
        return Err(JsValue::from_str(
            "napi-rs-webgpu: typed array has a different element type than the \
             binding declares",
        ));
    }
    Ok((data, length))
}

/// The coercions `wgpu` performs on these types, checked by compiling.
///
/// `js-sys` declares the typed-array constructors and `set` as taking a bare
/// value, and `wgpu` calls them with an `&ArrayBuffer` and an `&Uint8Array`; that
/// only works if the `Deref` chain runs from the view down to [`JsValue`].
/// Nothing here runs.
#[cfg(test)]
#[allow(
    dead_code,
    reason = "these are checked by being compiled, not by being called"
)]
mod coercion_checks {
    use super::{ArrayBuffer, JsValue, Object, Uint8Array};

    /// `Uint8Array::new_with_byte_offset_and_length(array_buffer, ..)` in
    /// `WebBuffer::get_mapped_range`.
    fn view_over_an_array_buffer(buffer: &ArrayBuffer) -> Uint8Array {
        Uint8Array::new_with_byte_offset_and_length(buffer, 0, 0)
    }

    /// `actual_mapping.set(&Uint8Array::view(..), 0)` in
    /// `WebBufferMappedRange::drop`.
    fn store_one_view_into_another(destination: &Uint8Array, source: &Uint8Array) {
        destination.set(source, 0);
    }

    /// An `ArrayBuffer` is an object, which is what a WebGPU descriptor field
    /// takes, and reaches the bottom of the chain.
    fn an_array_buffer_is_an_object(buffer: &ArrayBuffer) -> (&Object, &JsValue) {
        (buffer, buffer)
    }

    /// So is a view, which descends from `Object` rather than from the buffer it
    /// reads.
    fn a_view_is_an_object(view: &Uint8Array) -> (&Object, &JsValue) {
        (view, view)
    }
}
