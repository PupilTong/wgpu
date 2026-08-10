//! The `#[wasm_bindgen]` attribute, lowered onto Node-API instead of onto wasm
//! imports.
//!
//! `wgpu`'s WebGPU backend and the `web-sys` bindings it vendors are written
//! against wasm-bindgen's attribute. wasm-bindgen turns each declaration in an
//! `extern "C"` block into a wasm import from a placeholder module, which its CLI
//! resolves when it emits JavaScript glue — and it only emits that glue for
//! `wasm32-unknown-unknown`. This attribute accepts the same declarations and emits
//! the operation the glue would have performed instead: a property read, a property
//! write, a method call, a construction, made through
//! [`wgpu_napi_web`](https://docs.rs/wgpu-napi-web)'s Node-API runtime against the
//! same live JavaScript objects.
//!
//! # What it accepts
//!
//! Only the subset the vendored WebGPU bindings and the two hand-written blocks in
//! `wgpu`'s backend use, with keys in any order:
//!
//! | on | keys |
//! | -- | ---- |
//! | `pub type X;` | `extends = Path` (repeatable), `js_name`, `typescript_type`, or nothing |
//! | `pub fn f(this: &X, …) -> R;` | `method`, `structural`, `catch`, `getter[ = "name"]`, `setter[ = "name"]`, `js_class`, `js_name` |
//! | `pub fn new(…) -> R;` | `constructor`, `catch`, `js_class`, `js_name` |
//! | `pub enum X { V = "js-string", … }` | nothing |
//!
//! `structural` and `typescript_type` are accepted and carry no meaning here: every
//! operation is looked up by name on the receiver at call time, which is what
//! `structural` asks for, and nothing emits TypeScript. Anything else — a free
//! function, a static method, `js_namespace`, `variadic` — is rejected with a
//! `compile_error!` naming the form, so a declaration is never silently dropped.
//!
//! # What it emits
//!
//! Generated code names the runtime by the absolute path `::wgpu_napi_web::__rt`.
//! The bindings import this family under an alias (`crate::js::wasm_bindgen`), so
//! the crate name is not in scope where the expansion lands.
//!
//! An imported type becomes a `#[repr(transparent)]` struct over its first
//! `extends` (or over `JsValue`), with `Deref`, `AsRef` and `From` for every
//! ancestor, `JsCast`, `AsJs`, `FromJs` and `Promising`. A string enum becomes a
//! plain fieldless enum with `as_js_str` / `from_js_str` and the same two
//! conversions.

mod codegen;
mod parse;

use proc_macro::TokenStream;

/// Lowers wasm-bindgen import declarations onto Node-API calls.
///
/// See the [crate documentation](crate) for the accepted forms.
#[proc_macro_attribute]
pub fn wasm_bindgen(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut expansion = proc_macro2::TokenStream::new();

    // Every use in `wgpu` writes the outer attribute bare; the per-declaration keys
    // live on the items inside the block. Rejecting arguments here keeps a key that
    // would have to change the whole block from being read and ignored. The item is
    // still expanded, so one report is all that reaches the user.
    if !attr.is_empty() {
        let attr = proc_macro2::TokenStream::from(attr);
        expansion.extend(
            syn::Error::new_spanned(
                &attr,
                "`#[wasm_bindgen]`: arguments are only supported on the declarations inside \
                 an `extern \"C\"` block, not on the block or enum itself",
            )
            .to_compile_error(),
        );
    }

    match syn::parse::<syn::Item>(item) {
        Ok(item) => expansion.extend(codegen::expand(item)),
        Err(error) => expansion.extend(error.to_compile_error()),
    }
    expansion.into()
}
