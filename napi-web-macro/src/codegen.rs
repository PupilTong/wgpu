//! Lowering the parsed declarations onto `wgpu_napi_web::__rt` calls.
//!
//! wasm-bindgen turns each `extern "C"` declaration into a wasm import that its
//! CLI later wires to generated JavaScript. Nothing generates JavaScript here, so
//! each declaration becomes the Node-API operation the glue would have performed —
//! a property read, a property write, a method call, a construction — named at
//! compile time by a C string literal and resolved on the live object at call time.
//!
//! Generated code names the runtime by the absolute path `::wgpu_napi_web::__rt`.
//! The bindings import the crate under the alias `crate::js::wasm_bindgen`, so the
//! name `wgpu_napi_web` is not in scope where the expansion lands; only an absolute
//! path resolves.

use std::ffi::CString;

use proc_macro2::{Literal, Span, TokenStream};
use quote::{quote, ToTokens};
use syn::punctuated::Punctuated;
use syn::{
    Attribute, Expr, Fields, FnArg, ForeignItem, ForeignItemFn, ForeignItemType, Ident, Item,
    ItemEnum, ItemForeignMod, Lit, Pat, PathArguments, ReturnType, Type,
};

use crate::parse::{self, Attrs};

/// The absolute path to the runtime the expansion calls into.
fn rt() -> TokenStream {
    quote!(::wgpu_napi_web::__rt)
}

/// `::wgpu_napi_web::__rt::JsValue`, the one ABI every binding passes through.
fn js_value() -> TokenStream {
    let rt = rt();
    quote!(#rt::JsValue)
}

/// Expands `#[wasm_bindgen]` over one item.
pub(crate) fn expand(item: Item) -> TokenStream {
    match item {
        Item::ForeignMod(block) => foreign_mod(block),
        Item::Enum(item) => string_enum(item),
        other => syn::Error::new_spanned(
            &other,
            "`#[wasm_bindgen]` here supports only an `extern \"C\"` block of imports \
             and a string enum",
        )
        .to_compile_error(),
    }
}

/// Expands one `extern "C"` block: the types it declares, then one inherent `impl`
/// per receiver holding the operations declared on it.
fn foreign_mod(block: ItemForeignMod) -> TokenStream {
    let mut out = TokenStream::new();
    // Grouped by receiver, in declaration order, so the expansion reads like the
    // source rather than as one `impl` per function.
    let mut receivers: Vec<(String, TokenStream, Vec<TokenStream>)> = Vec::new();

    for item in block.items {
        match item {
            ForeignItem::Type(item) => out.extend(foreign_type(item)),
            ForeignItem::Fn(item) => match foreign_fn(item) {
                Ok(operation) => {
                    let key = operation.receiver.to_string();
                    match receivers.iter_mut().find(|(name, _, _)| *name == key) {
                        Some((_, _, items)) => items.push(operation.item),
                        None => receivers.push((key, operation.receiver, vec![operation.item])),
                    }
                }
                Err(error) => out.extend(error.to_compile_error()),
            },
            other => out.extend(
                syn::Error::new_spanned(
                    &other,
                    "`#[wasm_bindgen]`: only `type` and `fn` declarations are supported \
                     inside an `extern \"C\"` block",
                )
                .to_compile_error(),
            ),
        }
    }

    for (_, receiver, items) in receivers {
        out.extend(quote! {
            impl #receiver {
                #(#items)*
            }
        });
    }
    out
}

// -- types ------------------------------------------------------------------

/// Expands `pub type X;` into the `#[repr(transparent)]` wrapper and the trait
/// impls every JavaScript-value type in this crate must have.
fn foreign_type(mut item: ForeignItemType) -> TokenStream {
    let attrs = match parse::take(&mut item.attrs) {
        Ok(attrs) => attrs,
        Err(error) => return error.to_compile_error(),
    };
    if let Err(error) = reject_operation_keys(&attrs, &item.ident) {
        return error.to_compile_error();
    }
    if !item.generics.params.is_empty() {
        return syn::Error::new_spanned(
            &item.generics,
            "`#[wasm_bindgen]`: an imported type cannot be generic",
        )
        .to_compile_error();
    }

    let name = &item.ident;
    let class = attrs.js_name.clone().unwrap_or_else(|| name.unraw_string());
    let class = match c_literal(&class, name.span()) {
        Ok(literal) => literal,
        Err(error) => return error.to_compile_error(),
    };

    let rt = rt();
    let vis = &item.vis;
    let docs = &item.attrs;
    let extra_derives = supplemental_derives(&item.attrs);

    // The first `extends` is the `Deref` parent and the field type, matching how
    // wasm-bindgen lays out an imported type; with no `extends` the parent is
    // `JsValue` itself.
    let parents: Vec<TokenStream> = attrs
        .extends
        .iter()
        .map(ToTokens::to_token_stream)
        .collect();
    let parent = parents.first().cloned().unwrap_or_else(js_value);
    let js_value = js_value();

    // `AsRef<JsValue>` reaches the field directly when the field *is* a `JsValue`,
    // and through the parent's own impl otherwise.
    let as_ref_js_value = if parents.is_empty() {
        quote!(&self.obj)
    } else {
        quote!(::core::convert::AsRef::as_ref(&self.obj))
    };

    // The parent is one field access away. Every further ancestor is reached by
    // reinterpreting the `JsValue`, which needs nothing from the ancestor chain
    // beyond each link being `#[repr(transparent)]`.
    let ancestors = parents.iter().enumerate().map(|(index, ancestor)| {
        let (as_ref_body, from_body) = if index == 0 {
            (quote!(&self.obj), quote!(value.obj))
        } else {
            (
                quote! {
                    #rt::JsCast::unchecked_from_js_ref(
                        ::core::convert::AsRef::<#js_value>::as_ref(self),
                    )
                },
                quote! {
                    #rt::JsCast::unchecked_from_js(
                        ::core::convert::Into::<#js_value>::into(value),
                    )
                },
            )
        };
        quote! {
            impl ::core::convert::AsRef<#ancestor> for #name {
                #[inline]
                fn as_ref(&self) -> &#ancestor {
                    #as_ref_body
                }
            }

            impl ::core::convert::From<#name> for #ancestor {
                #[inline]
                fn from(value: #name) -> Self {
                    #from_body
                }
            }
        }
    });

    quote! {
        #(#docs)*
        #extra_derives
        #[repr(transparent)]
        #vis struct #name {
            obj: #parent,
        }

        impl ::core::ops::Deref for #name {
            type Target = #parent;

            #[inline]
            fn deref(&self) -> &#parent {
                &self.obj
            }
        }

        impl ::core::convert::AsRef<#js_value> for #name {
            #[inline]
            fn as_ref(&self) -> &#js_value {
                #as_ref_js_value
            }
        }

        impl ::core::convert::From<#name> for #js_value {
            #[inline]
            fn from(value: #name) -> Self {
                ::core::convert::Into::into(value.obj)
            }
        }

        impl ::core::convert::From<#js_value> for #name {
            #[inline]
            fn from(value: #js_value) -> Self {
                <Self as #rt::JsCast>::unchecked_from_js(value)
            }
        }

        #(#ancestors)*

        impl #rt::JsCast for #name {
            #[inline]
            fn instanceof(value: &#js_value) -> bool {
                #rt::instance_of(value, #class)
            }

            #[inline]
            fn unchecked_from_js(value: #js_value) -> Self {
                Self {
                    obj: <#parent as #rt::JsCast>::unchecked_from_js(value),
                }
            }

            #[inline]
            fn unchecked_from_js_ref(value: &#js_value) -> &Self {
                // SAFETY: `Self` is `#[repr(transparent)]` over its parent, which is
                // transparent over its own parent all the way down to `JsValue`, so
                // `Self` and `JsValue` have the same layout and the same set of valid
                // values. The reference keeps the lifetime of the one it came from.
                unsafe { &*::core::ptr::from_ref(value).cast::<Self>() }
            }
        }

        impl #rt::AsJs for #name {
            #[inline]
            fn as_js(&self) -> #js_value {
                ::core::clone::Clone::clone(::core::convert::AsRef::<#js_value>::as_ref(self))
            }
        }

        impl #rt::FromJs for #name {
            #[inline]
            fn from_js(value: #js_value) -> Self {
                <Self as #rt::JsCast>::unchecked_from_js(value)
            }
        }

        impl #rt::Promising for #name {
            type Resolution = Self;
        }
    }
}

/// Rejects keys that only make sense on a function.
fn reject_operation_keys(attrs: &Attrs, name: &Ident) -> syn::Result<()> {
    let key = if attrs.method.is_some() {
        "method"
    } else if attrs.constructor.is_some() {
        "constructor"
    } else if attrs.catch.is_some() {
        "catch"
    } else if attrs.getter.is_some() {
        "getter"
    } else if attrs.setter.is_some() {
        "setter"
    } else {
        return Ok(());
    };
    Err(syn::Error::new(
        attrs.span(name.span()),
        format!("`#[wasm_bindgen]`: `{key}` cannot be used on a type declaration"),
    ))
}

// -- operations -------------------------------------------------------------

/// One lowered declaration and the type whose inherent `impl` it belongs in.
struct Operation {
    receiver: TokenStream,
    item: TokenStream,
}

/// Expands one `pub fn …;` declaration.
fn foreign_fn(mut item: ForeignItemFn) -> syn::Result<Operation> {
    let attrs = parse::take(&mut item.attrs)?;
    let signature_span = item.sig.ident.span();

    if !item.sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &item.sig.generics,
            "`#[wasm_bindgen]`: an imported function cannot be generic",
        ));
    }
    if let Some(variadic) = &item.sig.variadic {
        return Err(syn::Error::new_spanned(
            variadic,
            "`#[wasm_bindgen]`: a variadic import is not supported",
        ));
    }
    if attrs.constructor.is_some() && attrs.method.is_some() {
        return Err(syn::Error::new(
            attrs.span(signature_span),
            "`#[wasm_bindgen]`: `constructor` and `method` are mutually exclusive",
        ));
    }
    if attrs.getter.is_some() && attrs.setter.is_some() {
        return Err(syn::Error::new(
            attrs.span(signature_span),
            "`#[wasm_bindgen]`: `getter` and `setter` are mutually exclusive",
        ));
    }

    if attrs.constructor.is_some() {
        constructor(item, &attrs)
    } else if attrs.method.is_some() {
        instance_operation(item, &attrs)
    } else {
        Err(syn::Error::new(
            signature_span,
            "`#[wasm_bindgen]`: this declaration has none of `method`, `getter`, `setter` \
             or `constructor`, and free-function and static-method imports are not \
             supported by wgpu's Node-API stand-in for wasm-bindgen",
        ))
    }
}

/// A `method`, with or without `getter` / `setter`: an inherent method on the type
/// of its first parameter.
fn instance_operation(item: ForeignItemFn, attrs: &Attrs) -> syn::Result<Operation> {
    let ForeignItemFn {
        attrs: docs,
        vis,
        sig,
        ..
    } = item;
    let name = sig.ident;
    let output = sig.output;

    let mut inputs = sig.inputs.into_iter();
    let receiver = match inputs.next() {
        Some(FnArg::Typed(receiver)) => match *receiver.ty {
            Type::Reference(reference) => *reference.elem,
            other => {
                return Err(syn::Error::new_spanned(
                    &other,
                    "`#[wasm_bindgen]`: the receiver of a `method` must be a reference, \
                     as in `this: &GpuDevice`",
                ));
            }
        },
        Some(FnArg::Receiver(receiver)) => {
            return Err(syn::Error::new_spanned(
                receiver,
                "`#[wasm_bindgen]`: write the receiver of a `method` as an ordinary \
                 first parameter, as in `this: &GpuDevice`",
            ));
        }
        None => {
            return Err(syn::Error::new(
                name.span(),
                "`#[wasm_bindgen]`: a `method` needs a receiver as its first parameter, \
                 as in `this: &GpuDevice`",
            ));
        }
    };

    let arguments = arguments(inputs)?;
    let class = class_name(attrs, &receiver);
    let rt = rt();
    let this = quote!(&<Self as #rt::AsJs>::as_js(self));

    // The JavaScript property this declaration names. `getter = "x"` / `setter =
    // "x"` give it outright; a bare `getter` / `setter` derives it the way
    // wasm-bindgen does, from `js_name` or the Rust name with the accessor prefix
    // removed.
    let (property, body) = if let Some(explicit) = &attrs.getter {
        let property = accessor_name(explicit.as_deref(), attrs, &name, "get_");
        if !arguments.idents.is_empty() {
            return Err(syn::Error::new(
                name.span(),
                "`#[wasm_bindgen]`: a `getter` takes no parameters beyond its receiver",
            ));
        }
        let key = c_literal(&property, name.span())?;
        (property, quote!(#rt::get(#this, #key)))
    } else if let Some(explicit) = &attrs.setter {
        let property = accessor_name(explicit.as_deref(), attrs, &name, "set_");
        let [value] = arguments.idents.as_slice() else {
            return Err(syn::Error::new(
                name.span(),
                "`#[wasm_bindgen]`: a `setter` takes exactly one parameter beyond its \
                 receiver",
            ));
        };
        let key = c_literal(&property, name.span())?;
        (
            property,
            quote!(#rt::set(#this, #key, &#rt::AsJs::as_js(&#value))),
        )
    } else {
        let property = attrs.js_name.clone().unwrap_or_else(|| name.unraw_string());
        let key = c_literal(&property, name.span())?;
        let values = &arguments.values;
        (
            property,
            quote!(#rt::call_method(#this, #key, &[#(#values),*])),
        )
    };

    let operation = if attrs.setter.is_some() {
        format!("setting {class}.{property}")
    } else {
        format!("{class}.{property}")
    };
    // A `setter` already produces `()`, so there is nothing to convert; the other
    // two produce a `JsValue` that the declared return type reinterprets.
    let body = complete(body, &output, attrs, &operation, attrs.setter.is_none())?;
    let parameters = &arguments.parameters;

    Ok(Operation {
        receiver: receiver.to_token_stream(),
        item: quote! {
            #(#docs)*
            #vis fn #name(&self #(, #parameters)*) #output {
                #body
            }
        },
    })
}

/// A `constructor`: an associated function that runs `new <class>(..)`.
fn constructor(item: ForeignItemFn, attrs: &Attrs) -> syn::Result<Operation> {
    let ForeignItemFn {
        attrs: docs,
        vis,
        sig,
        ..
    } = item;
    let name = sig.ident;
    let output = sig.output;

    let ReturnType::Type(_, constructed) = &output else {
        return Err(syn::Error::new(
            name.span(),
            "`#[wasm_bindgen]`: a `constructor` must declare the type it constructs as \
             its return type",
        ));
    };
    // With `catch` the declared type is `Result<T, JsValue>`, and `T` is the class.
    let constructed = if attrs.catch.is_some() {
        result_ok_type(constructed).ok_or_else(|| {
            syn::Error::new_spanned(
                constructed,
                "`#[wasm_bindgen]`: `catch` requires a `Result<T, JsValue>` return type",
            )
        })?
    } else {
        constructed.as_ref()
    };

    let arguments = arguments(sig.inputs.into_iter())?;
    // wasm-bindgen looks the constructor up as a global by `js_class`, then
    // `js_name`, then the Rust type name.
    let class = attrs
        .js_class
        .clone()
        .or_else(|| attrs.js_name.clone())
        .unwrap_or_else(|| type_name(constructed));
    let key = c_literal(&class, name.span())?;

    let rt = rt();
    let values = &arguments.values;
    let body = complete(
        quote!(#rt::construct(#key, &[#(#values),*])),
        &output,
        attrs,
        &format!("new {class}"),
        true,
    )?;
    let parameters = &arguments.parameters;

    Ok(Operation {
        receiver: constructed.to_token_stream(),
        item: quote! {
            #(#docs)*
            #vis fn #name(#(#parameters),*) #output {
                #body
            }
        },
    })
}

/// The parameters of a declaration, split into what the generated signature needs
/// and what the generated call needs.
struct Arguments {
    parameters: Vec<TokenStream>,
    idents: Vec<Ident>,
    values: Vec<TokenStream>,
}

fn arguments(inputs: impl Iterator<Item = FnArg>) -> syn::Result<Arguments> {
    let rt = rt();
    let mut arguments = Arguments {
        parameters: Vec::new(),
        idents: Vec::new(),
        values: Vec::new(),
    };
    for input in inputs {
        let FnArg::Typed(argument) = input else {
            return Err(syn::Error::new_spanned(
                input,
                "`#[wasm_bindgen]`: `self` is only valid as the receiver of a `method`, \
                 written as `this: &Type`",
            ));
        };
        let Pat::Ident(pattern) = &*argument.pat else {
            return Err(syn::Error::new_spanned(
                &argument.pat,
                "`#[wasm_bindgen]`: an imported parameter must be a plain name",
            ));
        };
        let ident = pattern.ident.clone();
        arguments.parameters.push(argument.to_token_stream());
        arguments.values.push(quote!(#rt::AsJs::as_js(&#ident)));
        arguments.idents.push(ident);
    }
    Ok(arguments)
}

/// Wraps the raw operation in the error handling and return conversion the
/// declaration asked for.
///
/// * `catch` hands the JavaScript exception back as `Err`, so the `Result` the
///   operation already returns is the declared return type once its success value
///   is converted.
/// * without `catch` there is nowhere to report the exception to — wasm-bindgen
///   would have let it escape through the import boundary — so `unwrap_js` takes it
///   and ends the module, naming the operation that threw.
fn complete(
    operation: TokenStream,
    output: &ReturnType,
    attrs: &Attrs,
    description: &str,
    convert: bool,
) -> syn::Result<TokenStream> {
    let rt = rt();
    if attrs.catch.is_some() {
        if let ReturnType::Type(_, declared) = output {
            if result_ok_type(declared).is_none() {
                return Err(syn::Error::new_spanned(
                    declared,
                    "`#[wasm_bindgen]`: `catch` requires a `Result<T, JsValue>` return type",
                ));
            }
        } else {
            return Err(syn::Error::new_spanned(
                output,
                "`#[wasm_bindgen]`: `catch` requires a `Result<T, JsValue>` return type",
            ));
        }
        if !convert {
            return Ok(operation);
        }
        return Ok(quote! {
            ::core::result::Result::map(#operation, #rt::FromJs::from_js)
        });
    }

    let unwrapped = quote!(#rt::unwrap_js(#operation, #description));
    if convert {
        Ok(quote!(#rt::FromJs::from_js(#unwrapped)))
    } else {
        Ok(unwrapped)
    }
}

/// The JavaScript property a bare `getter` / `setter` names: `js_name` if given,
/// otherwise the Rust name, in either case with the accessor prefix removed —
/// `get_label` names `label`, as it does under wasm-bindgen.
fn accessor_name(explicit: Option<&str>, attrs: &Attrs, name: &Ident, prefix: &str) -> String {
    if let Some(explicit) = explicit {
        return explicit.to_string();
    }
    let derived = attrs.js_name.clone().unwrap_or_else(|| name.unraw_string());
    match derived.strip_prefix(prefix) {
        Some(stripped) => stripped.to_string(),
        None => derived,
    }
}

/// The JavaScript class name used in diagnostics: `js_class` if the declaration
/// gives one, else the Rust receiver type.
fn class_name(attrs: &Attrs, receiver: &Type) -> String {
    attrs
        .js_class
        .clone()
        .unwrap_or_else(|| type_name(receiver))
}

/// The last path segment of a type, which is the name the JavaScript side uses when
/// the declaration does not say otherwise.
fn type_name(ty: &Type) -> String {
    match ty {
        Type::Path(path) => path.path.segments.last().map_or_else(
            || ty.to_token_stream().to_string(),
            |last| last.ident.unraw_string(),
        ),
        other => other.to_token_stream().to_string(),
    }
}

/// The `T` of a `Result<T, _>`, by the syntax of the declared return type.
fn result_ok_type(ty: &Type) -> Option<&Type> {
    let Type::Path(path) = ty else {
        return None;
    };
    let last = path.path.segments.last()?;
    if last.ident != "Result" {
        return None;
    }
    let PathArguments::AngleBracketed(arguments) = &last.arguments else {
        return None;
    };
    arguments.args.first().and_then(|argument| match argument {
        syn::GenericArgument::Type(ty) => Some(ty),
        _ => None,
    })
}

// -- string enums -----------------------------------------------------------

/// Expands `pub enum X { Variant = "js-string", … }` into a plain Rust enum plus
/// the conversions to and from its JavaScript spelling.
fn string_enum(mut item: ItemEnum) -> TokenStream {
    if let Err(error) = parse::take(&mut item.attrs) {
        return error.to_compile_error();
    }
    if !item.generics.params.is_empty() {
        return syn::Error::new_spanned(
            &item.generics,
            "`#[wasm_bindgen]`: a string enum cannot be generic",
        )
        .to_compile_error();
    }

    let mut names = Vec::with_capacity(item.variants.len());
    let mut strings = Vec::with_capacity(item.variants.len());
    let mut variants = Vec::with_capacity(item.variants.len());

    for mut variant in std::mem::take(&mut item.variants) {
        if let Err(error) = parse::take(&mut variant.attrs) {
            return error.to_compile_error();
        }
        if !matches!(variant.fields, Fields::Unit) {
            return syn::Error::new_spanned(
                &variant.fields,
                "`#[wasm_bindgen]`: a string enum variant cannot have fields",
            )
            .to_compile_error();
        }
        let Some((_, Expr::Lit(discriminant))) = &variant.discriminant else {
            return syn::Error::new_spanned(
                &variant,
                "`#[wasm_bindgen]`: a string enum variant must give its JavaScript \
                 spelling, as in `Repeat = \"repeat\"`",
            )
            .to_compile_error();
        };
        let Lit::Str(text) = &discriminant.lit else {
            return syn::Error::new_spanned(
                &discriminant.lit,
                "`#[wasm_bindgen]`: a string enum variant must be given a string literal",
            )
            .to_compile_error();
        };
        strings.push(text.value());
        names.push(variant.ident.clone());
        let attrs = &variant.attrs;
        let ident = &variant.ident;
        variants.push(quote!(#(#attrs)* #ident));
    }

    let rt = rt();
    let js_value = js_value();
    let name = &item.ident;
    let text_name = name.unraw_string();
    let vis = &item.vis;
    let docs = &item.attrs;
    let extra_derives = supplemental_derives(&item.attrs);
    let invalid_to_string = format!(
        "wgpu-napi-web: `{text_name}::__Invalid` has no JavaScript spelling — it is \
         what a value outside the enumeration was read as"
    );

    // A value outside the enumeration becomes `__Invalid`, exactly as wasm-bindgen's
    // string enums do, rather than being reported: the consumer's `_` arm is what
    // decides what an unrecognised value means. `#[non_exhaustive]` plus the hidden
    // variant is also what keeps that arm reachable, in this crate and outside it —
    // WebGPU adds enumerators (texture formats above all) faster than bindings are
    // re-vendored.
    quote! {
        #(#docs)*
        #extra_derives
        #[non_exhaustive]
        #vis enum #name {
            #(#variants,)*
            #[doc(hidden)]
            __Invalid,
        }

        impl #name {
            /// The JavaScript string this variant is spelled as.
            ///
            /// # Panics
            ///
            /// If called on a value read from JavaScript that was outside the
            /// enumeration.
            #[inline]
            #vis fn as_js_str(&self) -> &'static str {
                match self {
                    #(Self::#names => #strings,)*
                    Self::__Invalid => ::core::panic!(#invalid_to_string),
                }
            }

            /// The variant JavaScript spells `text`, or `None` if there is none.
            #vis fn from_js_str(text: &str) -> ::core::option::Option<Self> {
                match text {
                    #(#strings => ::core::option::Option::Some(Self::#names),)*
                    _ => ::core::option::Option::None,
                }
            }
        }

        impl #rt::AsJs for #name {
            #[inline]
            fn as_js(&self) -> #js_value {
                <#js_value>::from_str(self.as_js_str())
            }
        }

        // `wgpu`'s backend reads an enum's JavaScript spelling back out through
        // `JsValue::from(variant).as_string()`, which wasm-bindgen's string enums
        // also support.
        impl ::core::convert::From<#name> for #js_value {
            #[inline]
            fn from(variant: #name) -> Self {
                Self::from_str(variant.as_js_str())
            }
        }

        impl #rt::FromJs for #name {
            fn from_js(value: #js_value) -> Self {
                match value.as_string().as_deref().and_then(Self::from_js_str) {
                    ::core::option::Option::Some(variant) => variant,
                    ::core::option::Option::None => Self::__Invalid,
                }
            }
        }
    }
}

// -- shared helpers ---------------------------------------------------------

/// A `c"…"` literal for a property, class or method name.
///
/// Node-API's `*_named_property` entry points want a NUL-terminated name, and the
/// name is known here, so no allocation happens at run time.
fn c_literal(text: &str, span: Span) -> syn::Result<Literal> {
    let value = CString::new(text).map_err(|_| {
        syn::Error::new(
            span,
            format!("`#[wasm_bindgen]`: the JavaScript name {text:?} contains a NUL byte"),
        )
    })?;
    let mut literal = Literal::c_string(&value);
    literal.set_span(span);
    Ok(literal)
}

/// A `#[derive(…)]` for whichever of `Debug` and `Clone` the declaration did not
/// ask for itself.
///
/// Every JavaScript value type here is a handle that can be duplicated, and the
/// workspace warns on public types without a `Debug`, so both are unconditional.
/// The declaration's own derives are left where they are, so `PartialEq` / `Eq`
/// still appear only where the bindings ask for them.
fn supplemental_derives(attrs: &[Attribute]) -> TokenStream {
    let mut has_debug = false;
    let mut has_clone = false;
    for attr in attrs {
        if !attr.path().is_ident("derive") {
            continue;
        }
        let Ok(paths) =
            attr.parse_args_with(Punctuated::<syn::Path, syn::Token![,]>::parse_terminated)
        else {
            continue;
        };
        for path in paths {
            match path.segments.last().map(|last| last.ident.to_string()) {
                Some(name) if name == "Debug" => has_debug = true,
                Some(name) if name == "Clone" => has_clone = true,
                _ => {}
            }
        }
    }
    match (has_debug, has_clone) {
        (true, true) => TokenStream::new(),
        (true, false) => quote!(#[derive(::core::clone::Clone)]),
        (false, true) => quote!(#[derive(::core::fmt::Debug)]),
        (false, false) => quote!(#[derive(::core::fmt::Debug, ::core::clone::Clone)]),
    }
}

/// The name of an identifier with any `r#` removed, which is what JavaScript sees.
trait UnrawString {
    fn unraw_string(&self) -> String;
}

impl UnrawString for Ident {
    fn unraw_string(&self) -> String {
        use syn::ext::IdentExt as _;
        self.unraw().to_string()
    }
}
