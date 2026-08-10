//! Parsing the `#[wasm_bindgen(...)]` keys used by `wgpu`'s vendored bindings.
//!
//! Keys are order-insensitive, which is what the vendored files rely on: the
//! `web-sys` generator emits `method , structural , js_class , js_name` for class
//! methods but `method, getter = "…"` for dictionary accessors, and the two
//! hand-written blocks in the backend spell yet another order again.

use proc_macro2::Span;
use syn::ext::IdentExt as _;
use syn::parse::{Parse, ParseStream};
use syn::{Attribute, Ident, LitStr, Meta, Path, Token};

/// Every key one `#[wasm_bindgen(...)]` attribute can carry here.
///
/// Flags record their span so a rejected combination can point at the key that
/// caused it rather than at the whole declaration.
#[derive(Default)]
pub(crate) struct Attrs {
    /// `method`: the first parameter is the receiver.
    pub(crate) method: Option<Span>,
    /// `catch`: the declared return type is `Result<T, JsValue>`.
    pub(crate) catch: Option<Span>,
    /// `constructor`: `new <class>(..)`.
    pub(crate) constructor: Option<Span>,
    /// `getter` or `getter = "name"`.
    pub(crate) getter: Option<Option<String>>,
    /// `setter` or `setter = "name"`.
    pub(crate) setter: Option<Option<String>>,
    /// `js_name = ident` or `js_name = "name"`.
    pub(crate) js_name: Option<String>,
    /// `js_class = "name"`.
    pub(crate) js_class: Option<String>,
    /// `extends = Path`, repeated; the first is the `Deref` parent.
    pub(crate) extends: Vec<Path>,
}

impl Attrs {
    /// The span of whichever key is most specific to the declaration, for errors
    /// about combinations of keys.
    pub(crate) fn span(&self, fallback: Span) -> Span {
        self.constructor
            .or(self.method)
            .or(self.catch)
            .unwrap_or(fallback)
    }

    /// Folds a second attribute on the same item into this one. First key wins,
    /// except `extends`, which accumulates in declaration order.
    fn merge(&mut self, other: Self) {
        self.method = self.method.or(other.method);
        self.catch = self.catch.or(other.catch);
        self.constructor = self.constructor.or(other.constructor);
        self.getter = self.getter.take().or(other.getter);
        self.setter = self.setter.take().or(other.setter);
        self.js_name = self.js_name.take().or(other.js_name);
        self.js_class = self.js_class.take().or(other.js_class);
        self.extends.extend(other.extends);
    }

    fn parse_entry(&mut self, input: ParseStream<'_>) -> syn::Result<()> {
        let key = input.call(Ident::parse_any)?;
        match key.unraw().to_string().as_str() {
            "method" => {
                no_value(input, &key)?;
                self.method = Some(key.span());
            }
            "catch" => {
                no_value(input, &key)?;
                self.catch = Some(key.span());
            }
            "constructor" => {
                no_value(input, &key)?;
                self.constructor = Some(key.span());
            }
            // Non-structural imports differ from structural ones only in that
            // wasm-bindgen caches the function off the prototype at instantiation.
            // Every operation here is looked up by name on the receiver at call
            // time, which is what `structural` asks for, so the key carries no
            // additional meaning.
            "structural" => no_value(input, &key)?,
            // The TypeScript type name only ever reached wasm-bindgen's `.d.ts`
            // emitter, and nothing here emits TypeScript.
            "typescript_type" => {
                value(input)?;
            }
            "getter" => self.getter = Some(optional_value(input)?),
            "setter" => self.setter = Some(optional_value(input)?),
            "js_name" => self.js_name = Some(value(input)?),
            "js_class" => self.js_class = Some(value(input)?),
            "extends" => {
                input.parse::<Token![=]>()?;
                self.extends.push(input.parse::<Path>()?);
            }
            other => {
                return Err(syn::Error::new(
                    key.span(),
                    format!(
                        "`#[wasm_bindgen]`: `{other}` is not supported by wgpu's Node-API \
                         stand-in for wasm-bindgen"
                    ),
                ));
            }
        }
        Ok(())
    }
}

impl Parse for Attrs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut attrs = Self::default();
        while !input.is_empty() {
            attrs.parse_entry(input)?;
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }
        Ok(attrs)
    }
}

/// Removes every `#[wasm_bindgen…]` attribute from `attrs` and returns their
/// merged keys. What is left in `attrs` — `#[doc]`, `#[derive]`, anything else —
/// is carried through to the generated item untouched.
pub(crate) fn take(attrs: &mut Vec<Attribute>) -> syn::Result<Attrs> {
    let mut merged = Attrs::default();
    let mut error: Option<syn::Error> = None;
    let mut kept = Vec::with_capacity(attrs.len());

    for attr in attrs.drain(..) {
        if !attr.path().is_ident("wasm_bindgen") {
            kept.push(attr);
            continue;
        }
        match &attr.meta {
            // A bare `#[wasm_bindgen]` on a type, as the hand-written
            // `NavigatorWithGpu` block writes it.
            Meta::Path(_) => {}
            Meta::List(list) => match list.parse_args::<Attrs>() {
                Ok(parsed) => merged.merge(parsed),
                Err(parse_error) => push(&mut error, parse_error),
            },
            Meta::NameValue(name_value) => push(
                &mut error,
                syn::Error::new_spanned(
                    name_value,
                    "`#[wasm_bindgen]` takes a parenthesised key list, not `= value`",
                ),
            ),
        }
    }

    *attrs = kept;
    match error {
        Some(error) => Err(error),
        None => Ok(merged),
    }
}

fn push(slot: &mut Option<syn::Error>, error: syn::Error) {
    match slot {
        Some(existing) => existing.combine(error),
        None => *slot = Some(error),
    }
}

/// `= "text"` or `= ident`, returning the text either way.
///
/// The bare-token form is not decoration: `js_name = type` appears in the
/// bindings, and `type` cannot be parsed as an ordinary identifier.
fn value(input: ParseStream<'_>) -> syn::Result<String> {
    input.parse::<Token![=]>()?;
    if input.peek(LitStr) {
        return Ok(input.parse::<LitStr>()?.value());
    }
    Ok(input.call(Ident::parse_any)?.unraw().to_string())
}

fn optional_value(input: ParseStream<'_>) -> syn::Result<Option<String>> {
    if input.peek(Token![=]) {
        return Ok(Some(value(input)?));
    }
    Ok(None)
}

fn no_value(input: ParseStream<'_>, key: &Ident) -> syn::Result<()> {
    if input.peek(Token![=]) {
        return Err(syn::Error::new(
            key.span(),
            format!("`#[wasm_bindgen]`: `{key}` does not take a value"),
        ));
    }
    Ok(())
}
