//! Stand-in for the `web-sys` crate, covering the DOM types `wgpu`'s WebGPU
//! backend names.
//!
//! # What is here, and why only this
//!
//! WebGPU itself lives in [`crate::js_sys`]-shaped generated bindings under
//! `wgpu::backend::webgpu::webgpu_sys`; the DOM only appears at the edges of that
//! backend, in four places:
//!
//! * surface creation — [`HtmlCanvasElement`] / [`OffscreenCanvas`] and their
//!   `getContext("webgpu")`, plus [`window`] → [`Document::query_selector_all`] →
//!   [`NodeList::get`] for the `raw-window-handle` `Web` variant;
//! * adapter discovery — [`Window::navigator`] / [`WorkerGlobalScope::navigator`],
//!   whose `gpu` property `wgpu` reads through its own extension binding;
//! * surface capabilities — [`Window::match_media`] and [`MediaQueryList::matches`]
//!   for the CSS `color-gamut` and `dynamic-range` queries;
//! * external image copies — [`ImageBitmap`], [`ImageData`], [`HtmlImageElement`],
//!   [`HtmlVideoElement`], [`VideoFrame`] and the two canvas types, of which only
//!   the pixel dimensions are ever read.
//!
//! [`Event`] and [`EventTarget`] carry no methods: they exist because the
//! generated WebGPU bindings declare `GPUDevice` as extending `EventTarget` and
//! `GPUUncapturedErrorEvent` as extending `Event`, and those `extends` clauses
//! need the types to name.
//!
//! # The class hierarchy is the real one
//!
//! [`Node`], [`Element`], [`HtmlElement`] and [`HtmlMediaElement`] are declared
//! with no methods of their own, purely so that the `Deref` and `AsRef` chains
//! match `web-sys` exactly — `HtmlCanvasElement` → `HtmlElement` → `Element` →
//! `Node` → `EventTarget` → `js_sys::Object` → `JsValue`. `wgpu` depends on those
//! chains in two ways that would otherwise break: `&canvas` coerces to
//! `&JsValue` when a canvas is handed to `raw-window-handle`, and
//! `wgpu_types::ExternalImageSource::deref` returns `&js_sys::Object` for all
//! seven source variants.

use core::ffi::CStr;
use core::fmt;

use crate::convert::{AsJs, FromJs};
use crate::js_sys;
use crate::rt;
use crate::value::{JsCast, JsValue, Promising};

/// Declares one JavaScript class as a `#[repr(transparent)]` newtype over its
/// parent, with the trait set every type in this crate carries.
///
/// `class` is the JavaScript class name, which is what `instanceof` needs and
/// which differs from the Rust name for the HTML elements (`HTMLCanvasElement`
/// vs `HtmlCanvasElement`). `ancestors` lists every `extends` from `web-sys`, in
/// order, so the generated `AsRef`/`From` set is the same one the real bindings
/// provide.
macro_rules! js_class {
    (
        $(#[$attr:meta])*
        $name:ident : $parent:ty,
        class = $class:expr,
        ancestors = [$($ancestor:ty),* $(,)?]
        $(,)?
    ) => {
        $(#[$attr])*
        #[repr(transparent)]
        pub struct $name($parent);

        impl core::ops::Deref for $name {
            type Target = $parent;

            #[inline]
            fn deref(&self) -> &$parent {
                &self.0
            }
        }

        impl AsRef<JsValue> for $name {
            #[inline]
            fn as_ref(&self) -> &JsValue {
                self.0.as_ref()
            }
        }

        impl From<$name> for JsValue {
            #[inline]
            fn from(value: $name) -> Self {
                JsValue::from(value.0)
            }
        }

        $(
            impl AsRef<$ancestor> for $name {
                #[inline]
                fn as_ref(&self) -> &$ancestor {
                    <$ancestor as JsCast>::unchecked_from_js_ref(js_value(self))
                }
            }

            impl From<$name> for $ancestor {
                #[inline]
                fn from(value: $name) -> Self {
                    <$ancestor as JsCast>::unchecked_from_js(JsValue::from(value))
                }
            }
        )*

        /// Reinterprets a value as this type, which is what every generated
        /// binding does after a property read or a call.
        impl From<JsValue> for $name {
            #[inline]
            fn from(value: JsValue) -> Self {
                Self(<$parent>::from(value))
            }
        }

        impl JsCast for $name {
            #[inline]
            fn instanceof(value: &JsValue) -> bool {
                rt::instance_of(value, $class)
            }

            #[inline]
            fn unchecked_from_js(value: JsValue) -> Self {
                Self::from(value)
            }

            #[inline]
            fn unchecked_from_js_ref(value: &JsValue) -> &Self {
                // SAFETY: `$name` is `#[repr(transparent)]` over `$parent`, which
                // is itself transparent over its own parent and so on down to
                // `JsValue`, so `$name` and `JsValue` have identical layout and a
                // `&JsValue` is a valid `&$name`.
                unsafe { &*core::ptr::from_ref(value).cast::<$name>() }
            }
        }

        impl AsJs for $name {
            #[inline]
            fn as_js(&self) -> JsValue {
                js_value(self).clone()
            }
        }

        impl FromJs for $name {
            #[inline]
            fn from_js(value: JsValue) -> Self {
                Self::from(value)
            }
        }

        impl Promising for $name {
            type Resolution = Self;
        }

        impl Clone for $name {
            #[inline]
            fn clone(&self) -> Self {
                Self::from(js_value(self).clone())
            }
        }

        impl PartialEq for $name {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                js_value(self) == js_value(other)
            }
        }

        impl Eq for $name {}

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($name))
                    .field(js_value(self))
                    .finish()
            }
        }
    };
}

/// The untyped value behind a bindings type.
///
/// Each type has an `AsRef` impl per ancestor, so a bare `self.as_ref()` inside
/// this module is ambiguous; this pins the one that matters.
#[inline]
fn js_value<T: AsRef<JsValue>>(value: &T) -> &JsValue {
    value.as_ref()
}

/// Reads a `u32`-valued property.
///
/// `operation` names the property for the panic message, since an exception from
/// a plain property read is a broken object rather than something a `wgpu` call
/// site can act on — the same outcome wasm-bindgen produces by letting the throw
/// escape through the import boundary. `#[track_caller]` puts the accessor that
/// failed in the panic location, rather than this shared helper.
#[track_caller]
fn u32_property(target: &JsValue, name: &CStr, operation: &str) -> u32 {
    u32::from_js(rt::unwrap_js(rt::get(target, name), operation))
}

/// Writes a `u32`-valued property.
#[track_caller]
fn set_u32_property(target: &JsValue, name: &CStr, value: u32, operation: &str) {
    rt::unwrap_js(rt::set(target, name, &JsValue::from(value)), operation);
}

/// `canvas.getContext(context_id)`, shared by the two canvas types because the
/// WebIDL for both declares the identical operation.
fn canvas_context(
    canvas: &JsValue,
    context_id: &str,
    options: Option<&JsValue>,
) -> Result<Option<js_sys::Object>, JsValue> {
    let id = JsValue::from_str(context_id);
    let value = match options {
        Some(options) => rt::call_method(canvas, c"getContext", &[id, options.clone()])?,
        None => rt::call_method(canvas, c"getContext", &[id])?,
    };
    Ok(FromJs::from_js(value))
}

js_class! {
    /// The `EventTarget` class.
    ///
    /// Present only as the ancestor that the generated `GPUDevice` binding
    /// extends; `wgpu` registers its device callbacks through the `onuncapturederror`
    /// and `lost` properties rather than through `addEventListener`.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/EventTarget)
    EventTarget: js_sys::Object,
    class = c"EventTarget",
    ancestors = [js_sys::Object],
}

js_class! {
    /// The `Event` class.
    ///
    /// Present only as the ancestor that the generated `GPUUncapturedErrorEvent`
    /// binding extends; `wgpu` reads that event's own `error` property and no
    /// inherited one.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Event)
    Event: js_sys::Object,
    class = c"Event",
    ancestors = [js_sys::Object],
}

js_class! {
    /// The `Node` class.
    ///
    /// `wgpu` names it as what [`NodeList::get`] returns while looking up the
    /// canvas for a `raw-window-handle` `Web` handle, and immediately casts it to
    /// [`HtmlCanvasElement`].
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Node)
    Node: EventTarget,
    class = c"Node",
    ancestors = [EventTarget, js_sys::Object],
}

js_class! {
    /// The `Element` class, a link in the [`HtmlCanvasElement`] ancestry.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Element)
    Element: Node,
    class = c"Element",
    ancestors = [Node, EventTarget, js_sys::Object],
}

js_class! {
    /// The `HTMLElement` class, a link in the [`HtmlCanvasElement`] ancestry.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement)
    HtmlElement: Element,
    class = c"HTMLElement",
    ancestors = [Element, Node, EventTarget, js_sys::Object],
}

js_class! {
    /// The `HTMLMediaElement` class, a link in the [`HtmlVideoElement`] ancestry.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement)
    HtmlMediaElement: HtmlElement,
    class = c"HTMLMediaElement",
    ancestors = [HtmlElement, Element, Node, EventTarget, js_sys::Object],
}

js_class! {
    /// The `Window` class.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Window)
    Window: EventTarget,
    class = c"Window",
    ancestors = [EventTarget, js_sys::Object],
}

js_class! {
    /// The `WorkerGlobalScope` class, the global object inside a worker.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/WorkerGlobalScope)
    WorkerGlobalScope: EventTarget,
    class = c"WorkerGlobalScope",
    ancestors = [EventTarget, js_sys::Object],
}

js_class! {
    /// The `Navigator` class.
    ///
    /// The `gpu` property is not declared here: `wgpu` reads it through its own
    /// extension binding, so that this type keeps working if `web-sys` ever adds
    /// a `gpu` method of its own.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Navigator)
    Navigator: js_sys::Object,
    class = c"Navigator",
    ancestors = [js_sys::Object],
}

js_class! {
    /// The `WorkerNavigator` class, the worker counterpart of [`Navigator`].
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/WorkerNavigator)
    WorkerNavigator: js_sys::Object,
    class = c"WorkerNavigator",
    ancestors = [js_sys::Object],
}

js_class! {
    /// The `Document` class.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Document)
    Document: Node,
    class = c"Document",
    ancestors = [Node, EventTarget, js_sys::Object],
}

js_class! {
    /// The `NodeList` class.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/NodeList)
    NodeList: js_sys::Object,
    class = c"NodeList",
    ancestors = [js_sys::Object],
}

js_class! {
    /// The `MediaQueryList` class, the result of [`Window::match_media`].
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MediaQueryList)
    MediaQueryList: EventTarget,
    class = c"MediaQueryList",
    ancestors = [EventTarget, js_sys::Object],
}

js_class! {
    /// The `HTMLCanvasElement` class.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement)
    HtmlCanvasElement: HtmlElement,
    class = c"HTMLCanvasElement",
    ancestors = [HtmlElement, Element, Node, EventTarget, js_sys::Object],
}

js_class! {
    /// The `OffscreenCanvas` class.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas)
    OffscreenCanvas: EventTarget,
    class = c"OffscreenCanvas",
    ancestors = [EventTarget, js_sys::Object],
}

js_class! {
    /// The `HTMLImageElement` class.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/HTMLImageElement)
    HtmlImageElement: HtmlElement,
    class = c"HTMLImageElement",
    ancestors = [HtmlElement, Element, Node, EventTarget, js_sys::Object],
}

js_class! {
    /// The `HTMLVideoElement` class.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement)
    HtmlVideoElement: HtmlMediaElement,
    class = c"HTMLVideoElement",
    ancestors = [HtmlMediaElement, HtmlElement, Element, Node, EventTarget, js_sys::Object],
}

js_class! {
    /// The `ImageBitmap` class.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/ImageBitmap)
    ImageBitmap: js_sys::Object,
    class = c"ImageBitmap",
    ancestors = [js_sys::Object],
}

js_class! {
    /// The `ImageData` class.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/ImageData)
    ImageData: js_sys::Object,
    class = c"ImageData",
    ancestors = [js_sys::Object],
}

js_class! {
    /// The `VideoFrame` class.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/VideoFrame)
    VideoFrame: js_sys::Object,
    class = c"VideoFrame",
    ancestors = [js_sys::Object],
}

impl Window {
    /// Getter for the `navigator` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Window/navigator)
    pub fn navigator(&self) -> Navigator {
        let value = rt::unwrap_js(rt::get(js_value(self), c"navigator"), "Window.navigator");
        Navigator::from(value)
    }

    /// Getter for the `document` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Window/document)
    pub fn document(&self) -> Option<Document> {
        let value = rt::unwrap_js(rt::get(js_value(self), c"document"), "Window.document");
        FromJs::from_js(value)
    }

    /// The `matchMedia()` method.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Window/matchMedia)
    pub fn match_media(&self, query: &str) -> Result<Option<MediaQueryList>, JsValue> {
        let value = rt::call_method(js_value(self), c"matchMedia", &[JsValue::from_str(query)])?;
        Ok(FromJs::from_js(value))
    }
}

impl WorkerGlobalScope {
    /// Getter for the `navigator` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/WorkerGlobalScope/navigator)
    pub fn navigator(&self) -> WorkerNavigator {
        let value = rt::unwrap_js(
            rt::get(js_value(self), c"navigator"),
            "WorkerGlobalScope.navigator",
        );
        WorkerNavigator::from(value)
    }
}

impl Document {
    /// The `querySelectorAll()` method.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Document/querySelectorAll)
    pub fn query_selector_all(&self, selectors: &str) -> Result<NodeList, JsValue> {
        let value = rt::call_method(
            js_value(self),
            c"querySelectorAll",
            &[JsValue::from_str(selectors)],
        )?;
        Ok(NodeList::from(value))
    }
}

impl NodeList {
    /// Indexing getter. As in the literal JavaScript `this[index]`.
    pub fn get(&self, index: u32) -> Option<Node> {
        let value = rt::unwrap_js(rt::get_index(js_value(self), index), "NodeList indexing");
        FromJs::from_js(value)
    }
}

impl MediaQueryList {
    /// Getter for the `matches` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MediaQueryList/matches)
    pub fn matches(&self) -> bool {
        let value = rt::unwrap_js(
            rt::get(js_value(self), c"matches"),
            "MediaQueryList.matches",
        );
        bool::from_js(value)
    }
}

impl HtmlCanvasElement {
    /// Getter for the `width` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/width)
    pub fn width(&self) -> u32 {
        u32_property(js_value(self), c"width", "HTMLCanvasElement.width")
    }

    /// Setter for the `width` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/width)
    pub fn set_width(&self, value: u32) {
        set_u32_property(js_value(self), c"width", value, "HTMLCanvasElement.width");
    }

    /// Getter for the `height` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/height)
    pub fn height(&self) -> u32 {
        u32_property(js_value(self), c"height", "HTMLCanvasElement.height")
    }

    /// Setter for the `height` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/height)
    pub fn set_height(&self, value: u32) {
        set_u32_property(js_value(self), c"height", value, "HTMLCanvasElement.height");
    }

    /// The `getContext()` method.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/getContext)
    pub fn get_context(&self, context_id: &str) -> Result<Option<js_sys::Object>, JsValue> {
        canvas_context(js_value(self), context_id, None)
    }

    /// The `getContext()` method, passing the context options argument.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/getContext)
    pub fn get_context_with_context_options(
        &self,
        context_id: &str,
        context_options: &JsValue,
    ) -> Result<Option<js_sys::Object>, JsValue> {
        canvas_context(js_value(self), context_id, Some(context_options))
    }
}

impl OffscreenCanvas {
    /// The `new OffscreenCanvas(..)` constructor, creating a new instance of
    /// `OffscreenCanvas`.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas/OffscreenCanvas)
    pub fn new(width: u32, height: u32) -> Result<OffscreenCanvas, JsValue> {
        let value = rt::construct(
            c"OffscreenCanvas",
            &[JsValue::from(width), JsValue::from(height)],
        )?;
        Ok(OffscreenCanvas::from(value))
    }

    /// Getter for the `width` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas/width)
    pub fn width(&self) -> u32 {
        u32_property(js_value(self), c"width", "OffscreenCanvas.width")
    }

    /// Setter for the `width` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas/width)
    pub fn set_width(&self, value: u32) {
        set_u32_property(js_value(self), c"width", value, "OffscreenCanvas.width");
    }

    /// Getter for the `height` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas/height)
    pub fn height(&self) -> u32 {
        u32_property(js_value(self), c"height", "OffscreenCanvas.height")
    }

    /// Setter for the `height` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas/height)
    pub fn set_height(&self, value: u32) {
        set_u32_property(js_value(self), c"height", value, "OffscreenCanvas.height");
    }

    /// The `getContext()` method.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas/getContext)
    pub fn get_context(&self, context_id: &str) -> Result<Option<js_sys::Object>, JsValue> {
        canvas_context(js_value(self), context_id, None)
    }

    /// The `getContext()` method, passing the context options argument.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas/getContext)
    pub fn get_context_with_context_options(
        &self,
        context_id: &str,
        context_options: &JsValue,
    ) -> Result<Option<js_sys::Object>, JsValue> {
        canvas_context(js_value(self), context_id, Some(context_options))
    }
}

impl ImageBitmap {
    /// Getter for the `width` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/ImageBitmap/width)
    pub fn width(&self) -> u32 {
        u32_property(js_value(self), c"width", "ImageBitmap.width")
    }

    /// Getter for the `height` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/ImageBitmap/height)
    pub fn height(&self) -> u32 {
        u32_property(js_value(self), c"height", "ImageBitmap.height")
    }
}

impl ImageData {
    /// Getter for the `width` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/ImageData/width)
    pub fn width(&self) -> u32 {
        u32_property(js_value(self), c"width", "ImageData.width")
    }

    /// Getter for the `height` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/ImageData/height)
    pub fn height(&self) -> u32 {
        u32_property(js_value(self), c"height", "ImageData.height")
    }
}

impl HtmlImageElement {
    /// Getter for the `width` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/HTMLImageElement/width)
    pub fn width(&self) -> u32 {
        u32_property(js_value(self), c"width", "HTMLImageElement.width")
    }

    /// Getter for the `height` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/HTMLImageElement/height)
    pub fn height(&self) -> u32 {
        u32_property(js_value(self), c"height", "HTMLImageElement.height")
    }
}

impl HtmlVideoElement {
    /// Getter for the `videoWidth` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/videoWidth)
    pub fn video_width(&self) -> u32 {
        u32_property(js_value(self), c"videoWidth", "HTMLVideoElement.videoWidth")
    }

    /// Getter for the `videoHeight` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/videoHeight)
    pub fn video_height(&self) -> u32 {
        u32_property(
            js_value(self),
            c"videoHeight",
            "HTMLVideoElement.videoHeight",
        )
    }
}

impl VideoFrame {
    /// Getter for the `displayWidth` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/VideoFrame/displayWidth)
    pub fn display_width(&self) -> u32 {
        u32_property(js_value(self), c"displayWidth", "VideoFrame.displayWidth")
    }

    /// Getter for the `displayHeight` field of this object.
    ///
    /// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/VideoFrame/displayHeight)
    pub fn display_height(&self) -> u32 {
        u32_property(js_value(self), c"displayHeight", "VideoFrame.displayHeight")
    }
}

/// Getter for the `Window` object, or `None` when there is no document — a
/// worker, or a Node process.
///
/// `web-sys` implements this as `global().dyn_into::<Window>()`, an `instanceof`
/// against the `Window` constructor. That constructor is a browser main-thread
/// artefact: in a worker it is absent, and under `@napi-rs/wasm-runtime` in Node
/// it is absent as well, so the check would report "not a window" for reasons
/// unrelated to whether a document exists. Reading `globalThis.window` instead
/// asks the question directly — the property is self-referential on a `Window`
/// and undefined on a `WorkerGlobalScope` — which is what the callers in `wgpu`
/// (`matchMedia` for `color-gamut`, `document.querySelectorAll` for a canvas)
/// actually need to know.
///
/// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Window)
pub fn window() -> Option<Window> {
    let value = rt::global(c"window").ok()?;
    if value.is_undefined() || value.is_null() {
        return None;
    }
    Some(Window::from(value))
}
