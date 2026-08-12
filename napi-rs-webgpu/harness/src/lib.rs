//! A Node-API addon that drives `wgpu` over `napi-rs-webgpu`.
//!
//! This is the acceptance test for the crate: everything else can be checked by
//! compiling, but "the bindings reach a real WebGPU implementation" can only be
//! checked by reaching one. Built for `wasm32-wasip1-threads`, loaded by
//! `@napi-rs/wasm-runtime`, and handed the `GPU` that the `webgpu` npm package's
//! `create([])` returns, it uploads a colour with `queue.write_buffer`, clears a
//! 64×64 texture to another colour, draws a triangle over half of it in the
//! uploaded colour, and reads the pixels back.
//!
//! Half, not all: a render that covered the whole target would pass with the
//! draw silently doing nothing, since a clear alone would produce a uniform
//! image too. Two colours in known places test the clear, the draw and the
//! orientation separately.
//!
//! Three exports, because JavaScript owns the event loop:
//!
//! * [`install_webgpu`] gives the crate this thread's environment and the `GPU`
//!   object;
//! * [`start`] queues the render on the JavaScript microtask queue and returns;
//! * [`take_result`] hands back the outcome once it is there, and `null` until
//!   then, so the caller polls it from its own loop and Node stays in charge of
//!   when work runs.
//!
//! A promise would read better, but the point here is to exercise
//! `napi-rs-webgpu`, not napi-rs' promise machinery, so the plumbing is kept as
//! plain as it can be.

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

// Imported one by one rather than through `napi::bindgen_prelude::*`, which
// brings a `Result<T, S = Status>` alias that quietly shadows `core`'s and turns
// every `Result<T, String>` in this file into `Result<T, napi::Error<String>>`.
use napi::bindgen_prelude::Uint8Array as NapiUint8Array;
use napi::{Env, Error, JsValue as _, Unknown};
use napi_derive::napi;
use napi_rs_webgpu::Uint8Array as WebUint8Array;

/// What this addon's own fallible steps return, kept clear of napi-rs' `Result`.
type Outcome<T> = core::result::Result<T, String>;

/// The render's size. 64 pixels of `Rgba8Unorm` is exactly 256 bytes, which is
/// [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`], so the readback needs no row padding.
const SIZE: u32 = 64;

/// What the render pass clears to, and therefore what the half the triangle
/// misses must be.
///
/// Every channel is an exact `k / 255`, so the `Rgba8Unorm` conversion has
/// nothing to round and the expected bytes are not a matter of opinion.
const CLEAR: [u8; 4] = [0x11, 0x22, 0x33, 0xff];

/// What the fragment shader writes, and therefore what the half the triangle
/// covers must be.
const DRAW: [u8; 4] = [0x33, 0x99, 0x66, 0xff];

const SHADER: &str = r#"
@group(0) @binding(0) var<uniform> colour: vec4<u32>;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    // Half the viewport: the triangle below the diagonal x + y = 0, which in
    // clip space is the bottom-left half and in the image the lower-left half.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
    );
    return vec4<f32>(positions[index], 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(colour) / 255.0;
}
"#;

/// The rendered pixels, and what drew them.
type Rendered = (String, Vec<u8>);

thread_local! {
    /// What the adapter said about itself, kept for [`adapter`].
    static ADAPTER: RefCell<Option<String>> = const { RefCell::new(None) };

    /// Where [`start`] leaves its answer for [`take_result`].
    static OUTCOME: RefCell<Option<Outcome<Rendered>>> = const { RefCell::new(None) };
}

/// Gives `napi-rs-webgpu` this thread's environment and the `GPU` object.
///
/// Both happen here rather than at module load because napi-rs' `#[module_init]`
/// is a static constructor: it runs before Node-API exists and is handed no
/// environment. The first `#[napi]` call is the earliest point one is available,
/// and this is it.
///
/// node-webgpu hands the `GPU` back from `create([])` and puts nothing on the
/// global, which is exactly the case `napi_rs_webgpu::install_gpu` exists for.
#[napi]
pub fn install_webgpu(env: Env, gpu: Unknown) {
    // SAFETY: `env` is this thread's environment, live for as long as the module
    // is loaded, and `gpu` is live in the handle scope Node opened for this call.
    unsafe {
        napi_rs_webgpu::install(env.raw());
        napi_rs_webgpu::install_gpu(napi_rs_webgpu::adopt_js_value(env.raw(), gpu.raw()));
    }
}

/// Queues the render and returns immediately.
///
/// Nothing has run when this returns: `spawn_local` schedules the first poll on a
/// microtask, so every `await` inside resolves from Node's event loop.
#[napi]
pub fn start() {
    OUTCOME.with(|slot| *slot.borrow_mut() = None);
    napi_rs_webgpu::futures::spawn_local(async {
        let outcome = render().await;
        if let Ok((adapter, _)) = &outcome {
            ADAPTER.with(|slot| *slot.borrow_mut() = Some(adapter.clone()));
        }
        OUTCOME.with(|slot| *slot.borrow_mut() = Some(outcome));
    });
}

/// The rendered pixels, or `null` while the render is still in flight.
///
/// # Errors
///
/// If the render failed, with the step that failed and why.
#[napi]
pub fn take_result() -> napi::Result<Option<NapiUint8Array>> {
    match OUTCOME.with(|slot| slot.borrow_mut().take()) {
        None => Ok(None),
        Some(Ok((_, pixels))) => Ok(Some(pixels.into())),
        Some(Err(message)) => Err(Error::from_reason(message)),
    }
}

/// Whether an Emnapi typed-array view aliases Rust's shared Wasm memory in both
/// directions without an intermediate JavaScript backing store.
///
/// This is deliberately independent of WebGPU, so CI can cover the memory-view
/// bridge without requiring an adapter.
#[napi]
pub fn memory_view_aliases(env: Env) -> bool {
    const FROM_RUST: [u8; 8] = [0x13, 0x37, 0x42, 0x7f, 0x80, 0xa5, 0xcc, 0xfe];
    const FROM_JS: [u8; 8] = [0xfe, 0xcc, 0xa5, 0x80, 0x7f, 0x42, 0x37, 0x13];

    // SAFETY: this call owns a live `napi_env` for the current thread.
    unsafe { napi_rs_webgpu::install(env.raw()) };

    let source_bytes = FROM_JS;
    // SAFETY: `source_bytes` remains live and immutable through the synchronous
    // `TypedArray.set` call below.
    let source = unsafe { WebUint8Array::view(&source_bytes) };
    let mut backing = [0; FROM_RUST.len()];
    // SAFETY: `backing` stays live until every access through `view` is complete.
    // Rust and JavaScript touch it only in separate, synchronous steps, and this
    // threaded harness imports fixed shared memory, whose existing range remains
    // valid if the memory grows.
    let view = unsafe { WebUint8Array::view(&backing) };

    // A constructor-time copy would remain all-zero and fail this check.
    backing.copy_from_slice(&FROM_RUST);
    if view.to_vec() != FROM_RUST {
        return false;
    }

    // `TypedArray.set` writes through the JS view. Seeing the bytes in the Rust
    // array proves the backing store is the same shared Wasm memory in reverse.
    view.set(&source, 0);
    backing == FROM_JS
}

/// What the adapter reported about itself, once [`take_result`] has something.
///
/// Printed by the caller so the run says which device did the work rather than
/// only that some device did.
#[napi]
pub fn adapter() -> Option<String> {
    ADAPTER.with(|slot| slot.borrow().clone())
}

/// What the caller should expect back: the size, then the clear colour, then the
/// triangle colour. Flat, so the JavaScript side repeats none of it.
#[napi]
pub fn expected() -> Vec<u32> {
    let mut out = vec![SIZE, SIZE];
    out.extend(CLEAR.iter().map(|&byte| u32::from(byte)));
    out.extend(DRAW.iter().map(|&byte| u32::from(byte)));
    out
}

/// Renders the triangle and reads the texture back.
async fn render() -> Outcome<Rendered> {
    // `InstanceDescriptor` has no `Default`; the display handle is the field that
    // cannot have one, and a WebGPU build has no use for it.
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(descriptor);

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .map_err(|error| format!("request_adapter: {error}"))?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("harness"),
            required_features: wgpu::Features::empty(),
            // The WebGPU baseline, which every WebGPU adapter supports by
            // definition — so a failure here is the binding, not the device.
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|error| format!("request_device: {error}"))?;

    let info = adapter.get_info();
    let described = format!("{:?} / {} / {}", info.backend, info.name, info.driver);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("triangle"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("triangle"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        // The fragment state is what makes this test worth running: its `targets`
        // is a `sequence<GPUColorTargetState?>`, one of the four nullable
        // sequences in WebGPU's IDL.
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let colour = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("triangle colour"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let colour_bytes = DRAW
        .into_iter()
        .flat_map(|channel| u32::from(channel).to_le_bytes())
        .collect::<Vec<_>>();
    queue.write_buffer(&colour, 0, &colour_bytes);
    let colour_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("triangle colour"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: colour.as_entire_binding(),
        }],
    });

    let row_bytes = SIZE * 4;
    assert_eq!(row_bytes % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: u64::from(row_bytes * SIZE),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("harness"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("triangle"),
            // Also a nullable sequence, and the other reason for this test.
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: f64::from(CLEAR[0]) / 255.0,
                        g: f64::from(CLEAR[1]) / 255.0,
                        b: f64::from(CLEAR[2]) / 255.0,
                        a: f64::from(CLEAR[3]) / 255.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &colour_group, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let mapped = Settled::new();
    slice.map_async(wgpu::MapMode::Read, mapped.notifier());
    mapped
        .await
        .map_err(|error| format!("map_async: {error:?}"))?;

    let pixels = slice
        .get_mapped_range()
        .map_err(|error| format!("get_mapped_range: {error}"))?
        .to_vec();
    readback.unmap();
    Ok((described, pixels))
}

/// A future for a `wgpu` callback that fires once.
///
/// `map_async` reports completion by calling back, and on WebGPU that call comes
/// from the JavaScript event loop rather than from a `device.poll`, which is a
/// no-op there. `wgpu`'s own tests await `async_poll` instead; nothing like it
/// exists on this backend, so this is the smallest thing that turns one callback
/// into one `await`. `Rc`, not `Arc`: the callback runs on the thread that
/// registered it.
struct Settled<T>(Rc<RefCell<(Option<T>, Option<Waker>)>>);

impl<T> Settled<T> {
    fn new() -> Self {
        Self(Rc::new(RefCell::new((None, None))))
    }

    /// The callback to hand to `wgpu`.
    fn notifier(&self) -> impl FnOnce(T) + 'static
    where
        T: 'static,
    {
        let state = Rc::clone(&self.0);
        move |value| {
            let waker = {
                let mut state = state.borrow_mut();
                state.0 = Some(value);
                state.1.take()
            };
            if let Some(waker) = waker {
                waker.wake();
            }
        }
    }
}

impl<T> Future for Settled<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<T> {
        let mut state = self.0.borrow_mut();
        match state.0.take() {
            Some(value) => Poll::Ready(value),
            None => {
                state.1 = Some(context.waker().clone());
                Poll::Pending
            }
        }
    }
}
