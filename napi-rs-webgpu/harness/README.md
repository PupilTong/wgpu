# `napi-rs-webgpu` acceptance harness

Everything else about `napi-rs-webgpu` can be checked by compiling it. That the
bindings reach a real WebGPU implementation can only be checked by reaching one.

This builds `wgpu` for `wasm32-wasip1-threads` as a Node-API addon, loads it in
Node with `@napi-rs/wasm-runtime` + emnapi, hands it the `GPU` object from the
[`webgpu`](https://www.npmjs.com/package/webgpu) npm package (node-webgpu, which
is Dawn), and checks the pixels that come back.

## Running it

```bash
npm install
npm test
```

`npm test` builds the addon and runs it. Expected output on a machine with a GPU:

```
adapter: BrowserWebGpu / Metal driver on macOS Version 26.4.1 (Build 25E253) /
ok: 64x64 through napi-rs-webgpu — 2016 px 339966ff (triangle), 2080 px 112233ff (clear)
```

The GPU-free memory-view regression test is `npm run test:view`. It creates an
Emnapi `Uint8Array` view directly over Rust bytes in shared Wasm memory and
checks that mutations are visible from both sides without a staging allocation
or a synchronization copy. CI runs this narrower test without requiring an
adapter.

The build alone is `npm run build`, which is:

```bash
EMNAPI_LINK_DIR="$PWD/node_modules/emnapi/lib/wasm32-wasip1-threads" \
  cargo build --target wasm32-wasip1-threads --release
```

`EMNAPI_LINK_DIR` is what `napi-build` reads to find emnapi's archive; it must be
absolute, because the value is passed straight through to `rustc-link-search`.
Nothing else is needed — no `RUSTFLAGS`, no `-Z build-std`, no `WASI_SDK_PATH`,
and no nightly. The target ships with the pinned toolchain.

## What it renders, and what that proves

A 64×64 `Rgba8Unorm` texture, cleared to `#112233ff`, with a triangle covering the
lower-left half drawn in `#339966ff`, copied to a buffer and mapped for reading.
The triangle colour is uploaded through `queue.writeBuffer`, so the run also
checks that WebGPU accepts the temporary `Uint8Array` view over the shared Wasm
memory directly, without a Rust-to-JavaScript staging copy.
64 pixels of RGBA is exactly 256 bytes, which is `COPY_BYTES_PER_ROW_ALIGNMENT`,
so the readback needs no row padding and the mapped bytes are the image.

Half the target rather than all of it, and two colours rather than one, because a
full-screen draw would pass with the draw doing nothing — a clear alone produces a
uniform image too. `run.mjs` checks three separate things:

- every pixel is exactly one of the two colours, so a wrong channel order, format
  or colour space fails;
- each colour covers at least 40% of the image, so a draw that did nothing or a
  clear that was overwritten fails;
- `(4, 59)` is the triangle and `(59, 4)` is the clear, so a mirrored or rotated
  render fails.

Between them these exercise the parts of the binding that a `cargo check` cannot:
`requestAdapter` and `requestDevice` as real promises awaited from the JavaScript
event loop, `createShaderModule` with WGSL, `createRenderPipeline` with its
`sequence<GPUColorTargetState?>`, `beginRenderPass` with its
`sequence<GPURenderPassColorAttachment?>` — the nullable sequences that
`JsOption<T>` exists for — `queue.writeBuffer` from an Emnapi shared-memory
view, `copyTextureToBuffer`, and `mapAsync`, whose callback
arrives from a `then` job rather than from `device.poll`, which is a no-op on this
backend.

## Why it is shaped like this

Three exports rather than one promise, because JavaScript owns the event loop:
`installWebgpu` gives the crate the environment and the `GPU`, `start` queues the
render and returns, and `takeResult` answers `null` until there is something. The
caller polls from its own `setTimeout` loop, so Node decides when work runs and
the addon needs none of napi-rs' promise machinery — which is not what is under
test here.

`installWebgpu` is also where `napi_rs_webgpu::install` happens. napi-rs'
`#[module_init]` is a static constructor: it runs before Node-API exists and is
handed no environment, so the first `#[napi]` call is the earliest point one is
available.

node-webgpu returns its `GPU` from `create([])` and installs no `navigator`, so
`napi_rs_webgpu::gpu()`'s `navigator.gpu` fallback finds nothing and the explicit
`install_gpu` is not optional. Its WebGPU classes come from the `globals` export
and have to be put on the global object, because the bindings recognise types with
`instanceof`.

## Pixel test is local-only

CI runners have no GPU, and node-webgpu needs one. The build half of this is
covered — the WASI job compiles `wgpu` for both WASI targets, asserts the
dependency graph, and runs the GPU-free Emnapi memory-view alias test — but the
pixel run remains a local check.

## `wasm32-wasip1` without threads

The bindings also compile for plain `wasm32-wasip1`, and CI checks that target.
This runtime harness deliberately uses `wasm32-wasip1-threads`: it covers the
same shared-memory NAPI-RS/Emnapi route browser embedders use and keeps one
acceptance artifact rather than duplicating the run for both targets.
