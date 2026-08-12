// Loads the addon and checks what it drew.
//
// The three steps that matter are all here rather than hidden in a helper:
// bootstrap node-webgpu, instantiate the wasm module as a Node-API addon, and
// hand the addon the `GPU` object. Everything after that is the addon's.

import { create, globals } from "webgpu";
import { instantiateAddon } from "./instantiate.mjs";

// node-webgpu puts the WebGPU classes and the `GPUBufferUsage`-style namespaces
// on `globals` rather than on the global object, and `create` returns a `GPU`
// without installing a `navigator`. The classes have to be reachable by name
// because `napi-rs-webgpu` tests types with `instanceof`; the `GPU` itself is
// passed to the addon explicitly, which is what `install_gpu` is for.
Object.assign(globalThis, globals);
const gpu = create([]);

const addon = await instantiateAddon();
const [width, height, ...colours] = addon.expected();
const clear = Uint8Array.from(colours.slice(0, 4));
const draw = Uint8Array.from(colours.slice(4, 8));

addon.installWebgpu(gpu);
addon.start();

// The render runs on Node's event loop: `start` only queues it. Yielding to the
// macrotask queue lets the microtasks that drive it — the `then` callbacks
// behind every `await` in the addon — run between polls.
const pixels = await poll();
const counts = check(pixels);

console.log(`adapter: ${addon.adapter()}`);
console.log(
  `ok: ${width}x${height} through napi-rs-webgpu — ` +
    `${counts.draw} px ${hex(draw)} (triangle), ${counts.clear} px ${hex(clear)} (clear)`,
);
process.exit(0);

async function poll() {
  const deadline = Date.now() + 30_000;
  for (;;) {
    const result = addon.takeResult();
    if (result) return result;
    if (Date.now() > deadline) {
      throw new Error("timed out waiting for the render");
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

// Three things are checked, and they fail for different reasons:
//
//  * every pixel is exactly the clear colour or exactly the triangle colour —
//    a third value means a channel, a format or a colour space went wrong;
//  * neither colour is a rounding error's worth of the image — a draw that did
//    nothing, or a clear that was overwritten, shows up here;
//  * two pixels far from the diagonal are the right way round — this is what
//    distinguishes a correct render from a mirrored or rotated one.
function check(pixels) {
  const size = width * height * 4;
  if (pixels.length !== size) {
    throw new Error(`expected ${size} bytes back, got ${pixels.length}`);
  }

  const counts = { clear: 0, draw: 0 };
  for (let i = 0; i < pixels.length; i += 4) {
    const pixel = pixels.subarray(i, i + 4);
    if (equal(pixel, clear)) counts.clear++;
    else if (equal(pixel, draw)) counts.draw++;
    else {
      const index = i / 4;
      throw new Error(
        `pixel (${index % width}, ${Math.floor(index / width)}) is ${hex(pixel)}, ` +
          `expected ${hex(clear)} or ${hex(draw)}`,
      );
    }
  }

  const total = width * height;
  for (const [name, count] of Object.entries(counts)) {
    if (count < total * 0.4) {
      throw new Error(
        `only ${count} of ${total} pixels are the ${name} colour; the triangle ` +
          `should cover about half the target`,
      );
    }
  }

  // The triangle is the half below `x + y = 0` in clip space, which is the
  // lower-left half of the image: `(4, 59)` is inside it, `(59, 4)` is not.
  at(pixels, 4, 59, draw, "inside the triangle");
  at(pixels, 59, 4, clear, "outside the triangle");

  return counts;
}

function at(pixels, x, y, want, where) {
  const i = (y * width + x) * 4;
  const got = pixels.subarray(i, i + 4);
  if (!equal(got, want)) {
    throw new Error(
      `(${x}, ${y}) is ${where}: expected ${hex(want)}, got ${hex(got)}`,
    );
  }
}

function equal(a, b) {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
}

function hex(bytes) {
  return [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
}
