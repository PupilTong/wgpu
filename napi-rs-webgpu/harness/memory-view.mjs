import { instantiateAddon } from "./instantiate.mjs";

const addon = await instantiateAddon();
if (!addon.memoryViewAliases()) {
  throw new Error(
    "JavaScript Uint8Array did not alias the Rust bytes in shared Wasm memory",
  );
}

console.log("ok: Rust and JavaScript alias one shared Wasm memory view");
process.exit(0);
