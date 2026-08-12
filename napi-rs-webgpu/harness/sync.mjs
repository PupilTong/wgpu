import { instantiateAddon } from "./instantiate.mjs";

const addon = await instantiateAddon();
if (!addon.syncRoundtrip()) {
  throw new Error(
    "Rust bytes remained stale after crossing Emnapi staging memory",
  );
}

console.log("ok: Rust → Emnapi staging memory → JavaScript bytes round-trip");
process.exit(0);
