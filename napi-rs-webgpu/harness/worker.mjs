// The worker side of emnapi's threaded runtime.
//
// Nothing in this harness starts a thread, but a `wasm32-wasip1-threads` module
// links the threaded emnapi and the runtime will not instantiate without a way
// to make one. It re-instantiates the same module against the same shared memory
// and answers the main thread's messages.

import { parentPort } from "node:worker_threads";

import {
  MessageHandler,
  WASI,
  instantiateNapiModuleSync,
} from "@napi-rs/wasm-runtime";

const handler = new MessageHandler({
  onLoad({ wasmModule, wasmMemory }) {
    return instantiateNapiModuleSync(wasmModule, {
      childThread: true,
      wasi: new WASI({ print: console.log, printErr: console.error }),
      overwriteImports(imports) {
        imports.env = {
          ...imports.env,
          ...imports.napi,
          ...imports.emnapi,
          memory: wasmMemory,
        };
        return imports;
      },
    });
  },
});

parentPort.on("message", (data) => handler.handle({ data }));
