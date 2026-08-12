// Instantiates the harness as a Node-API WASI addon. Kept independent of
// node-webgpu so `memory-view.mjs` can exercise Emnapi's zero-copy Wasm memory
// view on a machine with no GPU.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";

import {
  WASI,
  createContext,
  emnapiAsyncWorkPlugin,
  emnapiTSFNPlugin,
  instantiateNapiModule,
} from "@napi-rs/wasm-runtime";

const WASM = fileURLToPath(
  new URL(
    "./target/wasm32-wasip1-threads/release/napi_rs_webgpu_harness.wasm",
    import.meta.url,
  ),
);

// The host supplies the memory because `napi-build` links with
// `--import-memory`. It must be shared and at least the module's declared
// minimum, which `-zstack-size=64000000` currently puts at 978 pages.
const DECLARED_MINIMUM_PAGES = 978;
const memory = new WebAssembly.Memory({
  initial: 4000,
  maximum: 65536,
  shared: true,
});

export async function instantiateAddon() {
  const context = createContext({ autoDestroy: false });
  context.suppressDestroy();
  const wasi = new WASI({ print: console.log, printErr: console.error });

  try {
    const { napiModule } = await instantiateNapiModule(await readFile(WASM), {
      context,
      plugins: [emnapiAsyncWorkPlugin, emnapiTSFNPlugin],
      wasi,
      overwriteImports(imports) {
        imports.env = {
          ...imports.env,
          ...imports.napi,
          ...imports.emnapi,
          memory,
        };
        return imports;
      },
      // A wasm module has no static constructors, so the `#[napi]`
      // registrations are exported and must run before the exports object is
      // created.
      beforeInit({ instance }) {
        for (const name of Object.keys(instance.exports)) {
          if (name.startsWith("__napi_register__")) instance.exports[name]();
        }
      },
      onCreateWorker: () =>
        new Worker(new URL("./worker.mjs", import.meta.url), {
          env: process.env,
        }),
    });
    return napiModule.exports;
  } catch (error) {
    if (String(error).includes("memory")) {
      throw new Error(
        `${error}\n\nThe host memory must satisfy napi-build's link line: ` +
          `at least ${DECLARED_MINIMUM_PAGES} pages and shared.`,
      );
    }
    throw error;
  }
}
