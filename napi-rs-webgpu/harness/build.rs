//! `napi-build` contributes the whole WASI link line: it links emnapi's archive
//! from `EMNAPI_LINK_DIR`, exports `napi_register_wasm_v1` and `_initialize`,
//! imports the memory so the host can supply a shared one, and links
//! `crt1-reactor.o` so the module initialises as a reactor rather than a command.
fn main() {
    napi_build::setup();
}
