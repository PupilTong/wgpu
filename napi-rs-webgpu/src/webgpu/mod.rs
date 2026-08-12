//! The WebGPU bindings.
//!
//! Every declaration is written in the DSL that [`crate::dsl`] defines: `js_type!`
//! for the handle, `webgpu_members!` for an interface's members,
//! `webgpu_dictionary!` for a dictionary's, `webgpu_enum!` for a string
//! enumeration. What each one is made of comes from `tools/surface.json`, which
//! `tools/extract_surface.py` derives from WebGPU's IDL — by way of the bindings
//! web-sys generates from it — and from the members `wgpu`'s backend actually
//! reaches.
//!
//! Nothing here is written by hand. `python3 tools/generate.py` rewrites
//! [`generated`] from that spec, so re-deriving the surface after a WebGPU update
//! is a command rather than an editing session.

mod generated;

pub use generated::*;
