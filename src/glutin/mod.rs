//! A cross-platform backend that defers to the [`glutin`] crate for OpenGL context and surface
//! creation.
//!
//! Unlike the other backends in this crate, which talk directly to EGL, GLX, CGL, or WGL,
//! this backend goes through `glutin`'s own cross-platform display/config/context/surface
//! abstraction. This is useful primarily for interoperating with applications that already use
//! `glutin` (for example, via `winit`) and want to opt into `surfman`'s surface-sharing model
//! without giving up `glutin` for context creation.
//!
//! [`glutin`]: https://crates.io/crates/glutin

pub mod connection;
pub mod context;
pub mod device;
pub mod surface;

crate::implement_interfaces!();
