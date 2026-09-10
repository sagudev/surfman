//! OpenGL contexts backed by the [`glutin`] crate.

use super::surface::Surface;
use crate::context::ContextID;
use crate::surface::Framebuffer;
use crate::Gl;

use glutin::config::Config as GlutinConfig;
use glutin::context::{NotCurrentContext, PossiblyCurrentContext, RawContext};
use std::cell::RefCell;
use std::rc::Rc;
use std::thread;

/// Information needed to create a context.
///
/// `surfman` calls this a "context descriptor"; other APIs call this a "config" or a "pixel
/// format".
#[derive(Clone)]
pub struct ContextDescriptor {
    pub(crate) config: GlutinConfig,
    pub(crate) attributes: crate::ContextAttributes,
}

/// The state of the underlying `glutin` context.
///
/// `glutin` represents contexts as either "not current" or "possibly current" types that get
/// *moved* between states. `surfman`, on the other hand, borrows its `Context` in its `Device`
/// methods, so the actual `glutin` context is kept behind a `RefCell` and swapped in and out of
/// these two states as needed.
pub(crate) enum ContextState {
    #[allow(dead_code)]
    NotCurrent(NotCurrentContext),
    Current(PossiblyCurrentContext),
}

/// Represents an OpenGL rendering context.
///
/// Each context has a descriptor associated with it, created with `Device::create_context`.
///
/// Contexts must be explicitly destroyed with `Device::destroy_context`, or a panic will occur.
pub struct Context {
    pub(crate) id: ContextID,
    pub(crate) gl: Gl,
    pub(crate) descriptor: ContextDescriptor,
    pub(crate) inner: Rc<RefCell<Option<ContextState>>>,
    pub(crate) framebuffer: RefCell<Framebuffer<Surface, ()>>,
    pub(crate) destroyed: bool,
}

/// A wrapper around a native `glutin` context.
#[derive(Clone, Copy, Debug)]
pub struct NativeContext(pub RawContext);

impl NativeContext {
    /// Returns the context that is currently current on this thread, if any.
    ///
    /// Note that `glutin` doesn't provide a way to reconstruct one of its own context types from
    /// a raw pointer, so `Device::create_context_from_native_context()` doesn't create an
    /// independent copy of the context; instead, it hands back a new `Context` value that shares
    /// the same underlying `glutin` context as whichever `surfman` context is current on the
    /// device (the only one that could plausibly be the "current" context to begin with).
    #[cfg(not(macos_platform))]
    pub fn current() -> Result<NativeContext, crate::Error> {
        crate::base::egl::device::EGL_FUNCTIONS.with(|egl| unsafe {
            let egl_context = egl.GetCurrentContext();
            if egl_context == crate::egl::NO_CONTEXT {
                Err(crate::Error::NoCurrentContext)
            } else {
                Ok(NativeContext(RawContext::Egl(egl_context as *const _)))
            }
        })
    }

    /// Returns the context that is currently current on this thread, if any.
    #[cfg(macos_platform)]
    pub fn current() -> Result<NativeContext, crate::Error> {
        Err(crate::Error::Unimplemented)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.destroyed && !thread::panicking() {
            panic!("Contexts must be destroyed explicitly with `destroy_context`!")
        }
    }
}
