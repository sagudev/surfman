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

impl Drop for Context {
    fn drop(&mut self) {
        if !self.destroyed && !thread::panicking() {
            panic!("Contexts must be destroyed explicitly with `destroy_context`!")
        }
    }
}
