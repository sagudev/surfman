//! Surfaces backed by the [`glutin`] crate.

use crate::renderbuffers::Renderbuffers;
use crate::{ContextID, SurfaceID};

use euclid::default::Size2D;
use glow::{NativeFramebuffer, NativeTexture};
use glutin::surface::{Surface as GlutinSurface, WindowSurface};
use rwh_06::RawWindowHandle;
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;
use std::thread;

/// Represents a hardware buffer of pixels that can be rendered to via the GPU and either
/// displayed in a native widget or bound to a texture for reading.
///
/// Surfaces come in two varieties: generic and widget surfaces. Generic surfaces are backed by
/// an OpenGL framebuffer object with a texture color attachment, and can be wrapped up in a
/// `SurfaceTexture` for reading from another context; widget surfaces are backed by a real
/// `glutin` window surface and can be presented on screen, but cannot be read from a texture.
///
/// Surfaces must be destroyed with the `destroy_surface()` method, or a panic will occur.
pub struct Surface {
    pub(crate) context_id: ContextID,
    pub(crate) size: Size2D<i32>,
    pub(crate) objects: SurfaceObjects,
    pub(crate) destroyed: bool,
}

pub(crate) enum SurfaceObjects {
    Window {
        glutin_surface: GlutinSurface<WindowSurface>,
    },
    Generic {
        framebuffer_object: Option<NativeFramebuffer>,
        texture_object: Option<NativeTexture>,
        renderbuffers: Renderbuffers,
    },
}

unsafe impl Send for Surface {}

impl Debug for Surface {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "Surface({:x})", self.id().0)
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        if !self.destroyed && !thread::panicking() {
            panic!("Should have destroyed the surface first with `destroy_surface()`!")
        }
    }
}

impl Surface {
    pub(crate) fn id(&self) -> SurfaceID {
        match self.objects {
            SurfaceObjects::Window { .. } => SurfaceID(self as *const Surface as usize),
            SurfaceObjects::Generic { texture_object, .. } => {
                SurfaceID(texture_object.map_or(0, |texture| texture.0.get() as usize))
            }
        }
    }
}

/// Represents an OpenGL texture that wraps a surface.
///
/// Reading from the associated OpenGL texture reads from the surface. It is undefined behavior
/// to write to such a texture.
///
/// Surface textures are local to a context, but that context does not have to be the same
/// context as that associated with the underlying surface, as long as both contexts are part of
/// the same device (and therefore share an OpenGL sharing group).
///
/// The surface texture must be destroyed with the `destroy_surface_texture()` method, or a panic
/// will occur.
pub struct SurfaceTexture {
    pub(crate) surface: Surface,
    pub(crate) phantom: PhantomData<*const ()>,
}

impl Debug for SurfaceTexture {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "SurfaceTexture({:?})", self.surface)
    }
}

/// A native widget (window or view) to render a widget surface into, along with its size.
pub struct NativeWidget {
    pub(crate) raw_window_handle: RawWindowHandle,
    pub(crate) size: Size2D<i32>,
}

/// Returns a pointer to the underlying surface data for reading or writing by the CPU.
///
/// This backend does not support direct CPU access to surface data.
pub struct SurfaceDataGuard<'a> {
    #[allow(dead_code)]
    phantom: PhantomData<&'a ()>,
}
