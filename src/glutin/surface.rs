//! Surfaces backed by the [`glutin`] crate.

use crate::base::egl::surface::{EGLBackedSurface, EGLSurfaceTexture};
use crate::{ContextID, SurfaceID, SurfaceInfo};

use euclid::default::Size2D;
use glutin::surface::{Surface as GlutinSurface, WindowSurface};
use rwh_06::RawWindowHandle;
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;

/// Represents a hardware buffer of pixels that can be rendered to via the GPU and either
/// displayed in a native widget or bound to a texture for reading.
///
/// Surfaces come in two varieties: generic and widget surfaces. Widget surfaces are backed by a
/// real `glutin` window surface and can be presented on screen, but cannot be read from a
/// texture. Generic surfaces are backed by an `EGLImage`-based OpenGL framebuffer object (the
/// same mechanism the other EGL-based backends use), which can be wrapped up in a
/// `SurfaceTexture` and read from *any* context on the same display connection, even one
/// belonging to a different `Device` or a different thread.
///
/// You must explicitly call `Device::destroy_surface()` to dispose of a surface.
pub struct Surface {
    pub(crate) objects: SurfaceObjects,
}

pub(crate) enum SurfaceObjects {
    Window {
        glutin_surface: GlutinSurface<WindowSurface>,
        context_id: ContextID,
        size: Size2D<i32>,
    },
    Generic(EGLBackedSurface),
}

unsafe impl Send for Surface {}

impl Debug for Surface {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f, "Surface({:x})", self.id().0)
    }
}

impl Surface {
    pub(crate) fn id(&self) -> SurfaceID {
        match &self.objects {
            SurfaceObjects::Window { .. } => SurfaceID(self as *const Surface as usize),
            SurfaceObjects::Generic(surface) => surface.id(),
        }
    }

    pub(crate) fn context_id(&self) -> ContextID {
        match &self.objects {
            SurfaceObjects::Window { context_id, .. } => *context_id,
            SurfaceObjects::Generic(surface) => surface.context_id,
        }
    }

    pub(crate) fn set_size(&mut self, new_size: Size2D<i32>) {
        match &mut self.objects {
            SurfaceObjects::Window { size, .. } => *size = new_size,
            SurfaceObjects::Generic(surface) => surface.size = new_size,
        }
    }

    pub(crate) fn info(&self) -> SurfaceInfo {
        match &self.objects {
            SurfaceObjects::Window {
                context_id, size, ..
            } => SurfaceInfo {
                size: *size,
                id: self.id(),
                context_id: *context_id,
                framebuffer_object: None,
            },
            SurfaceObjects::Generic(surface) => surface.info(),
        }
    }
}

/// Represents an OpenGL texture that wraps a surface.
///
/// Reading from the associated OpenGL texture reads from the surface. It is undefined behavior
/// to write to such a texture.
///
/// Surface textures are local to a context, but that context does not have to be the same
/// context as that associated with the underlying surface: it can belong to any device (or
/// thread) that shares the same display connection.
///
/// The surface texture must be destroyed with the `destroy_surface_texture()` method, or a panic
/// will occur.
pub struct SurfaceTexture {
    pub(crate) surface: EGLSurfaceTexture,
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
