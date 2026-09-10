//! A thread-local handle to a device, backed by the [`glutin`] crate.

use super::connection::{Connection, NativeConnection};
use super::context::{Context, ContextDescriptor, ContextState, NativeContext};
use super::surface::{NativeWidget, Surface, SurfaceObjects, SurfaceTexture};
use crate::base::egl::surface::EGLBackedSurface;
use crate::context::{ContextID, CREATE_CONTEXT_MUTEX};
use crate::egl::types::{EGLContext, EGLDisplay};
use crate::gl;
use crate::surface::Framebuffer;
use crate::{
    ContextAttributeFlags, ContextAttributes, Error, GLApi, Gl, SurfaceAccess, SurfaceInfo,
    SurfaceType, WindowingApiError,
};

use euclid::default::Size2D;
use glow::HasContext;
use glutin::config::{ConfigSurfaceTypes, ConfigTemplateBuilder, GetGlConfig};
use glutin::context::{
    AsRawContext, ContextApi, ContextAttributesBuilder, GlProfile, NotCurrentContext,
    PossiblyCurrentContext, RawContext, Version,
};
use glutin::display::{AsRawDisplay, RawDisplay};
use glutin::prelude::*;
use glutin::surface::{SurfaceAttributesBuilder, WindowSurface};
use std::cell::RefCell;
use std::env;
use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::os::raw::c_void;
use std::rc::Rc;

static LIBGL_ALWAYS_SOFTWARE_ENV_VAR: &str = "LIBGL_ALWAYS_SOFTWARE";
static DRI_PRIME_ENV_VAR: &str = "DRI_PRIME";

const SURFACE_GL_TEXTURE_TARGET: u32 = gl::TEXTURE_2D;

/// Represents a hardware display adapter that can be used for rendering.
///
/// Adapters can be sent between threads. To render with an adapter, open a thread-local `Device`.
#[derive(Clone, Debug)]
pub enum Adapter {
    #[doc(hidden)]
    Hardware,
    #[doc(hidden)]
    HardwarePrime,
    #[doc(hidden)]
    Software,
}

impl Adapter {
    #[inline]
    pub(crate) fn hardware() -> Adapter {
        Adapter::HardwarePrime
    }

    #[inline]
    pub(crate) fn low_power() -> Adapter {
        Adapter::Hardware
    }

    #[inline]
    pub(crate) fn software() -> Adapter {
        Adapter::Software
    }

    pub(crate) fn set_environment_variables(&self) {
        match *self {
            Adapter::Hardware | Adapter::HardwarePrime => {
                env::remove_var(LIBGL_ALWAYS_SOFTWARE_ENV_VAR);
            }
            Adapter::Software => {
                env::set_var(LIBGL_ALWAYS_SOFTWARE_ENV_VAR, "1");
            }
        }

        match *self {
            Adapter::Software | Adapter::Hardware => {
                env::remove_var(DRI_PRIME_ENV_VAR);
            }
            Adapter::HardwarePrime => {
                env::set_var(DRI_PRIME_ENV_VAR, "1");
            }
        }
    }
}

/// A thread-local handle to a device.
///
/// Devices contain most of the relevant surface management methods.
pub struct Device {
    pub(crate) connection: Connection,
    pub(crate) adapter: Adapter,
    /// The context, if any, that was last made current through this device.
    pub(crate) last_current: RefCell<Option<Rc<RefCell<Option<ContextState>>>>>,
}

/// Wraps an adapter.
#[derive(Clone)]
pub struct NativeDevice {
    /// The hardware adapter corresponding to this device.
    pub adapter: Adapter,
}

impl Device {
    pub(crate) fn new(connection: &Connection, adapter: &Adapter) -> Result<Device, Error> {
        adapter.set_environment_variables();
        Ok(Device {
            connection: connection.clone(),
            adapter: adapter.clone(),
            last_current: RefCell::new(None),
        })
    }

    /// Returns the native device corresponding to this device.
    #[inline]
    pub fn native_device(&self) -> NativeDevice {
        NativeDevice {
            adapter: self.adapter(),
        }
    }

    /// Returns the display server connection that this device was created with.
    #[inline]
    pub fn connection(&self) -> Connection {
        self.connection.clone()
    }

    /// Returns the adapter that this device was created with.
    #[inline]
    pub fn adapter(&self) -> Adapter {
        self.adapter.clone()
    }

    /// Returns the OpenGL API flavor that this device supports (OpenGL or OpenGL ES).
    #[inline]
    pub fn gl_api(&self) -> GLApi {
        self.connection.gl_api()
    }

    fn native_connection(&self) -> NativeConnection {
        self.connection.native_connection()
    }

    /// Creates a context descriptor with the given attributes.
    pub fn create_context_descriptor(
        &self,
        attributes: &ContextAttributes,
    ) -> Result<ContextDescriptor, Error> {
        self.adapter.set_environment_variables();

        let template = ConfigTemplateBuilder::new()
            .with_alpha_size(if attributes.flags.contains(ContextAttributeFlags::ALPHA) {
                8
            } else {
                0
            })
            .with_depth_size(if attributes.flags.contains(ContextAttributeFlags::DEPTH) {
                24
            } else {
                0
            })
            .with_stencil_size(
                if attributes.flags.contains(ContextAttributeFlags::STENCIL) {
                    8
                } else {
                    0
                },
            )
            .with_surface_type(ConfigSurfaceTypes::WINDOW)
            .build();

        let display = self.native_connection().display;
        let mut configs: Vec<_> = unsafe { display.find_configs(template) }
            .map_err(|_| Error::PixelFormatSelectionFailed(WindowingApiError::Failed))?
            .collect();
        if configs.is_empty() {
            return Err(Error::NoPixelFormatFound);
        }
        // Prefer configs with the fewest number of multisamples, then the most hardware
        // acceleration.
        configs.sort_by_key(|config| (config.num_samples(), !config.hardware_accelerated()));
        let config = configs.remove(0);

        Ok(ContextDescriptor {
            config,
            attributes: *attributes,
        })
    }

    /// Creates a new OpenGL context and makes it current.
    pub fn create_context(
        &self,
        descriptor: &ContextDescriptor,
        share_with: Option<&Context>,
    ) -> Result<Context, Error> {
        let context_api = match self.gl_api() {
            GLApi::GL => ContextApi::OpenGl(Some(Version::new(
                descriptor.attributes.version.major,
                descriptor.attributes.version.minor,
            ))),
            GLApi::GLES => ContextApi::Gles(Some(Version::new(
                descriptor.attributes.version.major,
                descriptor.attributes.version.minor,
            ))),
        };
        let profile = if descriptor
            .attributes
            .flags
            .contains(ContextAttributeFlags::COMPATIBILITY_PROFILE)
        {
            GlProfile::Compatibility
        } else {
            GlProfile::Core
        };

        let builder = ContextAttributesBuilder::new()
            .with_context_api(context_api)
            .with_profile(profile);

        let display = self.native_connection().display;

        let not_current = unsafe {
            match share_with {
                Some(context) => {
                    let inner = context.inner.borrow();
                    match inner.as_ref().ok_or(Error::IncompatibleSharedContext)? {
                        ContextState::NotCurrent(shared) => display.create_context(
                            &descriptor.config,
                            &builder.with_sharing(shared).build(None),
                        ),
                        ContextState::Current(shared) => display.create_context(
                            &descriptor.config,
                            &builder.with_sharing(shared).build(None),
                        ),
                    }
                }
                None => display.create_context(&descriptor.config, &builder.build(None)),
            }
        }
        .map_err(|_| Error::ContextCreationFailed(WindowingApiError::Failed))?;

        let current_state = make_surfaceless_current(not_current)
            .map_err(|_| Error::MakeCurrentFailed(WindowingApiError::Failed))?;

        let gl = unsafe {
            Gl::from_loader_function(|symbol| match std::ffi::CString::new(symbol) {
                Ok(c_str) => display.get_proc_address(&c_str) as *const _,
                Err(_) => std::ptr::null(),
            })
        };

        let id = {
            let mut next_context_id = CREATE_CONTEXT_MUTEX.lock().unwrap();
            let id = *next_context_id;
            next_context_id.0 += 1;
            id
        };

        Ok(Context {
            id,
            gl,
            descriptor: descriptor.clone(),
            inner: Rc::new(RefCell::new(Some(current_state))),
            framebuffer: RefCell::new(Framebuffer::None),
            destroyed: false,
        })
    }

    /// Wraps a native context object in an OpenGL context.
    ///
    /// `glutin` provides no API to reconstruct one of its own context types from a raw pointer,
    /// so this doesn't create an independent context object. Instead, provided the native
    /// context matches whichever `surfman` context is currently current on this device, it
    /// returns a new `Context` that shares the same underlying `glutin` context (and therefore
    /// the same GL state) as that context.
    pub unsafe fn create_context_from_native_context(
        &self,
        native_context: NativeContext,
    ) -> Result<Context, Error> {
        let inner = {
            let last_current = self.last_current.borrow();
            last_current
                .as_ref()
                .ok_or(Error::NoCurrentContext)?
                .clone()
        };

        let config = {
            let state = inner.borrow();
            let raw_context = match state.as_ref() {
                Some(ContextState::Current(pc)) => pc.raw_context(),
                Some(ContextState::NotCurrent(nc)) => nc.raw_context(),
                None => return Err(Error::NoCurrentContext),
            };
            if raw_context != native_context.0 {
                return Err(Error::IncompatibleNativeContext);
            }
            match state.as_ref() {
                Some(ContextState::Current(pc)) => pc.config(),
                Some(ContextState::NotCurrent(nc)) => nc.config(),
                None => unreachable!(),
            }
        };

        let display = self.native_connection().display;
        let gl = unsafe {
            Gl::from_loader_function(|symbol| match std::ffi::CString::new(symbol) {
                Ok(c_str) => display.get_proc_address(&c_str) as *const _,
                Err(_) => std::ptr::null(),
            })
        };

        let id = {
            let mut next_context_id = CREATE_CONTEXT_MUTEX.lock().unwrap();
            let id = *next_context_id;
            next_context_id.0 += 1;
            id
        };

        Ok(Context {
            id,
            gl,
            descriptor: ContextDescriptor {
                config,
                attributes: ContextAttributes::zeroed(),
            },
            inner,
            framebuffer: RefCell::new(Framebuffer::None),
            destroyed: false,
        })
    }

    /// Destroys a context.
    pub fn destroy_context(&self, context: &mut Context) -> Result<(), Error> {
        if let Ok(Some(mut surface)) = self.unbind_surface_from_context(context) {
            self.destroy_surface(context, &mut surface)?;
        }

        let is_last_current = matches!(
            &*self.last_current.borrow(),
            Some(last_current) if Rc::ptr_eq(last_current, &context.inner)
        );
        if is_last_current {
            self.last_current.borrow_mut().take();
        }

        *context.inner.borrow_mut() = None;
        context.destroyed = true;
        Ok(())
    }

    /// Returns the descriptor that this context was created with.
    #[inline]
    pub fn context_descriptor(&self, context: &Context) -> ContextDescriptor {
        context.descriptor.clone()
    }

    /// Returns the attributes that the context descriptor was created with.
    #[inline]
    pub fn context_descriptor_attributes(
        &self,
        context_descriptor: &ContextDescriptor,
    ) -> ContextAttributes {
        context_descriptor.attributes
    }

    /// Makes the context the current OpenGL context for this thread.
    pub fn make_context_current(&self, context: &Context) -> Result<(), Error> {
        let mut inner = context.inner.borrow_mut();
        let state = inner.take().ok_or(Error::NoCurrentContext)?;

        let framebuffer = context.framebuffer.borrow();
        let window_surface = match &*framebuffer {
            Framebuffer::Surface(surface) => match &surface.objects {
                SurfaceObjects::Window { glutin_surface, .. } => Some(glutin_surface),
                SurfaceObjects::Generic { .. } => None,
            },
            _ => None,
        };

        let result = match window_surface {
            Some(glutin_surface) => match state {
                ContextState::NotCurrent(nc) => {
                    nc.make_current(glutin_surface).map(ContextState::Current)
                }
                ContextState::Current(pc) => pc
                    .make_current(glutin_surface)
                    .map(|_| ContextState::Current(pc)),
            },
            None => match state {
                ContextState::NotCurrent(nc) => make_surfaceless_current(nc),
                ContextState::Current(pc) => {
                    make_surfaceless_current_in_place(&pc).map(|()| ContextState::Current(pc))
                }
            },
        };
        drop(framebuffer);

        match result {
            Ok(new_state) => {
                *inner = Some(new_state);
                drop(inner);
                *self.last_current.borrow_mut() = Some(context.inner.clone());
                Ok(())
            }
            Err(_) => Err(Error::MakeCurrentFailed(WindowingApiError::Failed)),
        }
    }

    /// Removes the current OpenGL context from this thread.
    pub fn make_no_context_current(&self) -> Result<(), Error> {
        if let Some(last_current) = self.last_current.borrow_mut().take() {
            if let Some(ContextState::Current(possibly_current)) = &*last_current.borrow() {
                possibly_current
                    .make_not_current_in_place()
                    .map_err(|_| Error::MakeCurrentFailed(WindowingApiError::Failed))?;
            }
        }
        Ok(())
    }

    /// Fetches the address of an OpenGL function associated with this context.
    #[inline]
    pub fn get_proc_address(&self, _context: &Context, symbol_name: &str) -> *const c_void {
        match std::ffi::CString::new(symbol_name) {
            Ok(c_str) => self.native_connection().display.get_proc_address(&c_str),
            Err(_) => std::ptr::null(),
        }
    }

    /// Attaches a surface to a context for rendering.
    pub fn bind_surface_to_context(
        &self,
        context: &mut Context,
        surface: Surface,
    ) -> Result<(), (Error, Surface)> {
        if surface.context_id() != context.id {
            return Err((Error::IncompatibleSurface, surface));
        }
        if !matches!(&*context.framebuffer.borrow(), Framebuffer::None) {
            return Err((Error::SurfaceAlreadyBound, surface));
        }

        *context.framebuffer.borrow_mut() = Framebuffer::Surface(surface);
        if let Err(err) = self.make_context_current(context) {
            let mut framebuffer = context.framebuffer.borrow_mut();
            let surface = match std::mem::replace(&mut *framebuffer, Framebuffer::None) {
                Framebuffer::Surface(surface) => surface,
                _ => unreachable!(),
            };
            return Err((err, surface));
        }

        Ok(())
    }

    /// Removes and returns any attached surface from this context.
    pub fn unbind_surface_from_context(
        &self,
        context: &mut Context,
    ) -> Result<Option<Surface>, Error> {
        let mut framebuffer = context.framebuffer.borrow_mut();
        let surface = match std::mem::replace(&mut *framebuffer, Framebuffer::None) {
            Framebuffer::None => return Ok(None),
            Framebuffer::External(_) => return Ok(None),
            Framebuffer::Surface(surface) => surface,
        };
        drop(framebuffer);

        self.make_context_current(context)?;
        unsafe {
            context.gl.finish();
        }

        Ok(Some(surface))
    }

    /// Returns a unique ID representing a context.
    #[inline]
    pub fn context_id(&self, context: &Context) -> ContextID {
        context.id
    }

    /// Returns various information about the surface attached to a context.
    pub fn context_surface_info(&self, context: &Context) -> Result<Option<SurfaceInfo>, Error> {
        match &*context.framebuffer.borrow() {
            Framebuffer::Surface(surface) => Ok(Some(self.surface_info(surface))),
            _ => Ok(None),
        }
    }

    /// Returns the native context associated with the given context.
    pub fn native_context(&self, context: &Context) -> NativeContext {
        let inner = context.inner.borrow();
        let raw = match inner.as_ref() {
            Some(ContextState::NotCurrent(nc)) => nc.raw_context(),
            Some(ContextState::Current(pc)) => pc.raw_context(),
            None => panic!("Context was already destroyed!"),
        };
        NativeContext(raw)
    }

    /// Creates either a generic or a widget surface, depending on the supplied surface type.
    pub fn create_surface(
        &self,
        context: &Context,
        _surface_access: SurfaceAccess,
        surface_type: SurfaceType<NativeWidget>,
    ) -> Result<Surface, Error> {
        match surface_type {
            SurfaceType::Generic { size } => self.create_generic_surface(context, size),
            SurfaceType::Widget { native_widget } => {
                self.create_window_surface(context, native_widget)
            }
        }
    }

    fn create_generic_surface(
        &self,
        context: &Context,
        size: Size2D<i32>,
    ) -> Result<Surface, Error> {
        self.make_context_current(context)?;
        let egl_display = raw_egl_display(&self.native_connection().display)?;
        let egl_context = raw_egl_context(context)?;

        let egl_surface = EGLBackedSurface::new_generic(
            &context.gl,
            egl_display,
            egl_context,
            context.id,
            &context.descriptor.attributes,
            &size,
        );

        Ok(Surface {
            objects: SurfaceObjects::Generic(egl_surface),
        })
    }

    fn create_window_surface(
        &self,
        context: &Context,
        native_widget: NativeWidget,
    ) -> Result<Surface, Error> {
        let width = NonZeroU32::new(native_widget.size.width.max(1) as u32).unwrap();
        let height = NonZeroU32::new(native_widget.size.height.max(1) as u32).unwrap();
        let attrs = SurfaceAttributesBuilder::<WindowSurface>::new().build(
            native_widget.raw_window_handle,
            width,
            height,
        );

        let display = self.native_connection().display;
        let glutin_surface =
            unsafe { display.create_window_surface(&context.descriptor.config, &attrs) }
                .map_err(|_| Error::SurfaceCreationFailed(WindowingApiError::Failed))?;

        Ok(Surface {
            objects: SurfaceObjects::Window {
                glutin_surface,
                context_id: context.id,
                size: native_widget.size,
            },
        })
    }

    /// Creates a surface texture from an existing generic surface for use with the given
    /// context.
    pub fn create_surface_texture(
        &self,
        context: &mut Context,
        surface: Surface,
    ) -> Result<SurfaceTexture, (Error, Surface)> {
        let egl_surface = match surface.objects {
            SurfaceObjects::Window { .. } => return Err((Error::WidgetAttached, surface)),
            SurfaceObjects::Generic(egl_surface) => egl_surface,
        };

        if let Err(err) = self.make_context_current(context) {
            return Err((
                err,
                Surface {
                    objects: SurfaceObjects::Generic(egl_surface),
                },
            ));
        }

        match egl_surface.to_surface_texture(&context.gl) {
            Ok(surface_texture) => Ok(SurfaceTexture {
                surface: surface_texture,
                phantom: PhantomData,
            }),
            Err((err, egl_surface)) => Err((
                err,
                Surface {
                    objects: SurfaceObjects::Generic(egl_surface),
                },
            )),
        }
    }

    /// Destroys a surface.
    pub fn destroy_surface(
        &self,
        context: &mut Context,
        surface: &mut Surface,
    ) -> Result<(), Error> {
        if surface.context_id() != context.id {
            return Err(Error::IncompatibleSurface);
        }

        match &mut surface.objects {
            SurfaceObjects::Window { .. } => {}
            SurfaceObjects::Generic(egl_surface) => {
                self.make_context_current(context)?;
                let egl_display = raw_egl_display(&self.native_connection().display)?;
                egl_surface.destroy(&context.gl, egl_display, context.id)?;
            }
        }

        Ok(())
    }

    /// Destroys a surface texture and returns the underlying surface.
    pub fn destroy_surface_texture(
        &self,
        context: &mut Context,
        surface_texture: SurfaceTexture,
    ) -> Result<Surface, (Error, SurfaceTexture)> {
        if let Err(err) = self.make_context_current(context) {
            return Err((err, surface_texture));
        }
        let egl_surface = surface_texture.surface.destroy(&context.gl);
        Ok(Surface {
            objects: SurfaceObjects::Generic(egl_surface),
        })
    }

    /// Returns the OpenGL texture target needed to read from this surface texture.
    #[inline]
    pub fn surface_gl_texture_target(&self) -> u32 {
        SURFACE_GL_TEXTURE_TARGET
    }

    /// Displays the contents of the currently bound surface to the screen, if it is a widget
    /// surface.
    pub fn present_bound_surface(&self, context: &mut Context) -> Result<(), Error> {
        let framebuffer = context.framebuffer.borrow();
        let surface = match &*framebuffer {
            Framebuffer::Surface(surface) => surface,
            _ => return Err(Error::NoWidgetAttached),
        };
        let glutin_surface = match &surface.objects {
            SurfaceObjects::Window { glutin_surface, .. } => glutin_surface,
            SurfaceObjects::Generic(_) => return Err(Error::NoWidgetAttached),
        };

        let inner = context.inner.borrow();
        match inner.as_ref().ok_or(Error::NoCurrentContext)? {
            ContextState::Current(possibly_current) => glutin_surface
                .swap_buffers(possibly_current)
                .map_err(|_| Error::PresentFailed(WindowingApiError::Failed)),
            ContextState::NotCurrent(_) => Err(Error::NoCurrentContext),
        }
    }

    /// Displays the contents of a widget surface on screen.
    pub fn present_surface(&self, context: &Context, surface: &mut Surface) -> Result<(), Error> {
        let glutin_surface = match &surface.objects {
            SurfaceObjects::Window { glutin_surface, .. } => glutin_surface,
            SurfaceObjects::Generic(_) => return Err(Error::NoWidgetAttached),
        };

        self.make_context_current(context)?;
        let inner = context.inner.borrow();
        match inner.as_ref().ok_or(Error::NoCurrentContext)? {
            ContextState::Current(possibly_current) => glutin_surface
                .swap_buffers(possibly_current)
                .map_err(|_| Error::PresentFailed(WindowingApiError::Failed)),
            ContextState::NotCurrent(_) => Err(Error::NoCurrentContext),
        }
    }

    /// If the currently bound surface is a widget surface, resize it.
    pub fn resize_bound_surface(
        &self,
        context: &mut Context,
        size: Size2D<i32>,
    ) -> Result<(), Error> {
        self.make_context_current(context)?;

        let mut framebuffer = context.framebuffer.borrow_mut();
        if let Framebuffer::Surface(surface) = &mut *framebuffer {
            if let SurfaceObjects::Window { glutin_surface, .. } = &surface.objects {
                let inner = context.inner.borrow();
                if let Some(ContextState::Current(possibly_current)) = inner.as_ref() {
                    let width = NonZeroU32::new(size.width.max(1) as u32).unwrap();
                    let height = NonZeroU32::new(size.height.max(1) as u32).unwrap();
                    glutin_surface.resize(possibly_current, width, height);
                }
            }
            surface.set_size(size);
        }
        Ok(())
    }

    /// Resizes a widget surface.
    pub fn resize_surface(
        &self,
        context: &Context,
        surface: &mut Surface,
        size: Size2D<i32>,
    ) -> Result<(), Error> {
        if let SurfaceObjects::Window { glutin_surface, .. } = &surface.objects {
            self.make_context_current(context)?;
            let inner = context.inner.borrow();
            if let Some(ContextState::Current(possibly_current)) = inner.as_ref() {
                let width = NonZeroU32::new(size.width.max(1) as u32).unwrap();
                let height = NonZeroU32::new(size.height.max(1) as u32).unwrap();
                glutin_surface.resize(possibly_current, width, height);
            }
        }
        surface.set_size(size);
        Ok(())
    }

    /// Returns various information about the surface, including the framebuffer object needed
    /// to render to this surface.
    pub fn surface_info(&self, surface: &Surface) -> SurfaceInfo {
        surface.info()
    }

    /// Returns the OpenGL texture object containing the contents of this surface.
    #[inline]
    pub fn surface_texture_object(
        &self,
        surface_texture: &SurfaceTexture,
    ) -> Option<glow::Texture> {
        surface_texture.surface.texture_object
    }
}

/// Makes a not-current context current without attaching any real drawable to it.
///
/// Not every windowing system exposes pbuffer-capable EGL configs (notably, Mesa's Wayland EGL
/// platform typically does not), so we can't rely on a dummy pbuffer surface to give newly
/// created contexts (and generic, texture-backed surfaces) a drawable to be current with.
/// Instead, on EGL we use the `EGL_KHR_surfaceless_context` extension (exposed by `glutin` as
/// `make_current_surfaceless`) to make the context current with no surface at all.
fn make_surfaceless_current(not_current: NotCurrentContext) -> glutin::error::Result<ContextState> {
    match not_current {
        #[cfg(not(macos_platform))]
        NotCurrentContext::Egl(egl) => egl
            .make_current_surfaceless()
            .map(|pc| ContextState::Current(PossiblyCurrentContext::Egl(pc))),
        // Other backends (GLX, WGL, CGL) don't have a safe cross-platform surfaceless mode
        // exposed by `glutin`. This should not be reachable in practice, since this backend
        // always requests an EGL (or, on macOS, a CGL) display, but we handle it defensively.
        other => Ok(ContextState::Current(other.treat_as_possibly_current())),
    }
}

/// The in-place equivalent of [`make_surfaceless_current`], used when the context is already
/// (possibly) current and we just need to detach whatever surface was previously bound.
fn make_surfaceless_current_in_place(
    possibly_current: &PossiblyCurrentContext,
) -> glutin::error::Result<()> {
    match possibly_current {
        #[cfg(not(macos_platform))]
        PossiblyCurrentContext::Egl(egl) => egl.make_current_surfaceless(),
        _ => Ok(()),
    }
}

/// Extracts the raw `EGLDisplay` out of a `glutin` display.
///
/// Generic surfaces are backed directly by `EGLImage`s (via `crate::base::egl`, shared with the
/// other EGL-based backends) rather than by anything `glutin` creates, since `glutin`'s safe API
/// doesn't expose `EGLImage`. That's also what lets a `SurfaceTexture` be read from any context
/// on the same display, even one belonging to a different `Device` or thread, without needing an
/// explicit OpenGL sharing group.
fn raw_egl_display(display: &glutin::display::Display) -> Result<EGLDisplay, Error> {
    match display.raw_display() {
        #[cfg(not(macos_platform))]
        RawDisplay::Egl(ptr) => Ok(ptr as *mut c_void as EGLDisplay),
        _ => Err(Error::UnsupportedOnThisPlatform),
    }
}

/// Extracts the raw `EGLContext` out of a `surfman` context.
fn raw_egl_context(context: &Context) -> Result<EGLContext, Error> {
    let inner = context.inner.borrow();
    let raw_context = match inner.as_ref() {
        Some(ContextState::Current(pc)) => pc.raw_context(),
        Some(ContextState::NotCurrent(nc)) => nc.raw_context(),
        None => return Err(Error::NoCurrentContext),
    };
    match raw_context {
        #[cfg(not(macos_platform))]
        RawContext::Egl(ptr) => Ok(ptr as *mut c_void as EGLContext),
        _ => Err(Error::UnsupportedOnThisPlatform),
    }
}
