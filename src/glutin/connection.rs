//! A connection to the display server, backed by the [`glutin`] crate.

use super::device::{Adapter, Device, NativeDevice};
use super::surface::NativeWidget;
use crate::{Error, GLApi};

use euclid::default::Size2D;
use glutin::display::{Display, DisplayApiPreference};
use rwh_06::RawDisplayHandle;
use std::os::raw::c_void;
use std::ptr::NonNull;

#[cfg(free_unix)]
use wayland_sys::client::wayland_client_handle;
#[cfg(x11_platform)]
use x11_dl::xlib::Xlib;

/// Identifies which native windowing system a `Connection` was opened against.
///
/// This is needed because, unlike the other backends, a single `glutin`-backed build of
/// `surfman` can support several native windowing systems (X11 and Wayland on Unix, for
/// instance), so we must remember which one we actually connected to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum WindowingSystem {
    #[cfg(x11_platform)]
    Xlib,
    #[cfg(free_unix)]
    Wayland,
    #[cfg(windows_platform)]
    Win32,
    #[cfg(macos_platform)]
    AppKit,
}

/// A connection to the display server.
#[derive(Clone)]
pub struct Connection {
    pub(crate) native_connection: NativeConnection,
}

/// The display wrapped by this connection.
#[derive(Clone)]
pub struct NativeConnection {
    /// The underlying `glutin` display.
    pub display: Display,
    pub(crate) windowing_system: WindowingSystem,
}

unsafe impl Send for Connection {}

impl Connection {
    /// Connects to the default display.
    pub fn new() -> Result<Connection, Error> {
        #[cfg(free_unix)]
        {
            if let Ok(connection) = Self::new_wayland() {
                return Ok(connection);
            }
            #[cfg(x11_platform)]
            {
                return Self::new_x11();
            }
            #[cfg(not(x11_platform))]
            {
                return Err(Error::ConnectionFailed);
            }
        }
        #[cfg(windows_platform)]
        {
            Self::new_windows()
        }
        #[cfg(macos_platform)]
        {
            Self::new_macos()
        }
        #[cfg(not(any(free_unix, windows_platform, macos_platform)))]
        {
            Err(Error::ConnectionFailed)
        }
    }

    /// Wraps an existing `glutin` display in a `Connection`.
    ///
    /// # Safety
    ///
    /// The display must be valid for as long as the `Connection` (and anything created from it)
    /// remains alive.
    pub unsafe fn from_native_connection(
        native_connection: NativeConnection,
    ) -> Result<Connection, Error> {
        Ok(Connection { native_connection })
    }

    /// Returns the underlying native connection.
    #[inline]
    pub fn native_connection(&self) -> NativeConnection {
        self.native_connection.clone()
    }

    /// Returns the OpenGL API flavor that this connection supports (OpenGL or OpenGL ES).
    #[inline]
    pub fn gl_api(&self) -> GLApi {
        if std::env::var("SURFMAN_FORCE_GLES").is_ok() {
            GLApi::GLES
        } else {
            GLApi::GL
        }
    }

    /// Returns the "best" adapter on this system, preferring high-performance hardware adapters.
    ///
    /// This is an alias for `Connection::create_hardware_adapter()`.
    #[inline]
    pub fn create_adapter(&self) -> Result<Adapter, Error> {
        self.create_hardware_adapter()
    }

    /// Returns the "best" adapter on this system, preferring high-performance hardware adapters.
    #[inline]
    pub fn create_hardware_adapter(&self) -> Result<Adapter, Error> {
        Ok(Adapter::hardware())
    }

    /// Returns the "best" adapter on this system, preferring low-power hardware adapters.
    #[inline]
    pub fn create_low_power_adapter(&self) -> Result<Adapter, Error> {
        Ok(Adapter::low_power())
    }

    /// Returns the "best" adapter on this system, preferring software adapters.
    #[inline]
    pub fn create_software_adapter(&self) -> Result<Adapter, Error> {
        Ok(Adapter::software())
    }

    /// Opens the hardware device corresponding to the given adapter.
    #[inline]
    pub fn create_device(&self, adapter: &Adapter) -> Result<Device, Error> {
        Device::new(self, adapter)
    }

    /// Opens the hardware device corresponding to the adapter wrapped in the given native
    /// device.
    #[inline]
    pub unsafe fn create_device_from_native_device(
        &self,
        native_device: NativeDevice,
    ) -> Result<Device, Error> {
        Device::new(self, &native_device.adapter)
    }

    /// Opens the display connection corresponding to the given `RawDisplayHandle`.
    #[cfg(feature = "sm-raw-window-handle-05")]
    pub fn from_raw_display_handle(
        raw_handle: rwh_05::RawDisplayHandle,
    ) -> Result<Connection, Error> {
        let raw_display_handle = rwh05_display_to_rwh06(raw_handle)?;
        Self::from_raw_display_handle_06(raw_display_handle)
    }

    /// Opens the display connection corresponding to the given `DisplayHandle`.
    #[cfg(feature = "sm-raw-window-handle-06")]
    pub fn from_display_handle(handle: rwh_06::DisplayHandle) -> Result<Connection, Error> {
        Self::from_raw_display_handle_06(handle.as_raw())
    }

    #[allow(dead_code)]
    fn from_raw_display_handle_06(
        raw_display_handle: RawDisplayHandle,
    ) -> Result<Connection, Error> {
        let (windowing_system, preference) = match raw_display_handle {
            #[cfg(x11_platform)]
            RawDisplayHandle::Xlib(_) => (WindowingSystem::Xlib, DisplayApiPreference::Egl),
            #[cfg(free_unix)]
            RawDisplayHandle::Wayland(_) => (WindowingSystem::Wayland, DisplayApiPreference::Egl),
            #[cfg(windows_platform)]
            RawDisplayHandle::Windows(_) => (WindowingSystem::Win32, DisplayApiPreference::Egl),
            #[cfg(macos_platform)]
            RawDisplayHandle::AppKit(_) => (WindowingSystem::AppKit, DisplayApiPreference::Cgl),
            _ => return Err(Error::IncompatibleRawDisplayHandle),
        };
        let display = unsafe { Display::new(raw_display_handle, preference) }
            .map_err(|_| Error::ConnectionFailed)?;
        Ok(Connection {
            native_connection: NativeConnection {
                display,
                windowing_system,
            },
        })
    }

    #[cfg(free_unix)]
    fn new_wayland() -> Result<Connection, Error> {
        unsafe {
            let wayland_display = (wayland_client_handle().wl_display_connect)(std::ptr::null());
            let wayland_display = NonNull::new(wayland_display).ok_or(Error::ConnectionFailed)?;
            let raw_display_handle = RawDisplayHandle::Wayland(rwh_06::WaylandDisplayHandle::new(
                wayland_display.cast(),
            ));
            Self::from_raw_display_handle_06(raw_display_handle)
        }
    }

    #[cfg(x11_platform)]
    fn new_x11() -> Result<Connection, Error> {
        unsafe {
            let xlib = Xlib::open().map_err(|_| Error::ConnectionFailed)?;
            let x11_display = (xlib.XOpenDisplay)(std::ptr::null());
            let x11_display = NonNull::new(x11_display).ok_or(Error::ConnectionFailed)?;
            let raw_display_handle =
                RawDisplayHandle::Xlib(rwh_06::XlibDisplayHandle::new(Some(x11_display.cast()), 0));
            Self::from_raw_display_handle_06(raw_display_handle)
        }
    }

    #[cfg(windows_platform)]
    fn new_windows() -> Result<Connection, Error> {
        let raw_display_handle = RawDisplayHandle::Windows(rwh_06::WindowsDisplayHandle::new());
        Self::from_raw_display_handle_06(raw_display_handle)
    }

    #[cfg(macos_platform)]
    fn new_macos() -> Result<Connection, Error> {
        let raw_display_handle = RawDisplayHandle::AppKit(rwh_06::AppKitDisplayHandle::new());
        Self::from_raw_display_handle_06(raw_display_handle)
    }

    /// Creates a native widget from a raw pointer.
    ///
    /// # Safety
    ///
    /// The pointer must correspond to a valid native window handle for the windowing system
    /// this connection is using (an X11 `Window`, a Wayland `wl_surface`, an `HWND`, or an
    /// `NSView`, depending on platform).
    pub unsafe fn create_native_widget_from_ptr(
        &self,
        raw: *mut c_void,
        size: Size2D<i32>,
    ) -> NativeWidget {
        let raw_window_handle = match self.native_connection.windowing_system {
            #[cfg(x11_platform)]
            WindowingSystem::Xlib => {
                rwh_06::RawWindowHandle::Xlib(rwh_06::XlibWindowHandle::new(raw as usize as u64))
            }
            #[cfg(free_unix)]
            WindowingSystem::Wayland => rwh_06::RawWindowHandle::Wayland(
                rwh_06::WaylandWindowHandle::new(NonNull::new(raw).expect("null wl_surface")),
            ),
            #[cfg(windows_platform)]
            WindowingSystem::Win32 => {
                rwh_06::RawWindowHandle::Win32(rwh_06::Win32WindowHandle::new(
                    std::num::NonZeroIsize::new(raw as isize).expect("null HWND"),
                ))
            }
            #[cfg(macos_platform)]
            WindowingSystem::AppKit => rwh_06::RawWindowHandle::AppKit(
                rwh_06::AppKitWindowHandle::new(NonNull::new(raw).expect("null NSView")),
            ),
        };
        NativeWidget {
            raw_window_handle,
            size,
        }
    }

    /// Create a native widget type from the given `RawWindowHandle`.
    #[cfg(feature = "sm-raw-window-handle-05")]
    pub fn create_native_widget_from_raw_window_handle(
        &self,
        window: rwh_05::RawWindowHandle,
        size: Size2D<i32>,
    ) -> Result<NativeWidget, Error> {
        let raw_window_handle = rwh05_window_to_rwh06(window)?;
        Ok(NativeWidget {
            raw_window_handle,
            size,
        })
    }

    /// Create a native widget type from the given `WindowHandle`.
    #[cfg(feature = "sm-raw-window-handle-06")]
    pub fn create_native_widget_from_window_handle(
        &self,
        window: rwh_06::WindowHandle,
        size: Size2D<i32>,
    ) -> Result<NativeWidget, Error> {
        Ok(NativeWidget {
            raw_window_handle: window.as_raw(),
            size,
        })
    }
}

/// Converts a `raw-window-handle` 0.5 display handle into the 0.6 equivalent that `glutin`
/// expects.
#[cfg(feature = "sm-raw-window-handle-05")]
fn rwh05_display_to_rwh06(handle: rwh_05::RawDisplayHandle) -> Result<RawDisplayHandle, Error> {
    match handle {
        #[cfg(x11_platform)]
        rwh_05::RawDisplayHandle::Xlib(handle) => Ok(RawDisplayHandle::Xlib(
            rwh_06::XlibDisplayHandle::new(NonNull::new(handle.display), handle.screen),
        )),
        #[cfg(free_unix)]
        rwh_05::RawDisplayHandle::Wayland(handle) => Ok(RawDisplayHandle::Wayland(
            rwh_06::WaylandDisplayHandle::new(
                NonNull::new(handle.display).ok_or(Error::IncompatibleRawDisplayHandle)?,
            ),
        )),
        #[cfg(windows_platform)]
        rwh_05::RawDisplayHandle::Windows(_) => Ok(RawDisplayHandle::Windows(
            rwh_06::WindowsDisplayHandle::new(),
        )),
        #[cfg(macos_platform)]
        rwh_05::RawDisplayHandle::AppKit(_) => {
            Ok(RawDisplayHandle::AppKit(rwh_06::AppKitDisplayHandle::new()))
        }
        _ => Err(Error::IncompatibleRawDisplayHandle),
    }
}

/// Converts a `raw-window-handle` 0.5 window handle into the 0.6 equivalent that `glutin`
/// expects.
#[cfg(feature = "sm-raw-window-handle-05")]
fn rwh05_window_to_rwh06(
    handle: rwh_05::RawWindowHandle,
) -> Result<rwh_06::RawWindowHandle, Error> {
    match handle {
        #[cfg(x11_platform)]
        rwh_05::RawWindowHandle::Xlib(handle) => Ok(rwh_06::RawWindowHandle::Xlib(
            rwh_06::XlibWindowHandle::new(handle.window),
        )),
        #[cfg(free_unix)]
        rwh_05::RawWindowHandle::Wayland(handle) => Ok(rwh_06::RawWindowHandle::Wayland(
            rwh_06::WaylandWindowHandle::new(
                NonNull::new(handle.surface).ok_or(Error::InvalidNativeWidget)?,
            ),
        )),
        #[cfg(windows_platform)]
        rwh_05::RawWindowHandle::Win32(handle) => Ok(rwh_06::RawWindowHandle::Win32(
            rwh_06::Win32WindowHandle::new(
                std::num::NonZeroIsize::new(handle.hwnd as isize)
                    .ok_or(Error::InvalidNativeWidget)?,
            ),
        )),
        #[cfg(macos_platform)]
        rwh_05::RawWindowHandle::AppKit(handle) => Ok(rwh_06::RawWindowHandle::AppKit(
            rwh_06::AppKitWindowHandle::new(
                NonNull::new(handle.ns_view).ok_or(Error::InvalidNativeWidget)?,
            ),
        )),
        _ => Err(Error::InvalidNativeWidget),
    }
}
