//! A smoke test for the `glutin`-backed backend: connects to the display, creates a device and
//! context, creates a generic (offscreen) surface, binds it, renders a solid color into it, and
//! reads the pixels back to make sure everything actually worked end to end.
#![cfg(feature = "sm-glutin")]

use euclid::default::Size2D;
use surfman::glutin::connection::Connection;
use surfman::{ContextAttributeFlags, ContextAttributes, GLVersion, SurfaceAccess, SurfaceType};

#[test]
fn create_context_and_render_to_generic_surface() {
    let connection = Connection::new().expect("failed to connect to the display server");
    let adapter = connection
        .create_adapter()
        .expect("failed to create adapter");
    let device = connection
        .create_device(&adapter)
        .expect("failed to create device");

    let context_attributes = ContextAttributes {
        version: GLVersion::new(3, 0),
        flags: ContextAttributeFlags::ALPHA,
    };
    let context_descriptor = device
        .create_context_descriptor(&context_attributes)
        .expect("failed to create context descriptor");
    let mut context = device
        .create_context(&context_descriptor, None)
        .expect("failed to create context");

    gl::load_with(|symbol| device.get_proc_address(&context, symbol) as *const _);

    let size = Size2D::new(64, 64);
    let surface = device
        .create_surface(
            &context,
            SurfaceAccess::GPUOnly,
            SurfaceType::Generic { size },
        )
        .expect("failed to create surface");
    device
        .bind_surface_to_context(&mut context, surface)
        .expect("failed to bind surface");
    device
        .make_context_current(&context)
        .expect("failed to make context current");

    let surface_info = device
        .context_surface_info(&context)
        .expect("failed to get surface info")
        .expect("no surface bound");
    let framebuffer_object = surface_info
        .framebuffer_object
        .map_or(0, |framebuffer| framebuffer.0.get());

    unsafe {
        gl::BindFramebuffer(gl::FRAMEBUFFER, framebuffer_object);
        gl::ClearColor(1.0, 0.0, 0.0, 1.0);
        gl::Clear(gl::COLOR_BUFFER_BIT);

        let mut pixels = [0u8; 4];
        gl::ReadPixels(
            0,
            0,
            1,
            1,
            gl::RGBA,
            gl::UNSIGNED_BYTE,
            pixels.as_mut_ptr() as *mut _,
        );
        assert_eq!(pixels, [255, 0, 0, 255]);
    }

    let mut surface = device
        .unbind_surface_from_context(&mut context)
        .expect("failed to unbind surface")
        .expect("no surface was bound");
    device
        .destroy_surface(&mut context, &mut surface)
        .expect("failed to destroy surface");
    device
        .destroy_context(&mut context)
        .expect("failed to destroy context");
}
