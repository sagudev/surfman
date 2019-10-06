// surfman/src/renderbuffers.rs

//! A utility module for backends that wrap surfaces in OpenGL FBOs.
//!
//! FIXME(pcwalton): This can only really be used by one backend at a
//! time, thanks to the use of `GL_FUNCTIONS`. This works for now, but
//! it will need to be revisited if a platform can ever have two
//! backends that need renderbuffers.

use crate::context::{ContextAttributeFlags, ContextAttributes};
use crate::gl::types::GLuint;
use crate::gl;
use crate::platform::default::context::GL_FUNCTIONS;

use euclid::default::Size2D;

pub(crate) enum Renderbuffers {
    IndividualDepthStencil {
        depth: GLuint,
        stencil: GLuint,
    },
    CombinedDepthStencil(GLuint),
}

impl Drop for Renderbuffers {
    fn drop(&mut self) {
        match *self {
            Renderbuffers::IndividualDepthStencil { depth: 0, stencil: 0 } |
            Renderbuffers::CombinedDepthStencil(0) => {}
            _ => panic!("Should have destroyed the FBO renderbuffers with `destroy()`!"),
        }
    }
}

impl Renderbuffers {
    pub(crate) fn new(size: &Size2D<i32>, attributes: &ContextAttributes) -> Renderbuffers {
        GL_FUNCTIONS.with(|gl| {
            unsafe {
                if attributes.flags.contains(ContextAttributeFlags::DEPTH |
                                            ContextAttributeFlags::STENCIL) {
                    let mut renderbuffer = 0;
                    gl.GenRenderbuffers(1, &mut renderbuffer);
                    gl.BindRenderbuffer(gl::RENDERBUFFER, renderbuffer);
                    gl.RenderbufferStorage(gl::RENDERBUFFER,
                                           gl::DEPTH24_STENCIL8,
                                           size.width,
                                           size.height);
                    gl.BindRenderbuffer(gl::RENDERBUFFER, 0);
                    return Renderbuffers::CombinedDepthStencil(renderbuffer);
                }

                let (mut depth_renderbuffer, mut stencil_renderbuffer) = (0, 0);
                if attributes.flags.contains(ContextAttributeFlags::DEPTH) {
                    gl.GenRenderbuffers(1, &mut depth_renderbuffer);
                    gl.BindRenderbuffer(gl::RENDERBUFFER, depth_renderbuffer);
                    gl.RenderbufferStorage(gl::RENDERBUFFER,
                                           gl::DEPTH_COMPONENT24,
                                           size.width,
                                           size.height);
                }
                if attributes.flags.contains(ContextAttributeFlags::STENCIL) {
                    gl.GenRenderbuffers(1, &mut stencil_renderbuffer);
                    gl.BindRenderbuffer(gl::RENDERBUFFER, stencil_renderbuffer);
                    gl.RenderbufferStorage(gl::RENDERBUFFER,
                                           gl::STENCIL_INDEX8,
                                           size.width,
                                           size.height);
                }
                gl.BindRenderbuffer(gl::RENDERBUFFER, 0);

                Renderbuffers::IndividualDepthStencil {
                    depth: depth_renderbuffer,
                    stencil: stencil_renderbuffer,
                }
            }
        })
    }

    pub(crate) fn bind_to_current_framebuffer(&self) {
        GL_FUNCTIONS.with(|gl| {
            unsafe {
                match *self {
                    Renderbuffers::CombinedDepthStencil(renderbuffer) => {
                        if renderbuffer != 0 {
                            gl.FramebufferRenderbuffer(gl::FRAMEBUFFER,
                                                       gl::DEPTH_STENCIL_ATTACHMENT,
                                                       gl::RENDERBUFFER,
                                                       renderbuffer);
                        }
                    }
                    Renderbuffers::IndividualDepthStencil {
                        depth: depth_renderbuffer,
                        stencil: stencil_renderbuffer,
                    } => {
                        if depth_renderbuffer != 0 {
                            gl.FramebufferRenderbuffer(gl::FRAMEBUFFER,
                                                       gl::DEPTH_ATTACHMENT,
                                                       gl::RENDERBUFFER,
                                                       depth_renderbuffer);
                        }
                        if stencil_renderbuffer != 0 {
                            gl.FramebufferRenderbuffer(gl::FRAMEBUFFER,
                                                       gl::STENCIL_ATTACHMENT,
                                                       gl::RENDERBUFFER,
                                                       stencil_renderbuffer);
                        }
                    }
                }
            }
        })
    }

    pub(crate) fn destroy(&mut self) {
        GL_FUNCTIONS.with(|gl| {
            unsafe {
                gl.BindRenderbuffer(gl::RENDERBUFFER, 0);
                match *self {
                    Renderbuffers::CombinedDepthStencil(ref mut renderbuffer) => {
                        if *renderbuffer != 0 {
                            gl.DeleteRenderbuffers(1, renderbuffer);
                            *renderbuffer = 0;
                        }
                    }
                    Renderbuffers::IndividualDepthStencil {
                        depth: ref mut depth_renderbuffer,
                        stencil: ref mut stencil_renderbuffer,
                    } => {
                        if *stencil_renderbuffer != 0 {
                            gl.DeleteRenderbuffers(1, stencil_renderbuffer);
                            *stencil_renderbuffer = 0;
                        }
                        if *depth_renderbuffer != 0 {
                            gl.DeleteRenderbuffers(1, depth_renderbuffer);
                            *depth_renderbuffer = 0;
                        }
                    }
                }
            }
        })
    }

    #[allow(dead_code)]
    pub(crate) fn leak(&mut self) {
        match *self {
            Renderbuffers::CombinedDepthStencil(ref mut renderbuffer) => *renderbuffer = 0,
            Renderbuffers::IndividualDepthStencil {
                depth: ref mut depth_renderbuffer,
                stencil: ref mut stencil_renderbuffer,
            } => {
                *stencil_renderbuffer = 0;
                *depth_renderbuffer = 0;
            }
        }
    }
}
