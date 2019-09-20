//! Platform-specific backends.

pub mod generic;
#[cfg(feature = "sm-osmesa")]
pub use generic::osmesa as default;

#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(all(target_os = "macos", not(any(feature = "sm-x11", feature = "sm-osmesa"))))]
pub use macos as default;

#[cfg(unix)]
pub mod unix;
#[cfg(all(any(feature = "sm-x11", all(unix, not(any(target_os = "macos", target_os = "android")))),
          not(feature = "sm-osmesa")))]
pub use unix::x11 as default;

#[cfg(target_os = "windows")]
pub mod windows;
#[cfg(all(target_os = "windows", not(feature = "sm-osmesa")))]
pub use windows::angle as default;
