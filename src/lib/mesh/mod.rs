mod simple_color;
pub use self::simple_color::*;

use std::path::PathBuf;
use super::ExampleBase;
use super::device::AshDevice;

pub trait Mesh: Sized {
    fn from_gltf<P: Into<PathBuf> + Clone>(base: &ExampleBase, path: P) -> Option<Self>;

    unsafe fn free(self, &AshDevice);
}