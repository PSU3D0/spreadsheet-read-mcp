//! Spike D: bounded umya Worksheet -> scene -> tiny-skia raster.
//! Throwaway measurement code for ticket 5003. Not product code.

pub mod raster;
pub mod scene;
pub mod text;

#[cfg(feature = "extract")]
pub mod extract;

pub use scene::{Cmd, Rect, Rgba, Scene, Warning};
pub use text::{FONT_STRATEGY, Fonts, STACK, embedded_font_bytes};

/// One-call path used by the harness and by the wasm size probe.
pub fn render_scene_png(scene: &Scene, fonts: &Fonts) -> Result<Vec<u8>, raster::RenderError> {
    let pix = raster::render(scene, fonts)?;
    raster::to_png(&pix)
}
