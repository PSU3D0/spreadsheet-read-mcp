//! wasm32 size probe. One exported function per build so nothing is dead-stripped.

#[cfg(not(feature = "baseline"))]
mod probe {
    use spike_d::scene::{Cmd, HAlign, Rect, Rgba, Scene, VAlign};

    /// Build a tiny scene, shape and raster it, return the PNG length.
    /// Touches: font bytes, the shaper, tiny-skia fill/stroke, png encode.
    #[unsafe(no_mangle)]
    pub extern "C" fn probe_render(size_px: f32) -> u32 {
        let fonts = spike_d::Fonts::new();
        let mut scene = Scene {
            width: 320.0,
            height: 64.0,
            ..Scene::default()
        };
        scene.cmds.push(Cmd::FillRect {
            rect: Rect { x: 0.0, y: 0.0, w: 320.0, h: 64.0 },
            color: Rgba::WHITE,
        });
        scene.cmds.push(Cmd::Line {
            x1: 0.0, y1: 32.0, x2: 320.0, y2: 32.0,
            width: 1.0, color: Rgba::GRIDLINE, dash: spike_d::scene::Dash::Dashed,
        });
        scene.cmds.push(Cmd::Text {
            rect: Rect { x: 2.0, y: 2.0, w: 316.0, h: 60.0 },
            text: "Revenue \u{20ac}1,234.56 \u{2014} AVg\u{fb01}".to_string(),
            size_px,
            color: Rgba::BLACK,
            bold: true,
            italic: true,
            underline: true,
            strike: false,
            halign: HAlign::Center,
            valign: VAlign::Center,
            wrap: true,
            clip: Rect { x: 0.0, y: 0.0, w: 320.0, h: 64.0 },
        });
        match spike_d::render_scene_png(&scene, &fonts) {
            Ok(v) => v.len() as u32,
            Err(_) => 0,
        }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn probe_font_bytes() -> u32 {
        spike_d::embedded_font_bytes() as u32
    }
}

#[cfg(feature = "baseline")]
#[unsafe(no_mangle)]
pub extern "C" fn probe_render(size_px: f32) -> u32 {
    size_px as u32
}
