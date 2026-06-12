//! Set-of-marks renderer behind `look`: numbered boxes over interactive
//! elements, every label a valid eid from the same snapshot.
//!
//! This file is the only consumer of the points→pixels transform (C5). The
//! model-facing image is downscaled to a configurable long edge (default
//! 1456px); the full-res raw capture stays alongside on disk.

use crate::perception::ax::classify::ElementClass;
use crate::perception::geometry::{rect_to_pixel, CaptureSpace, RectPt};
use crate::perception::ids::ElementRow;
use crate::perception::PerceptionError;
use ab_glyph::{FontRef, PxScale};
use image::{Rgba, RgbaImage};
use serde::Serialize;

/// Long edge of the model-facing marked image.
pub const DEFAULT_MARKED_LONG_EDGE: u32 = 1456;
/// Most marks ever drawn, selected by salience.
const MARK_CAP: usize = 150;
/// Elements thinner than this (points, either axis) are never marked.
const MIN_MARK_PT: f64 = 12.0;
const BOX_STROKE_PX: i32 = 2;
const CHIP_TEXT_HEIGHT_PX: f32 = 13.0;
const CHIP_PAD_X: i32 = 3;
const CHIP_PAD_Y: i32 = 1;
/// Legend labels truncate here (chars).
const LEGEND_LABEL_CHARS: usize = 40;

static CHIP_FONT: &[u8] = include_bytes!("assets/chip-font.ttf");

/// Fixed 8-color palette, one per markable class, picked for contrast on
/// both light and dark UIs. `dark_text` flags the one light chip color.
fn class_color(class: ElementClass) -> (Rgba<u8>, bool) {
    match class {
        ElementClass::Button => (Rgba([229, 72, 77, 255]), false), // red
        ElementClass::TextInput => (Rgba([0, 144, 255, 255]), false), // blue
        ElementClass::Link => (Rgba([142, 78, 198, 255]), false),  // purple
        ElementClass::Toggle => (Rgba([247, 107, 21, 255]), false), // orange
        ElementClass::MenuItem => (Rgba([18, 165, 148, 255]), false), // teal
        ElementClass::Slider => (Rgba([255, 197, 61, 255]), true), // amber
        ElementClass::Image => (Rgba([61, 214, 140, 255]), true),  // green
        // Generic and (never marked) Static share gray.
        ElementClass::Generic | ElementClass::Static => (Rgba([110, 110, 110, 255]), false),
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LegendEntry {
    pub eid: String,
    /// title → descr → role, first non-empty, truncated.
    pub label: String,
}

#[derive(Debug)]
pub struct Annotation {
    pub png: Vec<u8>,
    pub legend: Vec<LegendEntry>,
    pub width: u32,
    pub height: u32,
    /// Extra shrink applied on top of the capture's scale factor.
    pub downscale: f64,
    pub marks_drawn: usize,
    /// Markable elements that lost their slot to the cap.
    pub marks_capped: usize,
}

/// Salience order for the mark cap: focused first, then actionable by class
/// priority, then larger visible area. Returns indices into `rows`.
fn select_marks(rows: &[ElementRow]) -> (Vec<usize>, usize) {
    let mut candidates: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| {
            row.class_prefix != 'X'
                && !row.frame.is_degenerate()
                && row.frame.w >= MIN_MARK_PT
                && row.frame.h >= MIN_MARK_PT
        })
        .map(|(index, _)| index)
        .collect();
    candidates.sort_by(|&a, &b| {
        let (ra, rb) = (&rows[a], &rows[b]);
        let class_rank = |row: &ElementRow| {
            ElementClass::from_prefix(row.class_prefix)
                .map(|class| class.salience_rank())
                .unwrap_or(u8::MAX)
        };
        (
            !ra.focused,
            !ra.actionable,
            class_rank(ra),
            std::cmp::Reverse(ra.frame.area() as i64),
        )
            .cmp(&(
                !rb.focused,
                !rb.actionable,
                class_rank(rb),
                std::cmp::Reverse(rb.frame.area() as i64),
            ))
    });
    let capped = candidates.len().saturating_sub(MARK_CAP);
    candidates.truncate(MARK_CAP);
    (candidates, capped)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PxRect {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
}

impl PxRect {
    fn intersects(&self, other: &PxRect) -> bool {
        self.x < other.x + other.w
            && other.x < self.x + self.w
            && self.y < other.y + other.h
            && other.y < self.y + self.h
    }
}

/// Greedy chip placement: TL → TR → BL → BR → inside-center; `None` (drop
/// the chip, keep the box) when every candidate collides with an existing
/// chip. Chips are clamped into the image; clamping may itself collide.
fn place_chip(
    chip_w: i32,
    chip_h: i32,
    bounds: PxRect,
    occupied: &[PxRect],
    img_w: i32,
    img_h: i32,
) -> Option<PxRect> {
    let candidates = [
        (bounds.x, bounds.y),                                         // TL
        (bounds.x + bounds.w - chip_w, bounds.y),                     // TR
        (bounds.x, bounds.y + bounds.h - chip_h),                     // BL
        (bounds.x + bounds.w - chip_w, bounds.y + bounds.h - chip_h), // BR
        (
            bounds.x + (bounds.w - chip_w) / 2,
            bounds.y + (bounds.h - chip_h) / 2,
        ), // inside-center
    ];
    for (x, y) in candidates {
        let chip = PxRect {
            x: x.clamp(0, (img_w - chip_w).max(0)),
            y: y.clamp(0, (img_h - chip_h).max(0)),
            w: chip_w,
            h: chip_h,
        };
        if !occupied.iter().any(|other| chip.intersects(other)) {
            return Some(chip);
        }
    }
    None
}

fn legend_label(row: &ElementRow) -> String {
    let raw = row
        .title
        .as_deref()
        .or(row.descr.as_deref())
        .filter(|label| !label.trim().is_empty())
        .unwrap_or(&row.role);
    let mut label: String = raw.chars().take(LEGEND_LABEL_CHARS).collect();
    if raw.chars().count() > LEGEND_LABEL_CHARS {
        label.push('…');
    }
    label
}

/// Render the set-of-marks frame: `raw_png` is the snapshot's capture,
/// `cap_rect` its capture rect in points, `scale` the display's backing
/// factor.
pub fn annotate(
    raw_png: &[u8],
    rows: &[ElementRow],
    cap_rect: RectPt,
    scale: f64,
    long_edge: Option<u32>,
) -> Result<Annotation, PerceptionError> {
    let decoded = image::load_from_memory(raw_png)
        .map_err(|err| PerceptionError::Capture(format!("decode capture: {err}")))?;
    let (src_w, src_h) = (decoded.width(), decoded.height());
    let long_edge = long_edge.unwrap_or(DEFAULT_MARKED_LONG_EDGE).max(64);
    let src_long = src_w.max(src_h);
    let downscale = if src_long > long_edge {
        f64::from(long_edge) / f64::from(src_long)
    } else {
        1.0
    };
    let mut canvas: RgbaImage = if downscale < 1.0 {
        image::imageops::resize(
            &decoded.to_rgba8(),
            (f64::from(src_w) * downscale).round().max(1.0) as u32,
            (f64::from(src_h) * downscale).round().max(1.0) as u32,
            image::imageops::FilterType::Triangle,
        )
    } else {
        decoded.to_rgba8()
    };
    let (img_w, img_h) = (canvas.width() as i32, canvas.height() as i32);

    let space = CaptureSpace {
        origin: (cap_rect.x, cap_rect.y),
        scale,
        downscale,
    };

    let font = FontRef::try_from_slice(CHIP_FONT)
        .map_err(|err| PerceptionError::Capture(format!("chip font: {err}")))?;
    let text_scale = PxScale::from(CHIP_TEXT_HEIGHT_PX);

    let (selected, marks_capped) = select_marks(rows);
    let mut occupied: Vec<PxRect> = Vec::with_capacity(selected.len());
    let mut legend = Vec::with_capacity(selected.len());
    let mut marks_drawn = 0usize;

    for index in selected {
        let row = &rows[index];
        let (x, y, w, h) = rect_to_pixel(&space, &row.frame);
        let bounds = PxRect {
            x: x.round() as i32,
            y: y.round() as i32,
            w: w.round().max(1.0) as i32,
            h: h.round().max(1.0) as i32,
        };
        if bounds.x + bounds.w <= 0
            || bounds.y + bounds.h <= 0
            || bounds.x >= img_w
            || bounds.y >= img_h
        {
            continue;
        }
        let class = ElementClass::from_prefix(row.class_prefix).unwrap_or(ElementClass::Generic);
        let (color, dark_text) = class_color(class);

        draw_box(&mut canvas, bounds, color);
        marks_drawn += 1;
        legend.push(LegendEntry {
            eid: row.eid.clone(),
            label: legend_label(row),
        });

        let (text_w, text_h) = imageproc::drawing::text_size(text_scale, &font, &row.eid);
        let chip_w = text_w as i32 + 2 * CHIP_PAD_X;
        let chip_h = text_h as i32 + 2 * CHIP_PAD_Y;
        let Some(chip) = place_chip(chip_w, chip_h, bounds, &occupied, img_w, img_h) else {
            continue; // chip dropped, box kept
        };
        imageproc::drawing::draw_filled_rect_mut(
            &mut canvas,
            imageproc::rect::Rect::at(chip.x, chip.y).of_size(chip.w as u32, chip.h as u32),
            color,
        );
        let text_color = if dark_text {
            Rgba([0, 0, 0, 255])
        } else {
            Rgba([255, 255, 255, 255])
        };
        imageproc::drawing::draw_text_mut(
            &mut canvas,
            text_color,
            chip.x + CHIP_PAD_X,
            chip.y + CHIP_PAD_Y,
            text_scale,
            &font,
            &row.eid,
        );
        occupied.push(chip);
    }

    let mut png = Vec::new();
    image::write_buffer_with_format(
        &mut std::io::Cursor::new(&mut png),
        &canvas,
        canvas.width(),
        canvas.height(),
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )
    .map_err(|err| PerceptionError::Capture(format!("encode marked png: {err}")))?;

    Ok(Annotation {
        png,
        legend,
        width: canvas.width(),
        height: canvas.height(),
        downscale,
        marks_drawn,
        marks_capped,
    })
}

/// 2px hollow rectangle, clipped to the canvas.
fn draw_box(canvas: &mut RgbaImage, bounds: PxRect, color: Rgba<u8>) {
    for inset in 0..BOX_STROKE_PX {
        let x = bounds.x + inset;
        let y = bounds.y + inset;
        let w = bounds.w - 2 * inset;
        let h = bounds.h - 2 * inset;
        if w <= 0 || h <= 0 {
            break;
        }
        imageproc::drawing::draw_hollow_rect_mut(
            canvas,
            imageproc::rect::Rect::at(x, y).of_size(w as u32, h as u32),
            color,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mark_row(eid: &str, class: char, frame: RectPt) -> ElementRow {
        ElementRow {
            eid: eid.into(),
            fp: 1,
            role: "AXButton".into(),
            subrole: None,
            title: Some(format!("el {eid}")),
            descr: None,
            value: None,
            actions: vec!["AXPress".into()],
            actionable: class != 'X' && class != 'I',
            enabled: true,
            focused: false,
            frame,
            depth: 1,
            parent_eid: None,
            is_web_boundary: false,
            class_prefix: class,
        }
    }

    #[test]
    fn chip_font_covers_eid_alphabet() {
        use ab_glyph::Font;
        let font = FontRef::try_from_slice(CHIP_FONT).unwrap();
        for ch in ('a'..='z')
            .chain('A'..='Z')
            .chain('0'..='9')
            .chain([':', '.'])
        {
            assert_ne!(font.glyph_id(ch).0, 0, "missing glyph for {ch:?}");
        }
    }

    #[test]
    fn salience_selects_focused_then_class_then_area_and_caps() {
        let mut rows = Vec::new();
        // 160 generic marks — more than the cap.
        for index in 0..160 {
            rows.push(mark_row(
                &format!("ax:G{index}"),
                'G',
                RectPt::new(0.0, index as f64 * 20.0, 50.0, 18.0),
            ));
        }
        // A static row and a sub-12pt button: never marked.
        rows.push(mark_row("ax:X1", 'X', RectPt::new(0.0, 0.0, 100.0, 100.0)));
        rows.push(mark_row("ax:B9", 'B', RectPt::new(0.0, 0.0, 11.0, 30.0)));
        // A button (higher class), a bigger button, and a focused field.
        rows.push(mark_row("ax:B1", 'B', RectPt::new(0.0, 0.0, 40.0, 20.0)));
        rows.push(mark_row("ax:B2", 'B', RectPt::new(0.0, 0.0, 400.0, 20.0)));
        let mut focused = mark_row("ax:T1", 'T', RectPt::new(0.0, 0.0, 40.0, 20.0));
        focused.focused = true;
        rows.push(focused);

        let (selected, capped) = select_marks(&rows);
        assert_eq!(selected.len(), MARK_CAP);
        assert_eq!(capped, 163 - MARK_CAP);
        // Focused first, then bigger button before smaller, then generics.
        assert_eq!(rows[selected[0]].eid, "ax:T1");
        assert_eq!(rows[selected[1]].eid, "ax:B2");
        assert_eq!(rows[selected[2]].eid, "ax:B1");
        assert!(!selected.iter().any(|&i| rows[i].eid == "ax:X1"));
        assert!(!selected.iter().any(|&i| rows[i].eid == "ax:B9"));
    }

    #[test]
    fn chip_nudges_through_corners_then_drops() {
        let bounds = PxRect {
            x: 10,
            y: 10,
            w: 100,
            h: 40,
        };
        let mut occupied = Vec::new();
        // First chip lands top-left.
        let first = place_chip(30, 14, bounds, &occupied, 500, 500).unwrap();
        assert_eq!((first.x, first.y), (10, 10));
        occupied.push(first);
        // Second nudges to top-right.
        let second = place_chip(30, 14, bounds, &occupied, 500, 500).unwrap();
        assert_eq!((second.x, second.y), (80, 10));
        occupied.push(second);
        // Third and fourth take the bottom corners, fifth the center.
        occupied.push(place_chip(30, 14, bounds, &occupied, 500, 500).unwrap());
        occupied.push(place_chip(30, 14, bounds, &occupied, 500, 500).unwrap());
        let center = place_chip(30, 14, bounds, &occupied, 500, 500).unwrap();
        assert_eq!((center.x, center.y), (45, 23));
        occupied.push(center);
        // Nothing left: dropped.
        assert!(place_chip(30, 14, bounds, &occupied, 500, 500).is_none());
        // Clamping keeps chips inside a small image.
        let clamped = place_chip(30, 14, bounds, &[], 25, 12).unwrap();
        assert_eq!((clamped.x, clamped.y), (0, 0));
    }

    #[test]
    fn annotate_draws_boxes_chips_and_legend() {
        // White 400x200 capture at scale 2 covering a 200x100pt rect at
        // global origin (50, 60).
        let blank = RgbaImage::from_pixel(400, 200, Rgba([255, 255, 255, 255]));
        let mut raw_png = Vec::new();
        image::write_buffer_with_format(
            &mut std::io::Cursor::new(&mut raw_png),
            &blank,
            400,
            200,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .unwrap();

        let rows = vec![
            mark_row("ax:B1", 'B', RectPt::new(60.0, 70.0, 40.0, 20.0)),
            mark_row("ax:X1", 'X', RectPt::new(50.0, 60.0, 200.0, 100.0)),
        ];
        let result = annotate(
            &raw_png,
            &rows,
            RectPt::new(50.0, 60.0, 200.0, 100.0),
            2.0,
            None,
        )
        .unwrap();
        assert_eq!((result.width, result.height), (400, 200));
        assert_eq!(result.downscale, 1.0);
        assert_eq!(result.marks_drawn, 1);
        assert_eq!(result.legend.len(), 1);
        assert_eq!(result.legend[0].eid, "ax:B1");

        // Box top-left in pixels: ((60-50)*2, (70-60)*2) = (20, 20) — must
        // no longer be white.
        let marked = image::load_from_memory(&result.png).unwrap().to_rgba8();
        assert_ne!(marked.get_pixel(20, 20), &Rgba([255, 255, 255, 255]));
        // Far corner untouched.
        assert_eq!(marked.get_pixel(399, 199), &Rgba([255, 255, 255, 255]));
    }

    #[test]
    fn annotate_downscales_long_edge() {
        let blank = RgbaImage::from_pixel(2912, 1456, Rgba([255, 255, 255, 255]));
        let mut raw_png = Vec::new();
        image::write_buffer_with_format(
            &mut std::io::Cursor::new(&mut raw_png),
            &blank,
            2912,
            1456,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .unwrap();
        let result = annotate(
            &raw_png,
            &[],
            RectPt::new(0.0, 0.0, 1456.0, 728.0),
            2.0,
            None,
        )
        .unwrap();
        assert_eq!((result.width, result.height), (1456, 728));
        assert_eq!(result.downscale, 0.5);
    }
}
