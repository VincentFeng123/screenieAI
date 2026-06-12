//! Geometry for the perception layer.
//!
//! Contract (C5): every frame in this module and everything downstream of it
//! is in **global screen points, top-left origin** — the same space AX
//! reports positions in and the same space CGEvent posts clicks in. The one
//! and only points→pixels conversion lives in [`point_to_pixel`], used by the
//! set-of-marks annotation renderer. Nothing here may route through NSScreen
//! frames (bottom-left origin).

use serde::{Deserialize, Serialize};

/// A rectangle in global screen points, top-left origin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RectPt {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl RectPt {
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self { x, y, w, h }
    }

    /// Degenerate frames are pruned by the walker and never marked.
    pub fn is_degenerate(&self) -> bool {
        !(self.w > 0.0 && self.h > 0.0 && self.x.is_finite() && self.y.is_finite())
    }

    pub fn intersects(&self, other: &RectPt) -> bool {
        self.x < other.x + other.w
            && other.x < self.x + self.w
            && self.y < other.y + other.h
            && other.y < self.y + self.h
    }

    pub fn center(&self) -> (f64, f64) {
        (self.x + self.w / 2.0, self.y + self.h / 2.0)
    }

    pub fn area(&self) -> f64 {
        if self.is_degenerate() {
            0.0
        } else {
            self.w * self.h
        }
    }

    pub fn contains_point(&self, x: f64, y: f64) -> bool {
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }
}

/// Capture-space metadata a snapshot carries so the annotation renderer can
/// map element frames onto the captured bitmap. `origin` is the capture
/// rect's top-left in global points (negative on displays left of / above
/// the main display), `scale` is the display's backing scale factor, and
/// `downscale` is the extra long-edge shrink applied to the model-facing
/// image (1.0 = none).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaptureSpace {
    pub origin: (f64, f64),
    pub scale: f64,
    pub downscale: f64,
}

/// The ONLY points→pixels transform in the perception layer (C5):
/// `px = (pt − cap_origin) × scale × downscale_ratio`.
pub fn point_to_pixel(space: &CaptureSpace, x_pt: f64, y_pt: f64) -> (f64, f64) {
    (
        (x_pt - space.origin.0) * space.scale * space.downscale,
        (y_pt - space.origin.1) * space.scale * space.downscale,
    )
}

/// Rect variant of [`point_to_pixel`]; width/height scale without the origin
/// shift.
pub fn rect_to_pixel(space: &CaptureSpace, rect: &RectPt) -> (f64, f64, f64, f64) {
    let (x, y) = point_to_pixel(space, rect.x, rect.y);
    let k = space.scale * space.downscale;
    (x, y, rect.w * k, rect.h * k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_at_scale_1() {
        let space = CaptureSpace {
            origin: (0.0, 0.0),
            scale: 1.0,
            downscale: 1.0,
        };
        assert_eq!(point_to_pixel(&space, 100.0, 50.0), (100.0, 50.0));
    }

    #[test]
    fn transform_at_scale_2_with_downscale() {
        let space = CaptureSpace {
            origin: (0.0, 0.0),
            scale: 2.0,
            downscale: 0.5,
        };
        // Retina capture downscaled by half lands back on point values.
        assert_eq!(point_to_pixel(&space, 100.0, 50.0), (100.0, 50.0));
    }

    #[test]
    fn transform_negative_multi_display_origin() {
        // A display left of and above the main display has a negative global
        // origin; a point inside it must land at positive pixel coordinates
        // within its own capture.
        let space = CaptureSpace {
            origin: (-1512.0, -200.0),
            scale: 2.0,
            downscale: 1.0,
        };
        assert_eq!(point_to_pixel(&space, -1412.0, -100.0), (200.0, 200.0));
        let (x, y, w, h) = rect_to_pixel(&space, &RectPt::new(-1512.0, -200.0, 10.0, 20.0));
        assert_eq!((x, y, w, h), (0.0, 0.0, 20.0, 40.0));
    }

    #[test]
    fn degenerate_and_intersection() {
        assert!(RectPt::new(0.0, 0.0, 0.0, 10.0).is_degenerate());
        assert!(RectPt::new(0.0, 0.0, f64::NAN, 10.0).is_degenerate());
        assert!(!RectPt::new(0.0, 0.0, 1.0, 1.0).is_degenerate());
        let a = RectPt::new(0.0, 0.0, 100.0, 100.0);
        assert!(a.intersects(&RectPt::new(99.0, 99.0, 10.0, 10.0)));
        assert!(!a.intersects(&RectPt::new(100.0, 0.0, 10.0, 10.0)));
    }
}
