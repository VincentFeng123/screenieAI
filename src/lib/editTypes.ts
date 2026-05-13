/// Shared types for the on-image edit feature (marker / eraser / text / shapes).
///
/// All stroke coordinates are in *image-relative pixels* — i.e. they live in
/// the cropped image's coordinate space (range [0, cropped.width] ×
/// [0, cropped.height]). Rendering scales by `rect.w / cropped.width` so the
/// strokes follow the on-screen rect through resizes; remapForCrop translates
/// them through screenshot-space when the rect moves to a new region.

export type StrokeColor = string;

export type StrokeStyle = {
  color: StrokeColor;
  width: number;
  opacity?: number;
};

export type ShapeStyle = StrokeStyle & {
  fill?: boolean;
};

export type Tool = "marker" | "eraser" | "text" | "shape" | "colorpicker";

export type Shape = "rect" | "ellipse" | "line" | "arrow";

/// "pixel" — drag erases pixel-level paths on the canvas (standard eraser
///           behaviour: the marks under the eraser path are removed).
/// "object" — drag selects whole strokes; on release every stroke the
///            eraser swept over is deleted from the strokes array.
export type EraserMode = "pixel" | "object";

export type Point2 = readonly [number, number];

export type MarkerStroke = {
  id: string;
  kind: "marker";
  pts: Point2[];
  style: StrokeStyle;
};

export type EraserStroke = {
  id: string;
  kind: "eraser";
  pts: Point2[];
  width: number;
};

export type ShapeStroke = {
  id: string;
  kind: Shape;
  a: Point2;
  b: Point2;
  style: ShapeStyle;
};

export type TextStroke = {
  id: string;
  kind: "text";
  at: Point2;
  text: string;
  style: StrokeStyle & { fontSize: number };
};

export type Stroke = MarkerStroke | EraserStroke | ShapeStroke | TextStroke;

/// Anchor side for the edit affordance/pill placement output.
export type EditAnchorSide =
  | "top"
  | "bottom"
  | "left"
  | "right"
  | "corner-tl"
  | "corner-tr"
  | "corner-bl"
  | "corner-br";

export type EditAnchor = {
  side: EditAnchorSide;
  /// Top-left of the collapsed 32px button, in viewport pixels.
  x: number;
  y: number;
  /// Pill expansion direction when the user clicks the affordance.
  axis: "horizontal" | "vertical";
  /// Whether the pill grows toward viewport-start (left/up) or end (right/down)
  /// from the collapsed-button anchor.
  expandToward: "start" | "end";
  /// Hide flag: rect is too small to host an editor (< 48 on either axis).
  hidden?: boolean;
};

/// Default palette + size scale shared by the marker, text, and shape tools.
export const EDIT_COLORS: StrokeColor[] = [
  "#ffffff",
  "#0a0a0b",
  "#ef4444", // red
  "#f59e0b", // amber
  "#10b981", // emerald
  "#38bdf8", // sky
];

export const EDIT_STROKE_WIDTHS = [2, 4, 6, 10] as const;
export const EDIT_ERASER_WIDTHS = [10, 16, 24, 36] as const;
export const EDIT_TEXT_SIZES = [12, 14, 18, 24] as const;

export const DEFAULT_MARKER_STYLE: StrokeStyle = {
  color: "#ef4444",
  width: 4,
  opacity: 1,
};

export const DEFAULT_SHAPE_STYLE: ShapeStyle = {
  color: "#ef4444",
  width: 3,
  opacity: 1,
  fill: false,
};

export const DEFAULT_TEXT_STYLE: StrokeStyle & { fontSize: number } = {
  color: "#ef4444",
  width: 0,
  opacity: 1,
  fontSize: 18,
};

export const EDIT_AFFORDANCE_SIZE = 36;
export const EDIT_PILL_HORIZONTAL_LEN = 280;
export const EDIT_PILL_VERTICAL_LEN = 240;
