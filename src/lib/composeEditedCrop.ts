import type {
  Stroke,
  MarkerStroke,
  EraserStroke,
  ShapeStroke,
  TextStroke,
} from "./editTypes";

/// Composite the user's strokes onto the cropped PNG and return a fresh
/// base64-encoded PNG (no `data:` prefix — matches the shape `crop_capture`
/// returns from Rust).
///
/// Strokes live in image-relative pixels, so we draw them directly onto a
/// canvas sized to `dims.width × dims.height` after blitting the source PNG.
/// Eraser strokes use `destination-out` on an annotation-only layer, then the
/// layer is drawn over the original image. This matches the live editor:
/// erasing removes only marks and never smears/restores a rectangular patch of
/// the screenshot over nearby annotations.
///
/// Order of operations:
///   1. Draw the source image once.
///   2. Draw all strokes in order onto a transparent annotation layer.
///   3. Composite that annotation layer over the source image.
export async function composeEditedCrop(
  croppedB64: string,
  dims: { width: number; height: number },
  strokes: Stroke[],
): Promise<string> {
  if (strokes.length === 0) return croppedB64;

  const img = await loadImage(`data:image/png;base64,${croppedB64}`);

  const canvas = document.createElement("canvas");
  canvas.width = dims.width;
  canvas.height = dims.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("composeEditedCrop: could not get 2d context");
  }

  ctx.drawImage(img, 0, 0, dims.width, dims.height);

  const marks = document.createElement("canvas");
  marks.width = dims.width;
  marks.height = dims.height;
  const marksCtx = marks.getContext("2d");
  if (!marksCtx) {
    throw new Error("composeEditedCrop: could not get annotation 2d context");
  }

  for (const stroke of strokes) {
    drawStroke(marksCtx, stroke);
  }
  ctx.drawImage(marks, 0, 0);

  const blob = await canvasToBlob(canvas);
  if (!blob) throw new Error("composeEditedCrop: canvas.toBlob returned null");
  const dataUrl = await blobToDataUrl(blob);
  // Strip the `data:image/png;base64,` prefix to match the existing
  // CroppedCapture.png_base64 shape produced by Rust.
  const comma = dataUrl.indexOf(",");
  return comma === -1 ? dataUrl : dataUrl.slice(comma + 1);
}

function drawStroke(
  ctx: CanvasRenderingContext2D,
  stroke: Stroke,
): void {
  switch (stroke.kind) {
    case "marker":
      drawMarker(ctx, stroke);
      break;
    case "eraser":
      drawEraser(ctx, stroke);
      break;
    case "rect":
    case "ellipse":
    case "line":
    case "arrow":
      drawShape(ctx, stroke);
      break;
    case "text":
      drawText(ctx, stroke);
      break;
  }
}

function applyStrokeStyle(
  ctx: CanvasRenderingContext2D,
  style: { color: string; width: number; opacity?: number },
): void {
  ctx.strokeStyle = style.color;
  ctx.fillStyle = style.color;
  ctx.lineWidth = style.width;
  ctx.globalAlpha = style.opacity ?? 1;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

function drawMarker(ctx: CanvasRenderingContext2D, stroke: MarkerStroke): void {
  if (stroke.pts.length === 0) return;
  ctx.save();
  applyStrokeStyle(ctx, stroke.style);
  ctx.beginPath();
  const [x0, y0] = stroke.pts[0];
  ctx.moveTo(x0, y0);
  if (stroke.pts.length === 1) {
    // A single-tap mark — draw a filled disc so it's visible.
    ctx.arc(x0, y0, Math.max(1, stroke.style.width / 2), 0, Math.PI * 2);
    ctx.fill();
  } else {
    for (let i = 1; i < stroke.pts.length; i++) {
      const [x, y] = stroke.pts[i];
      ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function drawEraser(
  ctx: CanvasRenderingContext2D,
  stroke: EraserStroke,
): void {
  if (stroke.pts.length === 0) return;
  ctx.save();
  ctx.globalCompositeOperation = "destination-out";
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "rgba(0,0,0,1)";
  ctx.fillStyle = "rgba(0,0,0,1)";
  ctx.lineWidth = stroke.width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  const [ex0, ey0] = stroke.pts[0];
  ctx.moveTo(ex0, ey0);
  if (stroke.pts.length === 1) {
    ctx.arc(ex0, ey0, stroke.width / 2, 0, Math.PI * 2);
    ctx.fill();
  } else {
    for (let i = 1; i < stroke.pts.length; i++) {
      const [x, y] = stroke.pts[i];
      ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function drawShape(ctx: CanvasRenderingContext2D, stroke: ShapeStroke): void {
  ctx.save();
  applyStrokeStyle(ctx, stroke.style);
  const [ax, ay] = stroke.a;
  const [bx, by] = stroke.b;
  if (stroke.kind === "rect") {
    const x = Math.min(ax, bx);
    const y = Math.min(ay, by);
    const w = Math.abs(bx - ax);
    const h = Math.abs(by - ay);
    ctx.strokeRect(x, y, w, h);
  } else if (stroke.kind === "ellipse") {
    const cx = (ax + bx) / 2;
    const cy = (ay + by) / 2;
    const rx = Math.abs(bx - ax) / 2;
    const ry = Math.abs(by - ay) / 2;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();
  } else if (stroke.kind === "line") {
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  } else if (stroke.kind === "arrow") {
    drawArrow(ctx, ax, ay, bx, by, stroke.style.width);
  }
  ctx.restore();
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  ax: number,
  ay: number,
  bx: number,
  by: number,
  width: number,
): void {
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx, by);
  ctx.stroke();

  const dx = bx - ax;
  const dy = by - ay;
  const len = Math.hypot(dx, dy);
  if (len < 1e-3) return;
  // Arrowhead size scales with stroke width but is bounded so small marks
  // don't get a tiny head and giant marks don't dominate the shape.
  const head = Math.min(Math.max(width * 3.5, 8), 28);
  const angle = Math.atan2(dy, dx);
  const wing = Math.PI / 7; // ~25.7°
  const x1 = bx - head * Math.cos(angle - wing);
  const y1 = by - head * Math.sin(angle - wing);
  const x2 = bx - head * Math.cos(angle + wing);
  const y2 = by - head * Math.sin(angle + wing);
  ctx.beginPath();
  ctx.moveTo(bx, by);
  ctx.lineTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.closePath();
  ctx.fill();
}

function drawText(ctx: CanvasRenderingContext2D, stroke: TextStroke): void {
  if (!stroke.text) return;
  ctx.save();
  ctx.globalAlpha = ctx.globalAlpha * (stroke.style.opacity ?? 1);
  const fontSize = stroke.style.fontSize;
  ctx.font = `600 ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif`;
  ctx.textBaseline = "top";
  // No backing pill — text renders flat with just the user's chosen
  // colour, matching the in-overlay preview.
  ctx.fillStyle = stroke.style.color;
  ctx.fillText(stroke.text, stroke.at[0], stroke.at[1]);
  ctx.restore();
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("composeEditedCrop: image load failed"));
    img.src = src;
  });
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob | null> {
  return new Promise((resolve) => {
    canvas.toBlob((b) => resolve(b), "image/png");
  });
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : "");
    reader.onerror = () => reject(new Error("composeEditedCrop: blob read failed"));
    reader.readAsDataURL(blob);
  });
}

/// Stable hash of a stroke array for memoizing composite output.
/// Uses stroke ids + length so identity is enough for stable history; the
/// in-progress tail isn't part of `strokes` (live strokes live on a separate
/// upper canvas), so a length+last-id key is sufficient.
export function strokesFingerprint(strokes: Stroke[]): string {
  if (strokes.length === 0) return "0";
  const last = strokes[strokes.length - 1];
  return `${strokes.length}:${last.id}`;
}
