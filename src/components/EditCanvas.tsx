import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import {
  type EraserStroke,
  type MarkerStroke,
  type Shape,
  type ShapeStroke,
  type Stroke,
  type StrokeStyle,
  type TextStroke,
} from "../lib/editTypes";
import { nextStrokeId, type EditController } from "../lib/useEditController";

type Rect = { x: number; y: number; w: number; h: number };

type LiveDraft =
  | { kind: "marker"; pts: Array<[number, number]>; style: StrokeStyle }
  | {
      kind: "eraser";
      pts: Array<[number, number]>;
      width: number;
      /// Pixel mode commits an EraserStroke that punches a hole on render.
      /// Object mode tracks `hitIds` and removes those strokes on release.
      mode: "pixel" | "object";
    }
  | { kind: Shape; a: [number, number]; b: [number, number]; style: StrokeStyle };

export type EditCanvasProps = {
  ctl: EditController;
  /// On-screen rect (overlay coords) where the canvas should render.
  rect: Rect;
  /// Cropped-image dimensions so we can convert between screen ↔ image space.
  cropped: { width: number; height: number };
  /// Tells the canvas whether it's allowed to consume pointer events.
  /// When false, all input falls through (e.g. when the rect itself is being
  /// resized and the editor is closed).
  active: boolean;
  /// Source PNG to sample from when the colour-picker tool is active. The
  /// click coordinates are in image-space (cropped pixels); offset is added
  /// before sampling so the same code path works for both `AdjustingLayer`
  /// (source = full screen, offset = rect device px) and `ResultLayer`
  /// (source = cropped image, offset = 0).
  colorPickerSource?: {
    b64: string;
    offsetX: number;
    offsetY: number;
  };
};

/// Two-canvas drawing overlay aligned with the on-screen rect.
///
/// - `committedRef`: history strokes; redrawn whenever the strokes array
///   changes (which is rarely — once per pointerup).
/// - `liveRef`: in-progress drag (marker polyline / shape preview / eraser
///   sweep). Cleared and committed on pointerup.
///
/// Coordinates inside the canvas are in image-space (cropped.width × cropped.height).
/// We size the bitmap to that and CSS-stretch it to `rect.w × rect.h` so the
/// device-pixel resolution stays high regardless of the user's drag size.
export default function EditCanvas({
  ctl,
  rect,
  cropped,
  active,
  colorPickerSource,
}: EditCanvasProps) {
  const committedRef = useRef<HTMLCanvasElement | null>(null);
  const liveRef = useRef<HTMLCanvasElement | null>(null);
  const draftRef = useRef<LiveDraft | null>(null);
  const draggingRef = useRef(false);
  // Stroke ids the object-eraser has swept over during the active drag.
  // The live canvas is redrawn imperatively (via drawLiveDraft) so we
  // don't need React state to track this — a ref is sufficient.
  const hitIdsRef = useRef<Set<string>>(new Set());
  // While the eraser is mid-drag, hide the committed canvas so the live
  // preview is the sole visible layer (otherwise the underlying committed
  // strokes would show through the destination-out cutouts on the live).
  const [eraserDragging, setEraserDragging] = useState(false);
  type TextDraft = { at: [number, number]; text: string };
  const [textDraft, setTextDraftState] = useState<TextDraft | null>(null);
  // Mirror of `textDraft` so synchronous handlers (pointerdown, blur) can
  // read the latest value without waiting for a React render. Updated by
  // every setTextDraft call below.
  const textDraftRef = useRef<TextDraft | null>(null);
  const setTextDraft = useCallback((next: TextDraft | null) => {
    textDraftRef.current = next;
    setTextDraftState(next);
  }, []);
  const textInputRef = useRef<HTMLInputElement>(null);

  // Latest strokes & cropped dims kept in refs so the pointer handlers
  // (which are stable callbacks) read fresh values.
  const strokesRef = useRef(ctl.strokes);
  strokesRef.current = ctl.strokes;
  const croppedRef = useRef(cropped);
  croppedRef.current = cropped;
  const rectRef = useRef(rect);
  rectRef.current = rect;
  const ctlRef = useRef(ctl);
  ctlRef.current = ctl;
  const colorSourceRef = useRef(colorPickerSource);
  colorSourceRef.current = colorPickerSource;

  // Lazy-decoded source canvas for the colour-picker tool. We re-decode
  // only when the source PNG identity changes.
  const sampleCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const decodedSourceB64Ref = useRef<string | null>(null);
  const ensureSampleCanvas = useCallback(async (): Promise<HTMLCanvasElement | null> => {
    const src = colorSourceRef.current;
    if (!src) return null;
    if (decodedSourceB64Ref.current === src.b64 && sampleCanvasRef.current) {
      return sampleCanvasRef.current;
    }
    const img = new Image();
    const ready = new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error("color sampler: image load failed"));
    });
    img.src = `data:image/png;base64,${src.b64}`;
    await ready;
    const c = document.createElement("canvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const ctx = c.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(img, 0, 0);
    sampleCanvasRef.current = c;
    decodedSourceB64Ref.current = src.b64;
    return c;
  }, []);

  const sampleColorAt = useCallback(
    async (imgX: number, imgY: number): Promise<string | null> => {
      const src = colorSourceRef.current;
      if (!src) return null;
      const canvas = await ensureSampleCanvas();
      if (!canvas) return null;
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;
      const x = Math.max(
        0,
        Math.min(canvas.width - 1, Math.round(imgX + src.offsetX)),
      );
      const y = Math.max(
        0,
        Math.min(canvas.height - 1, Math.round(imgY + src.offsetY)),
      );
      const data = ctx.getImageData(x, y, 1, 1).data;
      return (
        "#" +
        [data[0], data[1], data[2]]
          .map((v) => v.toString(16).padStart(2, "0").toUpperCase())
          .join("")
      );
    },
    [ensureSampleCanvas],
  );

  // Imperative committed-canvas redraw, callable from outside the React
  // render cycle. We need this on pointerup so the committed canvas can be
  // updated to its post-commit state BEFORE its visibility is restored —
  // otherwise the canvas's old pixel data shows for one frame.
  const redrawCommitted = useCallback((strokes: Stroke[]) => {
    const canvas = committedRef.current;
    if (!canvas) return;
    const dims = croppedRef.current;
    canvas.width = dims.width;
    canvas.height = dims.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, dims.width, dims.height);
    for (const s of strokes) {
      drawStrokeOnLayer(ctx, s, dims);
    }
  }, []);

  // Redraw committed strokes whenever the array or cropped dims change.
  useLayoutEffect(() => {
    redrawCommitted(ctl.strokes);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ctl.strokes, cropped.width, cropped.height]);

  // Sync the live canvas size when the cropped dimensions change.
  useLayoutEffect(() => {
    const canvas = liveRef.current;
    if (!canvas) return;
    canvas.width = cropped.width;
    canvas.height = cropped.height;
  }, [cropped.width, cropped.height]);

  const tool = ctl.tool;

  const screenToImage = useCallback(
    (clientX: number, clientY: number): [number, number] => {
      const canvas = liveRef.current;
      if (!canvas) return [0, 0];
      const box = canvas.getBoundingClientRect();
      const sx = cropped.width / box.width;
      const sy = cropped.height / box.height;
      const ix = (clientX - box.left) * sx;
      const iy = (clientY - box.top) * sy;
      return [
        clamp(ix, 0, cropped.width),
        clamp(iy, 0, cropped.height),
      ];
    },
    [cropped.width, cropped.height],
  );

  const drawLiveDraft = useCallback(() => {
    const canvas = liveRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dims = croppedRef.current;
    ctx.clearRect(0, 0, dims.width, dims.height);
    const d = draftRef.current;
    if (!d) return;
    if (d.kind === "marker") {
      drawStrokeOnLayer(
        ctx,
        { id: "live", kind: "marker", pts: d.pts, style: d.style } as MarkerStroke,
        dims,
      );
    } else if (d.kind === "eraser") {
      // Both eraser modes need to mirror the committed strokes onto the
      // live canvas (the committed canvas is hidden during the drag — see
      // `eraserDragging` state). Then they apply the mode-specific effect:
      //   - "pixel": draw the in-progress eraser path with `destination-out`,
      //     so the affected portion of the marker layer disappears in real
      //     time. The user sees the marks actually being deleted.
      //   - "object": skip rendering of every stroke whose id is in
      //     `hitIdsRef` (rendered at low opacity to give a "marked for
      //     deletion" cue) and additionally show the eraser cursor outline.
      if (d.mode === "pixel") {
        for (const s of strokesRef.current) {
          drawStrokeOnLayer(ctx, s, dims);
        }
        ctx.save();
        ctx.globalCompositeOperation = "destination-out";
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.lineWidth = d.width;
        ctx.beginPath();
        if (d.pts.length === 1) {
          ctx.arc(d.pts[0][0], d.pts[0][1], d.width / 2, 0, Math.PI * 2);
          ctx.fill();
        } else {
          ctx.moveTo(d.pts[0][0], d.pts[0][1]);
          for (let i = 1; i < d.pts.length; i++) {
            ctx.lineTo(d.pts[i][0], d.pts[i][1]);
          }
          ctx.stroke();
        }
        ctx.restore();
      } else {
        const hits = hitIdsRef.current;
        for (const s of strokesRef.current) {
          if (hits.has(s.id)) {
            ctx.save();
            ctx.globalAlpha = 0.22;
            drawStrokeOnLayer(ctx, s, dims);
            ctx.restore();
          } else {
            drawStrokeOnLayer(ctx, s, dims);
          }
        }
        // Subtle outline of the eraser cursor's last position, so the user
        // sees where the hit-test radius is while sweeping.
        const last = d.pts[d.pts.length - 1];
        if (last) {
          ctx.save();
          ctx.strokeStyle = "rgba(255, 255, 255, 0.55)";
          ctx.lineWidth = 1.25;
          ctx.beginPath();
          ctx.arc(last[0], last[1], d.width / 2, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
        }
      }
    } else {
      drawStrokeOnLayer(
        ctx,
        { id: "live", kind: d.kind, a: d.a, b: d.b, style: d.style } as ShapeStroke,
        dims,
      );
    }
  }, []);

  // Idempotent commit. Reads the active draft from the ref so it works
  // synchronously inside other event handlers, and clears the ref before
  // dispatching state updates so subsequent commits within the same tick
  // are no-ops (this is what makes onBlur + onPointerDown safe to both
  // call commitDraft without double-committing the same text).
  //
  // The stroke's fontSize is converted to image-space pixels (the canvas's
  // native resolution = device pixels) by multiplying the user's CSS-px
  // selection by `scale = cropped.width / rect.w`. The input element
  // renders at the user-facing CSS px directly, so the in-progress draft
  // and the committed text appear at exactly the same visual size.
  const commitDraft = useCallback(() => {
    const draft = textDraftRef.current;
    if (!draft) return;
    textDraftRef.current = null;
    setTextDraftState(null);
    const trimmed = draft.text.trim();
    if (!trimmed) return;
    const r = rectRef.current;
    const c = croppedRef.current;
    const scale = r.w > 0 ? c.width / r.w : 1;
    const userText = ctlRef.current.text;
    const stroke: TextStroke = {
      id: nextStrokeId(),
      kind: "text",
      at: draft.at,
      text: trimmed,
      style: { ...userText, fontSize: userText.fontSize * scale },
    };
    ctlRef.current.addStroke(stroke);
  }, []);

  const onPointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!active || !tool) return;
      if (event.button !== 0) return;
      event.stopPropagation();
      event.preventDefault();
      const at = screenToImage(event.clientX, event.clientY);
      (event.target as Element).setPointerCapture?.(event.pointerId);

      // Any pointer-down inside the canvas means the user is acting on the
      // image (drawing, placing). Fade the per-tool settings panel out so
      // it doesn't cover the captured region while they work.
      if (ctl.popoverOpen) ctl.setPopoverOpen(false);

      // Color picker is a one-shot sample on click; doesn't begin a drag.
      if (tool === "colorpicker") {
        sampleColorAt(at[0], at[1])
          .then((hex) => {
            if (!hex) return;
            navigator.clipboard.writeText(hex).catch(() => {});
            ctlRef.current.setPickedColor({ hex, key: Date.now() });
          })
          .catch((err) => console.error("colorpicker sample failed:", err));
        return;
      }

      if (tool === "marker") {
        draftRef.current = { kind: "marker", pts: [at], style: ctl.marker };
        draggingRef.current = true;
      } else if (tool === "eraser") {
        const mode = ctl.eraserMode;
        draftRef.current = {
          kind: "eraser",
          pts: [at],
          width: ctl.eraserWidth,
          mode,
        };
        draggingRef.current = true;
        setEraserDragging(true);
        if (mode === "object") {
          // Hit-test the click point immediately so a single click can
          // delete a stroke without dragging.
          const hits = new Set<string>();
          for (const s of strokesRef.current) {
            if (
              strokeHitByEraserPoint(s, at[0], at[1], ctl.eraserWidth / 2)
            ) {
              hits.add(s.id);
            }
          }
          hitIdsRef.current = hits;
        } else {
          hitIdsRef.current = new Set();
        }
      } else if (tool === "shape") {
        draftRef.current = {
          kind: ctl.shape.kind,
          a: at,
          b: at,
          style: ctl.shape.style,
        };
        draggingRef.current = true;
      } else if (tool === "text") {
        // If a previous text draft is still open, commit it (with whatever
        // text the user has typed so far) before opening a new one. The
        // commit is synchronous and idempotent — onBlur firing afterwards
        // is a no-op because commitDraft cleared the ref.
        commitDraft();
        setTextDraft({ at, text: "" });
      }
      drawLiveDraft();
    },
    [
      active,
      tool,
      ctl.marker,
      ctl.eraserWidth,
      ctl.eraserMode,
      ctl.shape,
      drawLiveDraft,
      screenToImage,
      commitDraft,
      setTextDraft,
    ],
  );

  const onPointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!draggingRef.current) return;
      const at = screenToImage(event.clientX, event.clientY);
      const d = draftRef.current;
      if (!d) return;
      if (d.kind === "marker" || d.kind === "eraser") {
        const prev = d.pts[d.pts.length - 1];
        d.pts.push(at);
        // Object eraser: hit-test the latest segment against every
        // not-yet-hit stroke, and add to hit set.
        if (d.kind === "eraser" && d.mode === "object" && prev) {
          for (const s of strokesRef.current) {
            if (hitIdsRef.current.has(s.id)) continue;
            if (strokeHitByEraserSegment(s, prev, at, d.width / 2)) {
              hitIdsRef.current.add(s.id);
            }
          }
        }
      } else {
        d.b = at;
      }
      drawLiveDraft();
    },
    [drawLiveDraft, screenToImage],
  );

  const onPointerUp = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      const d = draftRef.current;
      draftRef.current = null;
      const dims = croppedRef.current;
      // Clear the live layer.
      const canvas = liveRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx?.clearRect(0, 0, dims.width, dims.height);
      }
      if (!d) {
        if (eraserDragging) setEraserDragging(false);
        return;
      }
      // Drop trivial drags (clicks that didn't move) for shapes — a 0×0
      // rectangle is just noise. Markers/erasers commit single taps as dots.
      if (d.kind !== "marker" && d.kind !== "eraser") {
        const dx = Math.abs(d.b[0] - d.a[0]);
        const dy = Math.abs(d.b[1] - d.a[1]);
        if (dx < 2 && dy < 2) return;
      }
      if (d.kind === "marker") {
        ctl.addStroke({
          id: nextStrokeId(),
          kind: "marker",
          pts: d.pts,
          style: d.style,
        } as MarkerStroke);
      } else if (d.kind === "eraser") {
        if (d.mode === "pixel") {
          const eraserStroke: EraserStroke = {
            id: nextStrokeId(),
            kind: "eraser",
            pts: d.pts,
            width: d.width,
          };
          // Paint the post-commit state into the committed canvas BEFORE
          // flipping visibility, so the unhidden canvas already contains
          // the post-erase pixels — eliminates the one-frame flash where
          // the old marker briefly shows through. The subsequent
          // ctl.addStroke triggers React's redraw, which produces the
          // exact same content (no second flash either).
          redrawCommitted([...strokesRef.current, eraserStroke]);
          setEraserDragging(false);
          ctl.addStroke(eraserStroke);
        } else {
          // Object mode: paint without the soon-to-be-removed strokes
          // before unhiding, then commit the removal so React state
          // catches up.
          const removeIds = hitIdsRef.current;
          const remaining = strokesRef.current.filter((s) => !removeIds.has(s.id));
          redrawCommitted(remaining);
          setEraserDragging(false);
          ctl.removeStrokes(removeIds);
          hitIdsRef.current = new Set();
        }
      } else {
        ctl.addStroke({
          id: nextStrokeId(),
          kind: d.kind,
          a: d.a,
          b: d.b,
          style: d.style,
        } as ShapeStroke);
      }
      void event;
    },
    [ctl, eraserDragging, redrawCommitted],
  );

  // If the tool changes away from text while a draft is open, commit it.
  useEffect(() => {
    if (tool !== "text") commitDraft();
  }, [tool, commitDraft]);

  // Convert the text-draft image-space anchor into screen coords for the
  // floating <input>'s position.
  const textDraftScreen: { left: number; top: number; fontSize: number } | null =
    textDraft
      ? {
          left: rect.x + (textDraft.at[0] / cropped.width) * rect.w,
          top: rect.y + (textDraft.at[1] / cropped.height) * rect.h,
          fontSize: ctl.text.fontSize,
        }
      : null;

  return (
    <>
      <div
        className="screenie-edit-canvas"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onMouseDown={(e) => {
          // Stop the overlay's backdrop-close handler when we're consuming
          // the click ourselves.
          if (active && tool) e.stopPropagation();
        }}
        style={{
          position: "absolute",
          left: rect.x,
          top: rect.y,
          width: rect.w,
          height: rect.h,
          pointerEvents: active && tool ? "auto" : "none",
          cursor: cursorForTool(tool, active),
          zIndex: 6,
        }}
      >
        <canvas
          ref={committedRef}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
            // Hidden mid-eraser-drag so the live preview's destination-out
            // cutouts are actually visible — otherwise the unmodified
            // committed strokes would peek through where the live layer is
            // transparent and the user wouldn't see anything being erased.
            visibility: eraserDragging ? "hidden" : "visible",
          }}
        />
        <canvas
          ref={liveRef}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
          }}
        />
      </div>
      {textDraftScreen && (
        <input
          ref={textInputRef}
          className="screenie-edit-text-input"
          value={textDraft?.text ?? ""}
          placeholder="Add text…"
          onChange={(e) => {
            const v = e.target.value;
            setTextDraft(
              textDraftRef.current
                ? { ...textDraftRef.current, text: v }
                : { at: [0, 0], text: v },
            );
          }}
          onKeyDown={(e) => {
            e.stopPropagation();
            if (e.key === "Enter") {
              e.preventDefault();
              commitDraft();
            } else if (e.key === "Escape") {
              e.preventDefault();
              setTextDraft(null);
            }
          }}
          onBlur={commitDraft}
          onMouseDown={(e) => e.stopPropagation()}
          // Styling kept minimal so the input's text occupies exactly the
          // pixels the committed `drawText` will fill: zero padding, zero
          // border, no background. With `box-sizing: content-box` (the
          // default) the input's text glyph baseline sits at the same y
          // offset the canvas renders at when textBaseline = "top".
          style={{
            position: "absolute",
            left: textDraftScreen.left,
            top: textDraftScreen.top,
            fontSize: textDraftScreen.fontSize,
            lineHeight: 1,
            fontWeight: 600,
            color: ctl.text.color,
            background: "transparent",
            border: "none",
            padding: 0,
            margin: 0,
            outline: "none",
            zIndex: 14,
          }}
        />
      )}
    </>
  );
}

function cursorForTool(tool: EditController["tool"], active: boolean): string {
  if (!active || !tool) return "default";
  if (tool === "marker") return "crosshair";
  if (tool === "eraser") return "cell";
  if (tool === "text") return "text";
  if (tool === "colorpicker") return "crosshair";
  return "crosshair";
}

function drawStrokeOnLayer(
  ctx: CanvasRenderingContext2D,
  stroke: Stroke,
  _dims: { width: number; height: number },
): void {
  switch (stroke.kind) {
    case "marker":
      drawMarker(ctx, stroke);
      break;
    case "eraser":
      drawCommittedEraser(ctx, stroke);
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

function applyStyle(
  ctx: CanvasRenderingContext2D,
  style: StrokeStyle,
): void {
  ctx.strokeStyle = style.color;
  ctx.fillStyle = style.color;
  ctx.lineWidth = style.width;
  // Multiply (not assign) so an outer ctx.globalAlpha set by the caller
  // (e.g. the object-eraser preview that dims hit strokes to 0.22) cascades
  // into the per-stroke draw.
  ctx.globalAlpha = ctx.globalAlpha * (style.opacity ?? 1);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

function drawMarker(ctx: CanvasRenderingContext2D, stroke: MarkerStroke): void {
  if (stroke.pts.length === 0) return;
  ctx.save();
  applyStyle(ctx, stroke.style);
  if (stroke.pts.length === 1) {
    ctx.beginPath();
    ctx.arc(stroke.pts[0][0], stroke.pts[0][1], Math.max(1, stroke.style.width / 2), 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.beginPath();
    ctx.moveTo(stroke.pts[0][0], stroke.pts[0][1]);
    for (let i = 1; i < stroke.pts.length; i++) ctx.lineTo(stroke.pts[i][0], stroke.pts[i][1]);
    ctx.stroke();
  }
  ctx.restore();
}

/// Render an EraserStroke from history onto the committed canvas. The
/// source must be FULLY OPAQUE — `destination-out` removes pixels in
/// proportion to the source alpha, so anything less than 1.0 leaves a
/// ghost of the erased content behind. (Earlier impl used 0.7 alpha for a
/// "translucent preview" effect, which is what produced the visible light
/// mark after every erase.)
function drawCommittedEraser(
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
  if (stroke.pts.length === 1) {
    ctx.arc(stroke.pts[0][0], stroke.pts[0][1], stroke.width / 2, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.moveTo(stroke.pts[0][0], stroke.pts[0][1]);
    for (let i = 1; i < stroke.pts.length; i++) {
      ctx.lineTo(stroke.pts[i][0], stroke.pts[i][1]);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function drawShape(ctx: CanvasRenderingContext2D, stroke: ShapeStroke): void {
  ctx.save();
  applyStyle(ctx, stroke.style);
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
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    const dx = bx - ax;
    const dy = by - ay;
    const len = Math.hypot(dx, dy);
    if (len >= 1e-3) {
      const head = Math.min(Math.max(stroke.style.width * 3.5, 8), 28);
      const angle = Math.atan2(dy, dx);
      const wing = Math.PI / 7;
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
  }
  ctx.restore();
}

function drawText(ctx: CanvasRenderingContext2D, stroke: TextStroke): void {
  if (!stroke.text) return;
  ctx.save();
  ctx.globalAlpha = ctx.globalAlpha * (stroke.style.opacity ?? 1);
  const fontSize = stroke.style.fontSize;
  ctx.font = `600 ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif`;
  ctx.textBaseline = "top";
  // No backing pill — the user explicitly asked for naked text. Glyph
  // origin (textBaseline = "top") matches the in-progress input element's
  // top-left so the visible text doesn't shift between draft and commit.
  ctx.fillStyle = stroke.style.color;
  ctx.fillText(stroke.text, stroke.at[0], stroke.at[1]);
  ctx.restore();
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(Math.max(n, lo), hi);
}

/// Object-eraser hit testing: does the eraser cursor at (x, y) with the
/// given radius (in image-space pixels) overlap any drawn pixel of the
/// given stroke? Approximate — bbox checks for shapes, polyline distance
/// for marker / eraser polylines. Good enough for "swept the eraser over
/// this object" UX without rasterizing every stroke per pointermove.
function strokeHitByEraserPoint(
  stroke: Stroke,
  x: number,
  y: number,
  radius: number,
): boolean {
  if (stroke.kind === "marker" || stroke.kind === "eraser") {
    const half =
      stroke.kind === "marker" ? stroke.style.width / 2 : stroke.width / 2;
    const r = radius + half;
    const r2 = r * r;
    const pts = stroke.pts;
    for (let i = 0; i < pts.length; i++) {
      const [px, py] = pts[i];
      if ((px - x) * (px - x) + (py - y) * (py - y) <= r2) return true;
      if (i + 1 < pts.length) {
        const [qx, qy] = pts[i + 1];
        if (pointSegDistSq(x, y, px, py, qx, qy) <= r2) return true;
      }
    }
    return false;
  }
  if (stroke.kind === "rect") {
    const r = radius + stroke.style.width / 2;
    const r2 = r * r;
    const x0 = Math.min(stroke.a[0], stroke.b[0]);
    const y0 = Math.min(stroke.a[1], stroke.b[1]);
    const x1 = Math.max(stroke.a[0], stroke.b[0]);
    const y1 = Math.max(stroke.a[1], stroke.b[1]);
    return (
      pointSegDistSq(x, y, x0, y0, x1, y0) <= r2 ||
      pointSegDistSq(x, y, x1, y0, x1, y1) <= r2 ||
      pointSegDistSq(x, y, x1, y1, x0, y1) <= r2 ||
      pointSegDistSq(x, y, x0, y1, x0, y0) <= r2
    );
  }
  if (stroke.kind === "line" || stroke.kind === "arrow") {
    const r = radius + stroke.style.width / 2;
    return (
      pointSegDistSq(x, y, stroke.a[0], stroke.a[1], stroke.b[0], stroke.b[1]) <=
      r * r
    );
  }
  if (stroke.kind === "ellipse") {
    const cx = (stroke.a[0] + stroke.b[0]) / 2;
    const cy = (stroke.a[1] + stroke.b[1]) / 2;
    const rx = Math.abs(stroke.b[0] - stroke.a[0]) / 2;
    const ry = Math.abs(stroke.b[1] - stroke.a[1]) / 2;
    if (rx <= 0 || ry <= 0) return false;
    const nx = (x - cx) / rx;
    const ny = (y - cy) / ry;
    const d = Math.hypot(nx, ny);
    const avgR = (rx + ry) / 2;
    const tol = (radius + stroke.style.width / 2) / Math.max(avgR, 1);
    return Math.abs(d - 1) <= tol;
  }
  if (stroke.kind === "text") {
    const fs = stroke.style.fontSize;
    // Approximate text bbox; we don't have measureText here.
    const w = fs * Math.max(stroke.text.length, 1) * 0.6 + fs;
    const h = fs * 1.4;
    return (
      x >= stroke.at[0] - radius &&
      x <= stroke.at[0] + w + radius &&
      y >= stroke.at[1] - radius &&
      y <= stroke.at[1] + h + radius
    );
  }
  return false;
}

/// Approximate hit test for the eraser segment from `a` to `b`. Sampling
/// the two endpoints is enough at typical pointermove granularity.
function strokeHitByEraserSegment(
  stroke: Stroke,
  a: readonly [number, number],
  b: readonly [number, number],
  radius: number,
): boolean {
  return (
    strokeHitByEraserPoint(stroke, a[0], a[1], radius) ||
    strokeHitByEraserPoint(stroke, b[0], b[1], radius)
  );
}

function pointSegDistSq(
  px: number,
  py: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): number {
  const dx = x2 - x1;
  const dy = y2 - y1;
  if (dx === 0 && dy === 0) return (px - x1) * (px - x1) + (py - y1) * (py - y1);
  const t = Math.max(
    0,
    Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)),
  );
  const cx = x1 + t * dx;
  const cy = y1 + t * dy;
  return (px - cx) * (px - cx) + (py - cy) * (py - cy);
}
