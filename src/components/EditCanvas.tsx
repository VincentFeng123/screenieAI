import {
  type CSSProperties,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { Move, Trash2 } from "lucide-react";
import {
  type EraserStroke,
  type MarkerStroke,
  type Point2,
  type Shape,
  type ShapeStroke,
  type Stroke,
  type StrokeStyle,
  type TextStroke,
} from "../lib/editTypes";
import { nextStrokeId, type EditController } from "../lib/useEditController";
import { BlurredBackdrop, TOOLBAR_FROST } from "./Frosted";

type Rect = { x: number; y: number; w: number; h: number };
const SELECTION_CONTROL_SIZE = 30;
const SELECTION_CONTROL_GAP = 4;
const SELECTION_CONTROLS_W =
  SELECTION_CONTROL_SIZE * 2 + SELECTION_CONTROL_GAP;

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

type EditableStroke = ShapeStroke | TextStroke;
type SelectedObject = { id: string; kind: "shape" | "text" };
type ObjectDrag = {
  id: string;
  start: EditableStroke;
  current: EditableStroke;
  pointerStart: [number, number];
  dragging: boolean;
};
type TextDraft = {
  id?: string;
  existing?: boolean;
  at: [number, number];
  text: string;
  style: StrokeStyle & { fontSize: number };
};

export type EditCanvasProps = {
  ctl: EditController;
  /// On-screen rect (overlay coords) where the canvas should render.
  rect: Rect;
  /// Cropped-image dimensions so we can convert between screen ↔ image space.
  cropped: { width: number; height: number };
  /// Optional temporary geometry used while the capture rect is being resized.
  /// The editing model still targets `rect`/`cropped`; this only controls where
  /// already-committed annotations are painted so they can stay visually fixed
  /// during a live resize.
  renderRect?: Rect;
  renderCropped?: { width: number; height: number };
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
  /// Full-screen capture used to render selected-object controls with the same
  /// frosted treatment as the edit toolbar.
  screenPngB64?: string;
  screenW?: number;
  screenH?: number;
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
  renderRect: renderRectProp,
  renderCropped: renderCroppedProp,
  active,
  colorPickerSource,
  screenPngB64,
  screenW,
  screenH,
}: EditCanvasProps) {
  const renderRect = renderRectProp ?? rect;
  const renderCropped = renderCroppedProp ?? cropped;
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
  const [textDraft, setTextDraftState] = useState<TextDraft | null>(null);
  const [selectedObject, setSelectedObject] = useState<SelectedObject | null>(null);
  const [dragPreviewStroke, setDragPreviewStroke] = useState<EditableStroke | null>(null);
  // Mirror of `textDraft` so synchronous handlers (pointerdown, blur) can
  // read the latest value without waiting for a React render. Updated by
  // every setTextDraft call below.
  const textDraftRef = useRef<TextDraft | null>(null);
  const hiddenStrokeIdRef = useRef<string | null>(null);
  const objectDragRef = useRef<ObjectDrag | null>(null);
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
  const renderCroppedRef = useRef(renderCropped);
  renderCroppedRef.current = renderCropped;
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
    const dims = renderCroppedRef.current;
    canvas.width = dims.width;
    canvas.height = dims.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, dims.width, dims.height);
    for (const s of strokes) {
      if (s.id === hiddenStrokeIdRef.current) continue;
      drawStrokeOnLayer(ctx, s, dims);
    }
  }, []);

  // Redraw committed strokes whenever the array or cropped dims change.
  useLayoutEffect(() => {
    redrawCommitted(ctl.strokes);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ctl.strokes, renderCropped.width, renderCropped.height]);

  // Sync the live canvas size when the cropped dimensions change.
  useLayoutEffect(() => {
    const canvas = liveRef.current;
    if (!canvas) return;
    canvas.width = renderCropped.width;
    canvas.height = renderCropped.height;
  }, [renderCropped.width, renderCropped.height]);

  const tool = ctl.tool;

  const screenToImageInfo = useCallback(
    (clientX: number, clientY: number): { point: [number, number]; inside: boolean } => {
      const canvas = liveRef.current;
      if (!canvas) return { point: [0, 0], inside: false };
      const box = canvas.getBoundingClientRect();
      if (box.width <= 0 || box.height <= 0) {
        return { point: [0, 0], inside: false };
      }
      const sx = renderCropped.width / box.width;
      const sy = renderCropped.height / box.height;
      const ix = (clientX - box.left) * sx;
      const iy = (clientY - box.top) * sy;
      const inside =
        ix >= 0 &&
        ix <= renderCropped.width &&
        iy >= 0 &&
        iy <= renderCropped.height;
      return {
        point: [
          clamp(ix, 0, renderCropped.width),
          clamp(iy, 0, renderCropped.height),
        ],
        inside,
      };
    },
    [renderCropped.width, renderCropped.height],
  );

  const screenToImage = useCallback(
    (clientX: number, clientY: number): [number, number] =>
      screenToImageInfo(clientX, clientY).point,
    [screenToImageInfo],
  );

  const drawLiveDraft = useCallback(() => {
    const canvas = liveRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dims = renderCroppedRef.current;
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
    hiddenStrokeIdRef.current = null;
    const trimmed = draft.text.trim();
    if (!trimmed) {
      if (draft.id && draft.existing) {
        ctlRef.current.removeStrokes(new Set([draft.id]));
      }
      setSelectedObject(null);
      redrawCommitted(strokesRef.current.filter((s) => s.id !== draft.id));
      return;
    }
    const stroke: TextStroke = {
      id: draft.id ?? nextStrokeId(),
      kind: "text",
      at: draft.at,
      text: trimmed,
      style: draft.style,
    };
    if (!strokeFullyWithinBounds(stroke, croppedRef.current)) {
      if (draft.id && draft.existing) {
        ctlRef.current.removeStrokes(new Set([draft.id]));
      }
      setSelectedObject(null);
      redrawCommitted(strokesRef.current.filter((s) => s.id !== draft.id));
      return;
    }
    if (draft.existing) {
      ctlRef.current.updateStroke(stroke);
    } else {
      ctlRef.current.addStroke(stroke);
    }
    setSelectedObject(draft.existing ? { id: stroke.id, kind: "text" } : null);
    redrawCommitted(
      draft.existing
        ? strokesRef.current.map((s) => (s.id === stroke.id ? stroke : s))
        : [...strokesRef.current, stroke],
    );
  }, [redrawCommitted]);

  const clearLiveLayer = useCallback(() => {
    const canvas = liveRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dims = renderCroppedRef.current;
    ctx?.clearRect(0, 0, dims.width, dims.height);
  }, []);

  const drawLiveStroke = useCallback((stroke: EditableStroke) => {
    const canvas = liveRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dims = renderCroppedRef.current;
    ctx.clearRect(0, 0, dims.width, dims.height);
    drawStrokeOnLayer(ctx, stroke, dims);
  }, []);

  const openTextDraftForStroke = useCallback(
    (stroke: TextStroke) => {
      const draft: TextDraft = {
        id: stroke.id,
        existing: true,
        at: [...stroke.at] as [number, number],
        text: stroke.text,
        style: { ...stroke.style },
      };
      hiddenStrokeIdRef.current = stroke.id;
      setSelectedObject({ id: stroke.id, kind: "text" });
      setTextDraft(draft);
      redrawCommitted(strokesRef.current);
    },
    [redrawCommitted, setTextDraft],
  );

  const finishObjectDrag = useCallback(() => {
    const drag = objectDragRef.current;
    if (!drag) return;
    objectDragRef.current = null;
    clearLiveLayer();
    const moved = !sameEditableStroke(drag.start, drag.current);
    if (drag.current.kind === "text") {
      const current = drag.current as TextStroke;
      if (textDraftRef.current?.id === drag.id) {
        setTextDraft({
          ...textDraftRef.current,
          at: [...current.at] as [number, number],
        });
        // Keep the original stroke hidden while its input is open; commitDraft
        // will replace it when editing finishes.
        setDragPreviewStroke(null);
        redrawCommitted(strokesRef.current);
        return;
      }
      hiddenStrokeIdRef.current = null;
      if (moved && strokeFullyWithinBounds(current, croppedRef.current)) {
        setDragPreviewStroke(current);
        ctlRef.current.updateStroke(current);
        redrawCommitted(
          strokesRef.current.map((s) =>
            s.id === current.id ? current : s,
          ),
        );
      } else {
        redrawCommitted(strokesRef.current);
        setDragPreviewStroke(null);
      }
      setSelectedObject({ id: drag.id, kind: "text" });
      return;
    }

    hiddenStrokeIdRef.current = null;
    if (moved && strokeFullyWithinBounds(drag.current, croppedRef.current)) {
      setDragPreviewStroke(drag.current);
      ctlRef.current.updateStroke(drag.current);
      redrawCommitted(
        strokesRef.current.map((s) =>
          s.id === drag.current.id ? drag.current : s,
        ),
      );
    } else {
      redrawCommitted(strokesRef.current);
      setDragPreviewStroke(null);
    }
    setSelectedObject({ id: drag.id, kind: "shape" });
  }, [clearLiveLayer, redrawCommitted, setTextDraft]);

  useEffect(() => {
    if (!dragPreviewStroke || objectDragRef.current) return;
    if (selectedObject?.id !== dragPreviewStroke.id) {
      setDragPreviewStroke(null);
      return;
    }
    const committed = selectedEditableStroke(
      {
        id: dragPreviewStroke.id,
        kind: dragPreviewStroke.kind === "text" ? "text" : "shape",
      },
      null,
      null,
      ctl.strokes,
    );
    if (committed && sameEditableStroke(committed, dragPreviewStroke)) {
      setDragPreviewStroke(null);
    }
  }, [ctl.strokes, dragPreviewStroke, selectedObject?.id]);

  const beginSelectedObjectDrag = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>) => {
      if (event.button !== 0) return;
      const stroke = selectedEditableStroke(
        selectedObject,
        textDraftRef.current,
        dragPreviewStroke,
        strokesRef.current,
      );
      if (!stroke) return;
      event.preventDefault();
      event.stopPropagation();
      const at = screenToImage(event.clientX, event.clientY);
      objectDragRef.current = {
        id: stroke.id,
        start: stroke,
        current: stroke,
        pointerStart: at,
        dragging: false,
      };
      setDragPreviewStroke(stroke);
      if (stroke.kind === "text" && textDraftRef.current?.id === stroke.id) {
        clearLiveLayer();
      } else {
        hiddenStrokeIdRef.current = stroke.id;
        redrawCommitted(strokesRef.current);
        drawLiveStroke(stroke);
      }
      event.currentTarget.setPointerCapture?.(event.pointerId);
    },
    [
      clearLiveLayer,
      drawLiveStroke,
      redrawCommitted,
      screenToImage,
      dragPreviewStroke,
      selectedObject,
    ],
  );

  const moveSelectedObjectDrag = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>) => {
      const drag = objectDragRef.current;
      if (!drag) return;
      event.preventDefault();
      event.stopPropagation();
      const at = screenToImage(event.clientX, event.clientY);
      const dx = at[0] - drag.pointerStart[0];
      const dy = at[1] - drag.pointerStart[1];
      if (!drag.dragging && Math.hypot(dx, dy) < 3) return;
      drag.dragging = true;
      drag.current = translateEditableStrokeWithinBounds(
        drag.start,
        dx,
        dy,
        croppedRef.current,
      );
      setDragPreviewStroke(drag.current);
      if (drag.current.kind === "text" && textDraftRef.current?.id === drag.id) {
        setTextDraft({
          ...textDraftRef.current,
          at: [...(drag.current as TextStroke).at] as [number, number],
        });
        clearLiveLayer();
        return;
      }
      drawLiveStroke(drag.current);
    },
    [clearLiveLayer, drawLiveStroke, screenToImage, setTextDraft],
  );

  const endSelectedObjectDrag = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>) => {
      if (!objectDragRef.current) return;
      event.preventDefault();
      event.stopPropagation();
      finishObjectDrag();
    },
    [finishObjectDrag],
  );

  const dismissSelectedObject = useCallback(
    (event?: React.PointerEvent<HTMLElement>) => {
      event?.preventDefault();
      event?.stopPropagation();
      commitDraft();
      objectDragRef.current = null;
      hiddenStrokeIdRef.current = null;
      setSelectedObject(null);
      setDragPreviewStroke(null);
      clearLiveLayer();
      redrawCommitted(strokesRef.current);
    },
    [clearLiveLayer, commitDraft, redrawCommitted],
  );

  const selectExistingStroke = useCallback(
    (strokeId: string, event: React.PointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return;
      const stroke = strokesRef.current.find((s): s is EditableStroke =>
        isEditableStroke(s) && s.id === strokeId,
      );
      if (!stroke) return;
      const atInfo = screenToImageInfo(event.clientX, event.clientY);
      if (
        !atInfo.inside ||
        !pointHitsEditableStroke(stroke, atInfo.point, croppedRef.current)
      ) {
        dismissSelectedObject(event);
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      if (ctlRef.current.popoverOpen) ctlRef.current.setPopoverOpen(false);
      commitDraft();
      clearLiveLayer();
      setDragPreviewStroke(null);
      if (stroke.kind === "text" && ctlRef.current.tool === "text") {
        openTextDraftForStroke(stroke);
        return;
      }
      hiddenStrokeIdRef.current = null;
      setSelectedObject({
        id: stroke.id,
        kind: stroke.kind === "text" ? "text" : "shape",
      });
      redrawCommitted(strokesRef.current);
    },
    [
      clearLiveLayer,
      commitDraft,
      dismissSelectedObject,
      openTextDraftForStroke,
      redrawCommitted,
      screenToImageInfo,
    ],
  );

  const deleteSelectedObject = useCallback(() => {
    const draft = textDraftRef.current;
    if (draft) {
      textDraftRef.current = null;
      setTextDraftState(null);
      hiddenStrokeIdRef.current = null;
      if (draft.id && draft.existing) {
        ctlRef.current.removeStrokes(new Set([draft.id]));
        redrawCommitted(strokesRef.current.filter((s) => s.id !== draft.id));
      } else {
        redrawCommitted(strokesRef.current);
      }
      setDragPreviewStroke(null);
      setSelectedObject(null);
      return;
    }
    const selected = selectedObject;
    if (!selected) return;
    ctlRef.current.removeStrokes(new Set([selected.id]));
    setSelectedObject(null);
    setDragPreviewStroke(null);
    redrawCommitted(strokesRef.current.filter((s) => s.id !== selected.id));
  }, [redrawCommitted, selectedObject]);

  const onPointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!active || !tool) return;
      if (event.button !== 0) return;
      event.stopPropagation();
      event.preventDefault();
      const atInfo = screenToImageInfo(event.clientX, event.clientY);
      if (!atInfo.inside) return;
      const at = atInfo.point;
      (event.target as Element).setPointerCapture?.(event.pointerId);

      // Any pointer-down inside the canvas means the user is acting on the
      // image (drawing, placing). Fade the per-tool settings panel out so
      // it doesn't cover the captured region while they work.
      if (ctl.popoverOpen) ctl.setPopoverOpen(false);

      if (tool === "text") {
        const hit = findTopmostTextAt(strokesRef.current, at, croppedRef.current);
        if (hit) {
          commitDraft();
          openTextDraftForStroke(hit);
          return;
        }
      } else if (tool === "shape") {
        const hit = findTopmostShapeAt(strokesRef.current, at);
        if (hit) {
          commitDraft();
          setSelectedObject({ id: hit.id, kind: "shape" });
          return;
        }
      } else {
        setSelectedObject(null);
      }

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
        setSelectedObject(null);
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
        setSelectedObject(null);
        setDragPreviewStroke(null);
        const r = rectRef.current;
        const c = croppedRef.current;
        const scale = r.w > 0 ? c.width / r.w : 1;
        const userText = ctlRef.current.text;
        const draftId = nextStrokeId();
        setTextDraft({
          id: draftId,
          existing: false,
          at,
          text: "",
          style: { ...userText, fontSize: userText.fontSize * scale },
        });
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
      screenToImageInfo,
      commitDraft,
      openTextDraftForStroke,
      setTextDraft,
    ],
  );

  const onPointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const drag = objectDragRef.current;
      if (drag) {
        const at = screenToImage(event.clientX, event.clientY);
        const dx = at[0] - drag.pointerStart[0];
        const dy = at[1] - drag.pointerStart[1];
        if (!drag.dragging && Math.hypot(dx, dy) < 3) return;
        drag.dragging = true;
        drag.current = translateEditableStrokeWithinBounds(
          drag.start,
          dx,
          dy,
          croppedRef.current,
        );
        if (drag.current.kind === "text" && textDraftRef.current?.id === drag.id) {
          setTextDraft({
            ...textDraftRef.current,
            at: [...(drag.current as TextStroke).at] as [number, number],
          });
          clearLiveLayer();
          return;
        }
        drawLiveStroke(drag.current);
        return;
      }
      if (!draggingRef.current) return;
      const atInfo = screenToImageInfo(event.clientX, event.clientY);
      if (!atInfo.inside) return;
      const at = atInfo.point;
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
    [
      clearLiveLayer,
      drawLiveDraft,
      drawLiveStroke,
      screenToImage,
      screenToImageInfo,
      setTextDraft,
    ],
  );

  const onPointerUp = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (objectDragRef.current) {
        finishObjectDrag();
        void event;
        return;
      }
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
        const shapeStroke = {
          id: nextStrokeId(),
          kind: d.kind,
          a: d.a,
          b: d.b,
          style: d.style,
        } as ShapeStroke;
        if (!strokeFullyWithinBounds(shapeStroke, dims)) return;
        ctl.addStroke(shapeStroke);
        setSelectedObject(null);
      }
      void event;
    },
    [ctl, eraserDragging, finishObjectDrag, redrawCommitted],
  );

  // If the tool changes away from text while a draft is open, commit it.
  useEffect(() => {
    if (tool !== "text") commitDraft();
    if (tool === "marker" || tool === "eraser" || tool === "colorpicker") {
      objectDragRef.current = null;
      hiddenStrokeIdRef.current = null;
      setSelectedObject(null);
      setDragPreviewStroke(null);
      clearLiveLayer();
      redrawCommitted(strokesRef.current);
      return;
    }
    if (tool !== "shape" && hiddenStrokeIdRef.current && !textDraftRef.current) {
      hiddenStrokeIdRef.current = null;
      clearLiveLayer();
      redrawCommitted(strokesRef.current);
    }
  }, [tool, commitDraft, clearLiveLayer, redrawCommitted]);

  // Collapsing the edit toolbar should not leave an empty transient text
  // object behind. Existing edited text is committed; brand-new empty drafts
  // are simply discarded by commitDraft's trim check.
  useEffect(() => {
    if (ctl.open) return;
    commitDraft();
    setSelectedObject(null);
    hiddenStrokeIdRef.current = null;
    setDragPreviewStroke(null);
    clearLiveLayer();
    redrawCommitted(strokesRef.current);
  }, [ctl.open, commitDraft, clearLiveLayer, redrawCommitted]);

  useLayoutEffect(() => {
    if (!textDraft) return;
    textInputRef.current?.focus();
    textInputRef.current?.select();
  }, [textDraft?.id]);

  // Convert the text-draft image-space anchor into screen coords for the
  // floating <input>'s position.
  const imageToScreen = useCallback(
    (p: Point2): [number, number] => [
      renderRect.x + (p[0] / renderCropped.width) * renderRect.w,
      renderRect.y + (p[1] / renderCropped.height) * renderRect.h,
    ],
    [
      renderCropped.height,
      renderCropped.width,
      renderRect.h,
      renderRect.w,
      renderRect.x,
      renderRect.y,
    ],
  );

  const textDraftScreen: { left: number; top: number; fontSize: number; color: string } | null =
    textDraft
      ? {
          left: renderRect.x + (textDraft.at[0] / renderCropped.width) * renderRect.w,
          top: renderRect.y + (textDraft.at[1] / renderCropped.height) * renderRect.h,
          fontSize:
            (textDraft.style.fontSize / Math.max(renderCropped.width, 1)) *
            renderRect.w,
          color: textDraft.style.color,
        }
      : null;

  const rawSelectedControlsPlacement = selectedObject
    ? controlsPlacementForSelection(
        selectedObject,
        textDraft,
        dragPreviewStroke,
        strokesRef.current,
        imageToScreen,
        renderCropped,
      )
    : null;
  const selectedControlsPlacement = rawSelectedControlsPlacement
    ? clampControlsPlacement(rawSelectedControlsPlacement, renderRect)
    : null;
  const objectSelectionEnabled =
    tool !== "marker" && tool !== "eraser" && tool !== "colorpicker";
  const annotationHitTargets = objectSelectionEnabled
    ? ctl.strokes
        .filter((s): s is EditableStroke => isEditableStroke(s))
        .filter((s) => s.id !== hiddenStrokeIdRef.current)
        .map((s) => ({
          stroke: s,
          style: hitTargetStyleForStroke(s, imageToScreen, renderCropped, renderRect),
        }))
        .filter(
          (item): item is { stroke: EditableStroke; style: CSSProperties } =>
            item.style !== null,
        )
    : [];

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
          left: renderRect.x,
          top: renderRect.y,
          width: renderRect.w,
          height: renderRect.h,
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
      {selectedObject && (
        <div
          aria-hidden
          onPointerDown={dismissSelectedObject}
          style={{
            position: "fixed",
            inset: 0,
            background: "transparent",
            zIndex: 11,
            pointerEvents: "auto",
          }}
        />
      )}
      {annotationHitTargets.map(({ stroke, style }) => (
        <div
          key={`annotation-hit-${stroke.id}`}
          aria-hidden
          onPointerDown={(e) => selectExistingStroke(stroke.id, e)}
          style={style}
        />
      ))}
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
                : {
                    at: [0, 0],
                    text: v,
                    style: {
                      ...ctl.text,
                      fontSize:
                        ctl.text.fontSize *
                        (cropped.width / Math.max(rect.w, 1)),
                    },
                  },
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
              hiddenStrokeIdRef.current = null;
              setSelectedObject(null);
              setDragPreviewStroke(null);
              clearLiveLayer();
              redrawCommitted(strokesRef.current);
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
            color: textDraftScreen.color,
            background: "transparent",
            border: "none",
            padding: 0,
            margin: 0,
            outline: "none",
            zIndex: 14,
          }}
        />
      )}
      {selectedControlsPlacement && (
        <div
          style={{
            position: "absolute",
            left: selectedControlsPlacement.left,
            top: selectedControlsPlacement.top,
            width: SELECTION_CONTROLS_W,
            height: SELECTION_CONTROL_SIZE,
            display: "flex",
            gap: SELECTION_CONTROL_GAP,
            zIndex: 15,
            pointerEvents: "auto",
          }}
        >
          <button
            type="button"
            className="screenie-edit-pill screenie-action-btn"
            aria-label="Move selected annotation"
            title="Move"
            onMouseDown={(e) => {
              e.preventDefault();
              e.stopPropagation();
            }}
            onPointerDown={beginSelectedObjectDrag}
            onPointerMove={moveSelectedObjectDrag}
            onPointerUp={endSelectedObjectDrag}
            onPointerCancel={endSelectedObjectDrag}
            style={selectionControlButtonStyle("grab")}
          >
            {screenPngB64 && screenW && screenH && (
              <BlurredBackdrop
                src={screenPngB64}
                screenW={screenW}
                screenH={screenH}
                blurRadius={TOOLBAR_FROST.blurRadius}
                imageBrightness={TOOLBAR_FROST.imageBrightness}
                tint={TOOLBAR_FROST.tint}
                fill={TOOLBAR_FROST.fill}
                persistImage
              />
            )}
            <span style={{ position: "relative", zIndex: 1, display: "flex" }}>
              <Move size={14} strokeWidth={2} aria-hidden />
            </span>
          </button>
          <button
            type="button"
            className="screenie-edit-pill screenie-action-btn"
            aria-label="Delete selected annotation"
            title="Delete"
            onMouseDown={(e) => {
              e.preventDefault();
              e.stopPropagation();
            }}
            onClick={(e) => {
              e.stopPropagation();
              deleteSelectedObject();
            }}
            style={selectionControlButtonStyle("pointer")}
          >
            {screenPngB64 && screenW && screenH && (
              <BlurredBackdrop
                src={screenPngB64}
                screenW={screenW}
                screenH={screenH}
                blurRadius={TOOLBAR_FROST.blurRadius}
                imageBrightness={TOOLBAR_FROST.imageBrightness}
                tint={TOOLBAR_FROST.tint}
                fill={TOOLBAR_FROST.fill}
                persistImage
              />
            )}
            <span style={{ position: "relative", zIndex: 1, display: "flex" }}>
              <Trash2 size={14} strokeWidth={2} aria-hidden />
            </span>
          </button>
        </div>
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

function isShapeStroke(stroke: Stroke): stroke is ShapeStroke {
  return (
    stroke.kind === "rect" ||
    stroke.kind === "ellipse" ||
    stroke.kind === "line" ||
    stroke.kind === "arrow"
  );
}

function isEditableStroke(stroke: Stroke): stroke is EditableStroke {
  return stroke.kind === "text" || isShapeStroke(stroke);
}

function findTopmostTextAt(
  strokes: Stroke[],
  point: Point2,
  dims: { width: number; height: number },
): TextStroke | null {
  for (let i = strokes.length - 1; i >= 0; i--) {
    const s = strokes[i];
    if (s.kind !== "text") continue;
    const b = strokeBounds(s, dims);
    if (
      point[0] >= b.x0 &&
      point[0] <= b.x1 &&
      point[1] >= b.y0 &&
      point[1] <= b.y1
    ) {
      return s;
    }
  }
  return null;
}

function findTopmostShapeAt(strokes: Stroke[], point: Point2): ShapeStroke | null {
  for (let i = strokes.length - 1; i >= 0; i--) {
    const s = strokes[i];
    if (!isShapeStroke(s)) continue;
    if (shapeHitTest(s, point[0], point[1])) return s;
  }
  return null;
}

function pointHitsEditableStroke(
  stroke: EditableStroke,
  point: Point2,
  dims: { width: number; height: number },
): boolean {
  if (stroke.kind === "text") {
    const b = strokeBounds(stroke, dims);
    return (
      point[0] >= b.x0 &&
      point[0] <= b.x1 &&
      point[1] >= b.y0 &&
      point[1] <= b.y1
    );
  }
  return shapeHitTest(stroke, point[0], point[1]);
}

function textDraftToStroke(draft: TextDraft): TextStroke {
  return {
    id: draft.id ?? "draft",
    kind: "text",
    at: draft.at,
    text: draft.text,
    style: draft.style,
  };
}

function selectedEditableStroke(
  selected: SelectedObject | null,
  draft: TextDraft | null,
  preview: EditableStroke | null,
  strokes: Stroke[],
): EditableStroke | null {
  if (!selected) return null;
  if (draft?.id === selected.id) return textDraftToStroke(draft);
  if (preview?.id === selected.id) return preview;
  return (
    strokes.find((s): s is EditableStroke =>
      selected.kind === "text"
        ? s.kind === "text" && s.id === selected.id
        : isShapeStroke(s) && s.id === selected.id,
    ) ?? null
  );
}

function sameEditableStroke(a: EditableStroke, b: EditableStroke): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function translateEditableStrokeWithinBounds(
  stroke: EditableStroke,
  dx: number,
  dy: number,
  dims: { width: number; height: number },
): EditableStroke {
  const b = strokeBounds(stroke, dims);
  const minDx = -b.x0;
  const maxDx = dims.width - b.x1;
  const minDy = -b.y0;
  const maxDy = dims.height - b.y1;
  const tx = clamp(dx, minDx, maxDx);
  const ty = clamp(dy, minDy, maxDy);
  if (stroke.kind === "text") {
    return {
      ...stroke,
      at: [stroke.at[0] + tx, stroke.at[1] + ty],
    };
  }
  return {
    ...stroke,
    a: [stroke.a[0] + tx, stroke.a[1] + ty],
    b: [stroke.b[0] + tx, stroke.b[1] + ty],
  };
}

function strokeFullyWithinBounds(
  stroke: EditableStroke,
  dims: { width: number; height: number },
): boolean {
  const b = strokeBounds(stroke, dims);
  return b.x0 >= 0 && b.y0 >= 0 && b.x1 <= dims.width && b.y1 <= dims.height;
}

function strokeBounds(
  stroke: EditableStroke,
  _dims: { width: number; height: number },
): { x0: number; y0: number; x1: number; y1: number } {
  if (stroke.kind === "text") {
    const metrics = measureTextStroke(stroke);
    return {
      x0: stroke.at[0],
      y0: stroke.at[1],
      x1: stroke.at[0] + metrics.width,
      y1: stroke.at[1] + metrics.height,
    };
  }
  const pad = Math.max(1, stroke.style.width / 2);
  if (stroke.kind === "ellipse" || stroke.kind === "rect") {
    return {
      x0: Math.min(stroke.a[0], stroke.b[0]) - pad,
      y0: Math.min(stroke.a[1], stroke.b[1]) - pad,
      x1: Math.max(stroke.a[0], stroke.b[0]) + pad,
      y1: Math.max(stroke.a[1], stroke.b[1]) + pad,
    };
  }
  const head =
    stroke.kind === "arrow"
      ? Math.min(Math.max(stroke.style.width * 3.5, 8), 28)
      : 0;
  const extra = pad + head;
  return {
    x0: Math.min(stroke.a[0], stroke.b[0]) - extra,
    y0: Math.min(stroke.a[1], stroke.b[1]) - extra,
    x1: Math.max(stroke.a[0], stroke.b[0]) + extra,
    y1: Math.max(stroke.a[1], stroke.b[1]) + extra,
  };
}

function shapeHitTest(stroke: ShapeStroke, x: number, y: number): boolean {
  const radius = Math.max(8, stroke.style.width + 5);
  if (stroke.kind === "rect") {
    const x0 = Math.min(stroke.a[0], stroke.b[0]);
    const y0 = Math.min(stroke.a[1], stroke.b[1]);
    const x1 = Math.max(stroke.a[0], stroke.b[0]);
    const y1 = Math.max(stroke.a[1], stroke.b[1]);
    if (
      stroke.style.fill &&
      x >= x0 &&
      x <= x1 &&
      y >= y0 &&
      y <= y1
    ) {
      return true;
    }
    return (
      pointSegDistSq(x, y, x0, y0, x1, y0) <= radius * radius ||
      pointSegDistSq(x, y, x1, y0, x1, y1) <= radius * radius ||
      pointSegDistSq(x, y, x1, y1, x0, y1) <= radius * radius ||
      pointSegDistSq(x, y, x0, y1, x0, y0) <= radius * radius
    );
  }
  if (stroke.kind === "line" || stroke.kind === "arrow") {
    return (
      pointSegDistSq(x, y, stroke.a[0], stroke.a[1], stroke.b[0], stroke.b[1]) <=
      radius * radius
    );
  }
  const cx = (stroke.a[0] + stroke.b[0]) / 2;
  const cy = (stroke.a[1] + stroke.b[1]) / 2;
  const rx = Math.abs(stroke.b[0] - stroke.a[0]) / 2;
  const ry = Math.abs(stroke.b[1] - stroke.a[1]) / 2;
  if (rx <= 0 || ry <= 0) return false;
  const nx = (x - cx) / rx;
  const ny = (y - cy) / ry;
  const d = Math.hypot(nx, ny);
  if (stroke.style.fill && d <= 1) return true;
  const avgR = (rx + ry) / 2;
  return Math.abs(d - 1) <= radius / Math.max(avgR, 1);
}

let textMeasureCanvas: HTMLCanvasElement | null = null;
function measureTextStroke(stroke: TextStroke): { width: number; height: number } {
  if (textMeasureCanvas === null) {
    textMeasureCanvas = document.createElement("canvas");
  }
  const ctx = textMeasureCanvas.getContext("2d");
  const fontSize = stroke.style.fontSize;
  if (!ctx) {
    return {
      width: fontSize * Math.max(stroke.text.length, 1) * 0.6,
      height: fontSize * 1.15,
    };
  }
  ctx.font = `600 ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif`;
  return {
    width: Math.max(1, ctx.measureText(stroke.text || " ").width),
    height: fontSize * 1.15,
  };
}

function controlsPlacementForSelection(
  selected: SelectedObject,
  draft: TextDraft | null,
  preview: EditableStroke | null,
  strokes: Stroke[],
  imageToScreen: (p: Point2) => [number, number],
  dims: { width: number; height: number },
): { left: number; top: number } | null {
  const stroke = selectedEditableStroke(selected, draft, preview, strokes);
  if (!stroke) return null;
  const b = strokeBounds(stroke, dims);
  const [sx0] = imageToScreen([b.x0, b.y1]);
  const [sx1, sy1] = imageToScreen([b.x1, b.y1]);
  return { left: (sx0 + sx1 - SELECTION_CONTROLS_W) / 2, top: sy1 + 8 };
}

function clampControlsPlacement(
  placement: { left: number; top: number },
  rect: Rect,
): { left: number; top: number } {
  const height = SELECTION_CONTROL_SIZE;
  const maxLeft = Math.max(rect.x, rect.x + rect.w - SELECTION_CONTROLS_W);
  const maxTop = Math.max(rect.y, rect.y + rect.h - height);
  return {
    left: clamp(placement.left, rect.x, maxLeft),
    top: clamp(placement.top, rect.y, maxTop),
  };
}

function hitTargetStyleForStroke(
  stroke: EditableStroke,
  imageToScreen: (p: Point2) => [number, number],
  dims: { width: number; height: number },
  rect: Rect,
): CSSProperties | null {
  const b = strokeBounds(stroke, dims);
  if (![b.x0, b.y0, b.x1, b.y1].every(Number.isFinite)) return null;
  const [sx0, sy0] = imageToScreen([b.x0, b.y0]);
  const [sx1, sy1] = imageToScreen([b.x1, b.y1]);
  const pad = stroke.kind === "text" ? 4 : 6;
  const minSize = stroke.kind === "text" ? 10 : 14;
  const cx = (sx0 + sx1) / 2;
  const cy = (sy0 + sy1) / 2;
  const width = Math.max(Math.abs(sx1 - sx0) + pad * 2, minSize);
  const height = Math.max(Math.abs(sy1 - sy0) + pad * 2, minSize);
  const left = clamp(cx - width / 2, rect.x, rect.x + rect.w);
  const top = clamp(cy - height / 2, rect.y, rect.y + rect.h);
  const right = clamp(cx + width / 2, rect.x, rect.x + rect.w);
  const bottom = clamp(cy + height / 2, rect.y, rect.y + rect.h);
  if (right <= left || bottom <= top) return null;
  return {
    position: "absolute",
    left,
    top,
    width: right - left,
    height: bottom - top,
    background: "transparent",
    cursor: "pointer",
    zIndex: 13,
    pointerEvents: "auto",
  };
}

function selectionControlButtonStyle(
  cursor: CSSProperties["cursor"],
): CSSProperties {
  return {
    position: "relative",
    width: SELECTION_CONTROL_SIZE,
    height: SELECTION_CONTROL_SIZE,
    minWidth: SELECTION_CONTROL_SIZE,
    minHeight: SELECTION_CONTROL_SIZE,
    padding: 0,
    border: "none",
    borderRadius: 9999,
    color: "rgba(255, 255, 255, 0.95)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
    pointerEvents: "auto",
    cursor,
    transition: "none",
    transform: "none",
  };
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
    if (stroke.style.fill) ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
  } else if (stroke.kind === "ellipse") {
    const cx = (ax + bx) / 2;
    const cy = (ay + by) / 2;
    const rx = Math.abs(bx - ax) / 2;
    const ry = Math.abs(by - ay) / 2;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    if (stroke.style.fill) ctx.fill();
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
    if (
      stroke.style.fill &&
      x >= x0 - radius &&
      x <= x1 + radius &&
      y >= y0 - radius &&
      y <= y1 + radius
    ) {
      return true;
    }
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
    if (stroke.style.fill && d <= 1 + radius / Math.max((rx + ry) / 2, 1)) {
      return true;
    }
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
