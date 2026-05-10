import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_MARKER_STYLE,
  DEFAULT_SHAPE_STYLE,
  DEFAULT_TEXT_STYLE,
  EDIT_ERASER_WIDTHS,
  type EraserMode,
  type MarkerStroke,
  type EraserStroke,
  type ShapeStroke,
  type Shape,
  type Stroke,
  type StrokeStyle,
  type TextStroke,
  type Tool,
} from "./editTypes";

/// Centralized state for the on-image editor. Lifted to the Overlay root so
/// strokes survive the `adjusting → result` transition and the editor's open
/// state survives an explicit Esc-to-collapse without losing user work.
export type EditController = {
  // Stroke history.
  strokes: Stroke[];
  hasStrokes: boolean;
  past: Stroke[][];
  canUndo: boolean;

  // UI state.
  open: boolean;          // pill expanded vs collapsed icon
  tool: Tool | null;      // null = no active tool (rect drag works normally)
  popoverOpen: boolean;   // tool's secondary controls visible

  // Per-tool styles.
  marker: StrokeStyle;
  eraserWidth: number;
  eraserMode: EraserMode;
  shape: { kind: Shape; style: StrokeStyle };
  text: StrokeStyle & { fontSize: number };

  // Trim notification (for the "N marks were trimmed" toast).
  trimmedNotice: { count: number; key: number } | null;
  // Last sampled colour (from the eyedropper tool); displayed as a transient
  // toast and copied to the clipboard.
  pickedColor: { hex: string; key: number } | null;
  setPickedColor: (next: { hex: string; key: number } | null) => void;

  // Mutators.
  setOpen: (next: boolean) => void;
  setTool: (next: Tool | null) => void;
  setPopoverOpen: (next: boolean) => void;
  setMarker: (next: Partial<StrokeStyle>) => void;
  setEraserWidth: (next: number) => void;
  setEraserMode: (next: EraserMode) => void;
  setShape: (next: { kind?: Shape; style?: Partial<StrokeStyle> }) => void;
  setText: (next: Partial<StrokeStyle & { fontSize: number }>) => void;
  addStroke: (s: Stroke) => void;
  removeStrokes: (ids: ReadonlySet<string>) => void;
  undo: () => void;
  clear: () => void;
  remapForCrop: (
    oldRect: { x: number; y: number; w: number; h: number },
    newRect: { x: number; y: number; w: number; h: number },
    oldDims: { width: number; height: number },
    newDims: { width: number; height: number },
  ) => void;
  dismissTrimmedNotice: () => void;
};

const UNDO_LIMIT = 25;

let _idCounter = 0;
export function nextStrokeId(): string {
  _idCounter += 1;
  return `s${Date.now().toString(36)}_${_idCounter}`;
}

export function useEditController(): EditController {
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [past, setPast] = useState<Stroke[][]>([]);
  const [open, setOpenState] = useState(false);
  const [tool, setToolState] = useState<Tool | null>(null);
  const [popoverOpen, setPopoverOpenState] = useState(false);
  const [marker, setMarkerState] = useState<StrokeStyle>(DEFAULT_MARKER_STYLE);
  const [eraserWidth, setEraserWidthState] = useState<number>(EDIT_ERASER_WIDTHS[1]);
  const [eraserMode, setEraserModeState] = useState<EraserMode>("pixel");
  const [shape, setShapeState] = useState<{ kind: Shape; style: StrokeStyle }>({
    kind: "rect",
    style: DEFAULT_SHAPE_STYLE,
  });
  const [text, setTextState] = useState<StrokeStyle & { fontSize: number }>(
    DEFAULT_TEXT_STYLE,
  );
  const [trimmedNotice, setTrimmedNotice] = useState<{
    count: number;
    key: number;
  } | null>(null);
  const [pickedColor, setPickedColor] = useState<{
    hex: string;
    key: number;
  } | null>(null);

  const setOpen = useCallback((next: boolean) => {
    setOpenState(next);
    if (!next) setPopoverOpenState(false);
  }, []);

  const setTool = useCallback((next: Tool | null) => {
    setToolState(next);
    setPopoverOpenState(false);
  }, []);

  const setPopoverOpen = useCallback((next: boolean) => {
    setPopoverOpenState(next);
  }, []);

  const setMarker = useCallback((next: Partial<StrokeStyle>) => {
    setMarkerState((prev) => ({ ...prev, ...next }));
  }, []);

  const setEraserWidth = useCallback((next: number) => {
    setEraserWidthState(next);
  }, []);

  const setEraserMode = useCallback((next: EraserMode) => {
    setEraserModeState(next);
  }, []);

  const setShape = useCallback(
    (next: { kind?: Shape; style?: Partial<StrokeStyle> }) => {
      setShapeState((prev) => ({
        kind: next.kind ?? prev.kind,
        style: next.style ? { ...prev.style, ...next.style } : prev.style,
      }));
    },
    [],
  );

  const setText = useCallback((next: Partial<StrokeStyle & { fontSize: number }>) => {
    setTextState((prev) => ({ ...prev, ...next }));
  }, []);

  const addStroke = useCallback((s: Stroke) => {
    setStrokes((prev) => {
      setPast((p) => {
        const np = [...p, prev];
        return np.length > UNDO_LIMIT ? np.slice(np.length - UNDO_LIMIT) : np;
      });
      return [...prev, s];
    });
  }, []);

  /// Object-eraser commit path: drop every stroke whose id is in `ids`,
  /// pushing the prior list onto the undo stack as a single step.
  const removeStrokes = useCallback((ids: ReadonlySet<string>) => {
    if (ids.size === 0) return;
    setStrokes((prev) => {
      const next = prev.filter((s) => !ids.has(s.id));
      if (next.length === prev.length) return prev;
      setPast((p) => {
        const np = [...p, prev];
        return np.length > UNDO_LIMIT ? np.slice(np.length - UNDO_LIMIT) : np;
      });
      return next;
    });
  }, []);

  const undo = useCallback(() => {
    setPast((p) => {
      if (p.length === 0) return p;
      const last = p[p.length - 1];
      setStrokes(last);
      return p.slice(0, -1);
    });
  }, []);

  const clear = useCallback(() => {
    setStrokes([]);
    setPast([]);
    setTrimmedNotice(null);
    setPickedColor(null);
  }, []);

  const dismissTrimmedNotice = useCallback(() => setTrimmedNotice(null), []);

  // After the rect is dragged to a new region, translate every stroke from
  // the old cropped image-space → screenshot-space → new cropped image-space.
  // Strokes whose bounding box ends up entirely outside the new crop are
  // dropped. Partially-clipped strokes are kept (canvas clipping handles them).
  const remapForCrop = useCallback(
    (
      oldRect: { x: number; y: number; w: number; h: number },
      newRect: { x: number; y: number; w: number; h: number },
      oldDims: { width: number; height: number },
      newDims: { width: number; height: number },
    ) => {
      if (oldRect.w <= 0 || oldRect.h <= 0 || oldDims.width <= 0 || oldDims.height <= 0) {
        return;
      }
      // Old image-space → screenshot-space scale factors.
      const sxOld = oldRect.w / oldDims.width;
      const syOld = oldRect.h / oldDims.height;
      // Screenshot-space → new image-space scale factors.
      const sxNew = newDims.width / newRect.w;
      const syNew = newDims.height / newRect.h;

      const remapPt = (p: readonly [number, number]): [number, number] => {
        const screenX = oldRect.x + p[0] * sxOld;
        const screenY = oldRect.y + p[1] * syOld;
        return [(screenX - newRect.x) * sxNew, (screenY - newRect.y) * syNew];
      };

      let trimmed = 0;
      setStrokes((prev) => {
        const next: Stroke[] = [];
        for (const s of prev) {
          const remapped = remapStroke(s, remapPt);
          if (strokeIntersectsBounds(remapped, newDims.width, newDims.height)) {
            next.push(remapped);
          } else {
            trimmed += 1;
          }
        }
        if (trimmed > 0) {
          setPast((p) => [...p, prev].slice(-UNDO_LIMIT));
        }
        return next;
      });
      if (trimmed > 0) {
        setTrimmedNotice({ count: trimmed, key: Date.now() });
      }
    },
    [],
  );

  // Auto-dismiss the trimmed-notice toast after 1.8s.
  useEffect(() => {
    if (!trimmedNotice) return;
    const id = setTimeout(() => setTrimmedNotice(null), 1800);
    return () => clearTimeout(id);
  }, [trimmedNotice]);

  // Auto-dismiss the picked-colour toast after 1.6s.
  useEffect(() => {
    if (!pickedColor) return;
    const id = setTimeout(() => setPickedColor(null), 1600);
    return () => clearTimeout(id);
  }, [pickedColor]);

  // Stable refs so consumers can read latest state without re-subscribing.
  const ctlRef = useRef<EditController | null>(null);

  const ctl: EditController = {
    strokes,
    hasStrokes: strokes.length > 0,
    past,
    canUndo: past.length > 0,
    open,
    tool,
    popoverOpen,
    marker,
    eraserWidth,
    eraserMode,
    shape,
    text,
    trimmedNotice,
    pickedColor,
    setPickedColor,
    setOpen,
    setTool,
    setPopoverOpen,
    setMarker,
    setEraserWidth,
    setEraserMode,
    setShape,
    setText,
    addStroke,
    removeStrokes,
    undo,
    clear,
    remapForCrop,
    dismissTrimmedNotice,
  };
  ctlRef.current = ctl;
  return ctl;
}

function remapStroke(
  s: Stroke,
  remapPt: (p: readonly [number, number]) => [number, number],
): Stroke {
  switch (s.kind) {
    case "marker": {
      const r: MarkerStroke = { ...s, pts: s.pts.map(remapPt) };
      return r;
    }
    case "eraser": {
      const r: EraserStroke = { ...s, pts: s.pts.map(remapPt) };
      return r;
    }
    case "rect":
    case "ellipse":
    case "line":
    case "arrow": {
      const r: ShapeStroke = { ...s, a: remapPt(s.a), b: remapPt(s.b) };
      return r;
    }
    case "text": {
      const r: TextStroke = { ...s, at: remapPt(s.at) };
      return r;
    }
  }
}

function strokeIntersectsBounds(s: Stroke, w: number, h: number): boolean {
  // Bounding box test — keep stroke if any part lies inside [0,w] × [0,h].
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  const expand = (x: number, y: number) => {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  };
  if (s.kind === "marker" || s.kind === "eraser") {
    for (const [x, y] of s.pts) expand(x, y);
  } else if (s.kind === "text") {
    expand(s.at[0], s.at[1]);
    // Approximate text width by fontSize × 8 chars-worth — generous so we
    // don't drop labels that are partially in-bounds.
    expand(s.at[0] + s.style.fontSize * 8, s.at[1] + s.style.fontSize * 1.4);
  } else {
    expand(s.a[0], s.a[1]);
    expand(s.b[0], s.b[1]);
  }
  if (!Number.isFinite(minX)) return false;
  return !(maxX < 0 || minX > w || maxY < 0 || minY > h);
}
