import { useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  Pencil,
  Eraser,
  Type,
  Shapes,
  Pipette,
  Square,
  Circle,
  Minus,
  MoveRight,
  Undo2,
  X,
  Check,
} from "lucide-react";
import {
  EDIT_AFFORDANCE_SIZE,
  EDIT_COLORS,
  EDIT_ERASER_WIDTHS,
  EDIT_PILL_HORIZONTAL_LEN,
  EDIT_PILL_VERTICAL_LEN,
  EDIT_STROKE_WIDTHS,
  EDIT_TEXT_SIZES,
  type EditAnchor,
  type Shape,
  type Tool,
} from "../lib/editTypes";
import type { EditController } from "../lib/useEditController";
import { BlurredBackdrop, SvgInsetBorder, TOOLBAR_FROST } from "./Frosted";

const TOOL_ICON: Record<Tool, typeof Pencil> = {
  marker: Pencil,
  eraser: Eraser,
  text: Type,
  shape: Shapes,
  colorpicker: Pipette,
};

const TOOL_LABEL: Record<Tool, string> = {
  marker: "Marker",
  eraser: "Eraser",
  text: "Text",
  shape: "Shapes",
  colorpicker: "Color picker",
};

const TOOLS: Tool[] = ["marker", "eraser", "text", "shape", "colorpicker"];

const SHAPE_ICON: Record<Shape, typeof Square> = {
  rect: Square,
  ellipse: Circle,
  line: Minus,
  arrow: MoveRight,
};

const SHAPE_LABEL: Record<Shape, string> = {
  rect: "Rectangle",
  ellipse: "Ellipse",
  line: "Line",
  arrow: "Arrow",
};

const PILL_RADIUS = 9999;
const POPOVER_RADIUS = 14;
// Distance between the pill and the popover. Tight on purpose — the
// popover should read as "extending from" the specific tool button.
const POPOVER_GAP = 4;
// Pill internal layout: must mirror the values used by ExpandedContents
// so the popover can compute the active tool button's exact center for
// alignment.
const PILL_INNER_PAD = 5;
const PILL_TOOL_SIZE = 26;
const PILL_TOOL_GAP = 4;
const PILL_CONTENT_COLLAPSE_MS = 60;
const PILL_GEOMETRY_MS = 180;
const TOOL_LAYOUT_INDEX: Record<Tool, number> = {
  marker: 0,
  eraser: 1,
  text: 2,
  shape: 3,
  colorpicker: 4,
};

type PlacementBox = { x: number; y: number; w: number; h: number };

export type EditAffordanceProps = {
  ctl: EditController;
  anchor: EditAnchor;
  /// Used by the BlurredBackdrop inside the pill — the original full-screen
  /// PNG so the frost samples real pixels rather than the dim backdrop layer.
  screenPngB64: string;
  screenW: number;
  screenH: number;
  /// Boxes (in viewport coords) the popover should avoid (toolbar, chat
  /// panel). Used to flip the popover to the opposite side of the pill when
  /// the natural side would overlap.
  avoidBoxes?: ReadonlyArray<PlacementBox>;
};

export default function EditAffordance(props: EditAffordanceProps) {
  const { ctl, anchor } = props;
  if (anchor.hidden) return null;

  // Render the popover whenever the pill is open and a tool is selected,
  // even when `popoverOpen` is false. We pass `visible` through so the
  // popover can run a CSS fade-out animation before the (next) unmount —
  // necessary so that pointer-down on the canvas (which sets popoverOpen
  // to false to hide the panel while the user draws) produces a smooth
  // fade rather than an abrupt disappearance.
  return (
    <>
      <EditPill {...props} />
      {ctl.open && ctl.tool && (
        <EditPopover {...props} visible={ctl.popoverOpen} />
      )}
    </>
  );
}

/// The pill: collapsed (32 px circle, pencil icon) ↔ expanded row/column of
/// tool buttons. Both states are always rendered so we can cross-fade between
/// them while the wrapper grows. Horizontal axis (top/bottom edge) ⇒ width
/// grows; vertical axis (left/right edge) ⇒ height grows.
function EditPill({
  ctl,
  anchor,
  screenPngB64,
  screenW,
  screenH,
}: EditAffordanceProps) {
  const horizontal = anchor.axis === "horizontal";
  const collapsedSize = EDIT_AFFORDANCE_SIZE;
  const expandedLen = horizontal ? EDIT_PILL_HORIZONTAL_LEN : EDIT_PILL_VERTICAL_LEN;
  const [expanded, setExpanded] = useState(ctl.open);
  const [contentsVisible, setContentsVisible] = useState(ctl.open);
  const [iconVisible, setIconVisible] = useState(!ctl.open);

  useEffect(() => {
    let frame = 0;
    let collapseTimer = 0;
    let iconTimer = 0;
    if (ctl.open) {
      setIconVisible(false);
      setExpanded(true);
      frame = window.requestAnimationFrame(() => setContentsVisible(true));
    } else {
      setContentsVisible(false);
      collapseTimer = window.setTimeout(() => {
        setExpanded(false);
        iconTimer = window.setTimeout(
          () => setIconVisible(true),
          PILL_GEOMETRY_MS,
        );
      }, PILL_CONTENT_COLLAPSE_MS);
    }
    return () => {
      if (frame) window.cancelAnimationFrame(frame);
      if (collapseTimer) window.clearTimeout(collapseTimer);
      if (iconTimer) window.clearTimeout(iconTimer);
    };
  }, [ctl.open]);

  const isStart = anchor.expandToward === "start";
  const pillLen = expanded ? expandedLen : collapsedSize;
  const pillW = horizontal ? pillLen : collapsedSize;
  const pillH = horizontal ? collapsedSize : pillLen;
  const fullyCollapsed = !expanded && iconVisible;

  // The wrap is just a positioning anchor pinned to the icon's home — its
  // left/top track the rect-drag anchor with no transition, so the pill
  // doesn't lag behind during a resize. The actual expansion lives on the
  // inner pill, which is anchored to the wrap's icon-side edge and animates
  // only `width` (or `height`). Avoiding `transform` here matters because
  // WebKit composites transforms on the GPU while width/height animate on
  // the main thread; mixing the two desyncs each frame, and the icon visibly
  // drifts then snaps back during the 180 ms expansion.
  const pillSideAnchor: React.CSSProperties = horizontal
    ? { top: 0, [isStart ? "right" : "left"]: 0 }
    : { left: 0, [isStart ? "bottom" : "top"]: 0 };

  const stop = (event: React.MouseEvent) => event.stopPropagation();

  return (
    <div
      className="screenie-edit-pill-wrap"
      style={{
        position: "absolute",
        left: anchor.x,
        top: anchor.y,
        width: collapsedSize,
        height: collapsedSize,
        pointerEvents: "auto",
        zIndex: 12,
      }}
      onMouseDown={stop}
    >
      <div
        className="screenie-edit-pill"
        data-open={!fullyCollapsed}
        data-axis={anchor.axis}
        data-expand={anchor.expandToward}
        style={{
          position: "absolute",
          ...pillSideAnchor,
          width: pillW,
          height: pillH,
          borderRadius: PILL_RADIUS,
          overflow: "hidden",
          color: "rgba(255, 255, 255, 0.95)",
          cursor: fullyCollapsed ? "pointer" : "default",
          transition: `width ${PILL_GEOMETRY_MS}ms cubic-bezier(.2,.8,.2,1), height ${PILL_GEOMETRY_MS}ms cubic-bezier(.2,.8,.2,1), transform 140ms cubic-bezier(.2,.8,.2,1)`,
        }}
        onClick={(event) => {
          if (!fullyCollapsed) return;
          event.stopPropagation();
          ctl.setOpen(true);
        }}
      >
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
        <CollapsedIcon
          visible={iconVisible}
          horizontal={horizontal}
          expandToward={anchor.expandToward}
        />
        <ExpandedContents
          ctl={ctl}
          visible={contentsVisible}
          horizontal={horizontal}
          expandToward={anchor.expandToward}
        />
        <SvgInsetBorder radius={PILL_RADIUS} />
      </div>
    </div>
  );
}

function CollapsedIcon({
  visible,
  horizontal,
  expandToward,
}: {
  visible: boolean;
  horizontal: boolean;
  expandToward: "start" | "end";
}) {
  // Pin the icon to the side of the wrap that corresponds to its collapsed
  // anchor, instead of centering it. As the pill expands, the wrap grows
  // away from the icon — the icon stays put while it fades. Centering would
  // cause the icon to slide toward the new center as the wrap widens, which
  // reads as a janky drift during the expansion.
  const isStart = expandToward === "start";
  const sideStyles: React.CSSProperties = horizontal
    ? { left: isStart ? "auto" : 0, right: isStart ? 0 : "auto", top: 0, bottom: 0 }
    : { top: isStart ? "auto" : 0, bottom: isStart ? 0 : "auto", left: 0, right: 0 };
  return (
    <div
      className="screenie-edit-pill-icon"
      data-visible={visible}
      aria-hidden={!visible}
      style={{
        position: "absolute",
        ...sideStyles,
        width: horizontal ? EDIT_AFFORDANCE_SIZE : undefined,
        height: horizontal ? undefined : EDIT_AFFORDANCE_SIZE,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "inherit",
        zIndex: 1,
        // The wrapper itself catches the click (so the user can hit anywhere
        // in the 32 px circle), so the icon is purely decorative.
        pointerEvents: "none",
      }}
    >
      <Pencil size={14} strokeWidth={1.85} aria-hidden />
    </div>
  );
}

function ExpandedContents({
  ctl,
  visible,
  horizontal,
  expandToward,
}: {
  ctl: EditController;
  visible: boolean;
  horizontal: boolean;
  expandToward: "start" | "end";
}) {
  const onPickTool = (event: React.MouseEvent, t: Tool) => {
    event.stopPropagation();
    // Color picker is a one-shot tool with no secondary controls — clicking
    // it just arms the cursor, no popover.
    const HAS_POPOVER = t !== "colorpicker";
    if (ctl.tool === t) {
      if (HAS_POPOVER) ctl.setPopoverOpen(!ctl.popoverOpen);
      return;
    }
    ctl.setTool(t);
    ctl.setPopoverOpen(HAS_POPOVER);
  };
  const onCollapse = (event: React.MouseEvent) => {
    event.stopPropagation();
    ctl.setOpen(false);
    ctl.setTool(null);
  };

  // When pill expands toward "start" the icon ends up at the trailing end of
  // the pill; mirror flex direction so the close button sits next to the
  // icon location.
  const reverse = expandToward === "start";
  const flexDir: React.CSSProperties["flexDirection"] = horizontal
    ? reverse
      ? "row-reverse"
      : "row"
    : reverse
      ? "column-reverse"
      : "column";

  return (
    <div
      className="screenie-edit-pill-row"
      data-visible={visible}
      data-axis={horizontal ? "horizontal" : "vertical"}
      aria-hidden={!visible}
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        flexDirection: flexDir,
        alignItems: "center",
        justifyContent: "space-between",
        // Equal 5 px breathing room on all 4 sides between the pill edge
        // and any tool button. With pill = 36 px and button = 26 px, the
        // vertical 5 px is automatic via align-items: center; we set
        // padding to lock the horizontal margins to the same value.
        padding: 5,
        zIndex: 1,
        pointerEvents: visible ? "auto" : "none",
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: flexDir,
          alignItems: "center",
          gap: 4,
        }}
      >
        {TOOLS.map((t) => {
          const Icon = TOOL_ICON[t];
          const active = ctl.tool === t;
          return (
            <button
              key={t}
              type="button"
              className="screenie-edit-tool"
              data-active={active}
              aria-label={TOOL_LABEL[t]}
              tabIndex={visible ? 0 : -1}
              onClick={(e) => onPickTool(e, t)}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <Icon size={14} strokeWidth={1.85} aria-hidden />
            </button>
          );
        })}
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: flexDir,
          alignItems: "center",
          gap: 4,
        }}
      >
        <button
          type="button"
          className="screenie-edit-tool"
          aria-label="Undo"
          disabled={!ctl.canUndo}
          tabIndex={visible ? 0 : -1}
          onClick={(e) => {
            e.stopPropagation();
            ctl.undo();
          }}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <Undo2 size={14} strokeWidth={1.85} aria-hidden />
        </button>
        <button
          type="button"
          className="screenie-edit-tool"
          aria-label="Close edit toolbar"
          tabIndex={visible ? 0 : -1}
          onClick={onCollapse}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <X size={13} strokeWidth={2} aria-hidden />
        </button>
      </div>
    </div>
  );
}

/// Per-tool secondary controls. Sized to its content; positions itself on the
/// side of the pill that is opposite the prompt toolbar (so it never
/// overlaps the toolbar). Falls back to clamped placement near the screen
/// edge when both sides would collide. Uses the same frost recipe as the
/// pill (and toolbar / chat panel) for visual consistency.
function EditPopover({
  ctl,
  anchor,
  screenPngB64,
  screenW,
  screenH,
  avoidBoxes,
  visible = true,
}: EditAffordanceProps & { visible?: boolean }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  // Measure the popover content so we can position it next to the pill on
  // whichever side has room.
  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const update = () => {
      const r = el.getBoundingClientRect();
      setSize((prev) => {
        if (Math.abs(prev.w - r.width) < 0.5 && Math.abs(prev.h - r.height) < 0.5) {
          return prev;
        }
        return { w: r.width, h: r.height };
      });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // The pill's expanded bounding box. For a horizontal pill it grows along
  // x; for a vertical pill it grows along y. `expandToward` decides whether
  // the icon is at the leading or trailing end.
  const horizontal = anchor.axis === "horizontal";
  const collapsedSize = EDIT_AFFORDANCE_SIZE;
  const expandedLen = horizontal ? EDIT_PILL_HORIZONTAL_LEN : EDIT_PILL_VERTICAL_LEN;
  const pillW = horizontal ? expandedLen : collapsedSize;
  const pillH = horizontal ? collapsedSize : expandedLen;
  const pillLeft = horizontal
    ? anchor.expandToward === "start"
      ? anchor.x + collapsedSize - expandedLen
      : anchor.x
    : anchor.x;
  const pillTop = horizontal
    ? anchor.y
    : anchor.expandToward === "start"
      ? anchor.y + collapsedSize - expandedLen
      : anchor.y;
  const pillRight = pillLeft + pillW;
  const pillBottom = pillTop + pillH;
  const pillCenterY = pillTop + pillH / 2;

  // Center of the *active tool button* inside the pill. The popover anchors
  // to this point so it visibly extends from the specific button the user
  // clicked rather than from the whole pill. Layout values must match
  // those used by ExpandedContents above.
  const toolIdx = ctl.tool ? TOOL_LAYOUT_INDEX[ctl.tool] : 0;
  const toolOffset =
    PILL_INNER_PAD + toolIdx * (PILL_TOOL_SIZE + PILL_TOOL_GAP) + PILL_TOOL_SIZE / 2;
  const activeToolX = horizontal
    ? anchor.expandToward === "end"
      ? pillLeft + toolOffset
      : pillLeft + pillW - toolOffset
    : pillLeft + pillW / 2;
  const activeToolY = horizontal
    ? pillTop + pillH / 2
    : anchor.expandToward === "end"
      ? pillTop + toolOffset
      : pillTop + pillH - toolOffset;

  const popW = size.w || 240;
  const popH = size.h || 120;
  const PAD = 8;

  const overlapsAvoid = (b: PlacementBox): boolean =>
    (avoidBoxes ?? []).some(
      (box) =>
        !(
          b.x + b.w + PAD <= box.x ||
          b.x >= box.x + box.w + PAD ||
          b.y + b.h + PAD <= box.y ||
          b.y >= box.y + box.h + PAD
        ),
    );

  const fitsOnScreen = (b: PlacementBox): boolean =>
    b.x >= PAD &&
    b.x + b.w <= screenW - PAD &&
    b.y >= PAD &&
    b.y + b.h <= screenH - PAD;

  let chosen: PlacementBox;

  if (horizontal) {
    // Horizontal pill — popover goes ABOVE or BELOW (away from the toolbar).
    // Horizontally, center the popover on the *active tool button* so it
    // reads as expanding from that exact button. If centering would clip
    // off-screen, the clamp shifts the popover toward whichever side has
    // more room — which lines up with the user's "expand into the roomier
    // direction" rule.
    const popX = clampNum(
      activeToolX - popW / 2,
      PAD,
      Math.max(PAD, screenW - popW - PAD),
    );

    const aboveBox: PlacementBox = {
      x: popX,
      y: pillTop - POPOVER_GAP - popH,
      w: popW,
      h: popH,
    };
    const belowBox: PlacementBox = {
      x: popX,
      y: pillBottom + POPOVER_GAP,
      w: popW,
      h: popH,
    };
    const aboveBad = !fitsOnScreen(aboveBox) || overlapsAvoid(aboveBox);
    const belowBad = !fitsOnScreen(belowBox) || overlapsAvoid(belowBox);

    if (!aboveBad && !belowBad) {
      const toolbar = avoidBoxes && avoidBoxes[0];
      if (toolbar) {
        const toolbarMidY = toolbar.y + toolbar.h / 2;
        chosen = pillCenterY > toolbarMidY ? aboveBox : belowBox;
      } else {
        chosen = belowBox;
      }
    } else if (!aboveBad) {
      chosen = aboveBox;
    } else if (!belowBad) {
      chosen = belowBox;
    } else {
      chosen = fitsOnScreen(aboveBox) ? aboveBox : belowBox;
    }
  } else {
    // Vertical pill — popover goes LEFT or RIGHT, biased toward the side
    // with more horizontal room. This is the user's "expand toward more
    // space" rule for the L/R edge case. Vertically, center the popover on
    // the active tool button so it reads as expanding from that button.
    const leftRoom = pillLeft - PAD;
    const rightRoom = screenW - pillRight - PAD;
    const popY = clampNum(
      activeToolY - popH / 2,
      PAD,
      Math.max(PAD, screenH - popH - PAD),
    );
    const leftBox: PlacementBox = {
      x: pillLeft - POPOVER_GAP - popW,
      y: popY,
      w: popW,
      h: popH,
    };
    const rightBox: PlacementBox = {
      x: pillRight + POPOVER_GAP,
      y: popY,
      w: popW,
      h: popH,
    };
    const leftBad = !fitsOnScreen(leftBox) || overlapsAvoid(leftBox);
    const rightBad = !fitsOnScreen(rightBox) || overlapsAvoid(rightBox);

    if (!leftBad && !rightBad) {
      chosen = rightRoom >= leftRoom ? rightBox : leftBox;
    } else if (!leftBad) {
      chosen = leftBox;
    } else if (!rightBad) {
      chosen = rightBox;
    } else {
      chosen = rightRoom >= leftRoom ? rightBox : leftBox;
    }
  }

  return (
    <div
      ref={wrapRef}
      className="screenie-edit-popover-wrap"
      data-visible={visible}
      style={{
        position: "absolute",
        left: chosen.x,
        top: chosen.y,
        zIndex: 13,
        pointerEvents: visible ? "auto" : "none",
      }}
      onMouseDown={(e) => e.stopPropagation()}
    >
      <div
        className="screenie-edit-popover"
        style={{
          position: "relative",
          borderRadius: POPOVER_RADIUS,
          overflow: "hidden",
          padding: 14,
          color: "rgba(255, 255, 255, 0.96)",
          display: "flex",
          flexDirection: "column",
          gap: 12,
          minWidth: 220,
          maxWidth: Math.min(360, screenW - PAD * 2),
        }}
      >
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
        <div
          style={{
            position: "relative",
            zIndex: 1,
            display: "flex",
            flexDirection: "column",
            gap: 12,
          }}
        >
          {ctl.tool === "marker" && <MarkerControls ctl={ctl} />}
          {ctl.tool === "eraser" && <EraserControls ctl={ctl} />}
          {ctl.tool === "shape" && <ShapeControls ctl={ctl} />}
          {ctl.tool === "text" && <TextControls ctl={ctl} />}
        </div>
        <SvgInsetBorder radius={POPOVER_RADIUS} />
      </div>
    </div>
  );
}

function MarkerControls({ ctl }: { ctl: EditController }) {
  return (
    <>
      <ColorRow
        label="Color"
        value={ctl.marker.color}
        onChange={(c) => ctl.setMarker({ color: c })}
      />
      <WidthRow
        label="Size"
        value={ctl.marker.width}
        options={EDIT_STROKE_WIDTHS as unknown as number[]}
        onChange={(w) => ctl.setMarker({ width: w })}
      />
    </>
  );
}

function EraserControls({ ctl }: { ctl: EditController }) {
  return (
    <>
      <div className="screenie-edit-popover-row">
        <span className="screenie-edit-popover-label">Mode</span>
        <div className="screenie-edit-segment">
          <button
            type="button"
            className="screenie-edit-segment-btn"
            data-active={ctl.eraserMode === "pixel"}
            onClick={() => ctl.setEraserMode("pixel")}
          >
            Pixel
          </button>
          <button
            type="button"
            className="screenie-edit-segment-btn"
            data-active={ctl.eraserMode === "object"}
            onClick={() => ctl.setEraserMode("object")}
          >
            Object
          </button>
        </div>
      </div>
      {/* Single unified size row — same value (`ctl.eraserWidth`) and same
          label across both modes, so flipping the mode doesn't reset or
          rename the size. */}
      <WidthRow
        label="Eraser size"
        value={ctl.eraserWidth}
        options={EDIT_ERASER_WIDTHS as unknown as number[]}
        onChange={(w) => ctl.setEraserWidth(w)}
      />
    </>
  );
}

function ShapeControls({ ctl }: { ctl: EditController }) {
  const shapes: Shape[] = ["rect", "ellipse", "line", "arrow"];
  return (
    <>
      <div className="screenie-edit-popover-row">
        <span className="screenie-edit-popover-label">Shape</span>
        <div style={{ display: "flex", gap: 6 }}>
          {shapes.map((s) => {
            const Icon = SHAPE_ICON[s];
            const active = ctl.shape.kind === s;
            return (
              <button
                key={s}
                type="button"
                className="screenie-edit-tool screenie-edit-tool-md"
                data-active={active}
                aria-label={SHAPE_LABEL[s]}
                onClick={() => ctl.setShape({ kind: s })}
              >
                <Icon size={14} strokeWidth={1.85} aria-hidden />
              </button>
            );
          })}
        </div>
      </div>
      <ColorRow
        label="Color"
        value={ctl.shape.style.color}
        onChange={(c) => ctl.setShape({ style: { color: c } })}
      />
      <WidthRow
        label="Stroke"
        value={ctl.shape.style.width}
        options={EDIT_STROKE_WIDTHS as unknown as number[]}
        onChange={(w) => ctl.setShape({ style: { width: w } })}
      />
    </>
  );
}

function TextControls({ ctl }: { ctl: EditController }) {
  return (
    <>
      <ColorRow
        label="Color"
        value={ctl.text.color}
        onChange={(c) => ctl.setText({ color: c })}
      />
      <WidthRow
        label="Size"
        value={ctl.text.fontSize}
        options={EDIT_TEXT_SIZES as unknown as number[]}
        onChange={(s) => ctl.setText({ fontSize: s })}
      />
    </>
  );
}

function ColorRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (c: string) => void;
}) {
  return (
    <div className="screenie-edit-popover-row">
      <span className="screenie-edit-popover-label">{label}</span>
      <div style={{ display: "flex", gap: 6 }}>
        {EDIT_COLORS.map((c) => (
          <button
            key={c}
            type="button"
            className="screenie-edit-swatch"
            data-active={value.toLowerCase() === c.toLowerCase()}
            aria-label={`Color ${c}`}
            onClick={() => onChange(c)}
            style={{ background: c }}
          >
            {value.toLowerCase() === c.toLowerCase() && (
              <Check
                size={11}
                strokeWidth={3}
                aria-hidden
                style={{ color: contrastInk(c) }}
              />
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

function WidthRow({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: number;
  options: number[];
  onChange: (w: number) => void;
}) {
  return (
    <div className="screenie-edit-popover-row">
      <span className="screenie-edit-popover-label">{label}</span>
      <div style={{ display: "flex", gap: 6 }}>
        {options.map((w) => (
          <button
            key={w}
            type="button"
            className="screenie-edit-tool screenie-edit-tool-md"
            data-active={value === w}
            aria-label={`${label} ${w}`}
            onClick={() => onChange(w)}
          >
            <span
              style={{
                display: "inline-block",
                width: Math.min(w, 16),
                height: Math.min(w, 16),
                borderRadius: 999,
                background: "currentColor",
              }}
            />
          </button>
        ))}
      </div>
    </div>
  );
}

function clampNum(n: number, lo: number, hi: number): number {
  return Math.min(Math.max(n, lo), Math.max(lo, hi));
}

function contrastInk(hex: string): string {
  const v = hex.startsWith("#") ? hex.slice(1) : hex;
  if (v.length !== 6) return "rgba(0,0,0,0.85)";
  const r = parseInt(v.slice(0, 2), 16);
  const g = parseInt(v.slice(2, 4), 16);
  const b = parseInt(v.slice(4, 6), 16);
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return lum > 0.6 ? "rgba(0,0,0,0.85)" : "rgba(255,255,255,0.95)";
}
