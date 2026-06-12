# Perception layer — tool contract for the agent operating prompt

This was the insert the build spec's appendix called for; the operating
prompt now exists and carries this contract. The backend half is implemented
in `src-tauri/src/perception/` and exposed as the Tauri commands
`perception_see`, `perception_read`, `perception_read_sql`, `perception_look`,
`perception_resolve` (and as the in-process API
`perception::commands::{see_impl, read_impl, look_impl}` +
`perception::ids::resolve` for the agent loop).

## Agent-loop wiring (shipped)

- `read` and `look` are registry actions (`agent/actions.rs`, appended after
  `fail` — the schema enum order is byte-pinned). The prose rules live in
  `build_system_prompt` (`agent/planner.rs`); the blocks below remain the
  reference text.
- Model-facing fields: `read(text?, roles?, actionable_only?, limit?, snap?)`
  and `look(snap?)`. `region`/`focused_only` are deliberately NOT exposed to
  the model (raw coordinates are rejected by the action contract; both stay
  available on the `perception_read` command). `read_sql` stays a
  command-only power tool, never a model action.
- Acting on an element: `click`/`doubleClick`/`type` accept `eid` (+
  optional `snap`, defaulting to the latest read's snapshot) as an
  alternative to the observation `id`; `target_name` is still required. The
  executor resolves through `resolve_perception_target` and rewrites to the
  id form, so every existing gate (safety, confirmation, intent, preflight)
  applies unchanged.
- The loop perceives through the `PerceptionTools` seam
  (`agent/perception_tools.rs`); `read`'s SNAP block is rendered into the
  planner's goal context under "Element index", `look`'s marked PNG rides
  the captureFrame attachment path.
- `StaleSnapshot` on an eid action triggers a transparent re-read; the
  element is re-acquired in the fresh snapshot by fingerprint when exactly
  one row matches, otherwise the fresh block replaces the goal context and
  the planner re-picks with current ids.

```
### read — structured perception (default sense)
read(roles?, text?, region?, actionable_only?, focused_only?, limit?) → compact element lines
for the current frontmost window. Always fresh: stale snapshots are re-captured automatically.
Element IDs look like ax:B3 and are valid only with the SNAP id in the header — pass both to
any action. Lines ending in web⊥ are web-content boundaries: switch to the dom: channel for
anything inside them. If the footer says more elements exist, query again with filters instead
of asking for a screenshot.

### read_sql — power queries (when filters aren't enough)
read_sql(select_statement) → rows from the elements table (read-only, auto-LIMIT 200).
Schema: elements(snapshot_id, eid, fp, class, role, subrole, title, descr, value, actionable,
enabled, focused, x, y, w, h, depth, parent_eid, actions, is_web_boundary).

### look — annotated frame (vision fallback only)
look(snapshot_id?) → set-of-marks screenshot where every box label is a valid ax: element ID
from the same snapshot. Use only when read returns ax_empty or the target genuinely isn't in
the index; an ID found via look is actioned exactly like one found via read.
```

## Implementation notes the prompt author should know

- **Snapshot scoping (C4).** `resolve(eid, snapshot_id, allow_stale)` is the
  only way refs become click targets. A superseded or dirty snapshot returns
  a typed `StaleSnapshot`; the executor surfaces it as
  `ExecutionError::StaleSnapshot` — the correct reaction is re-`read`, not
  retry. `dom:`/`vis:` refs get a typed `WrongNamespace` pointing at the
  right channel.
- **Freshness (C3).** Snapshots are stamped with the AX change-counter value
  (the same AXObserver feed the settle path uses). `read`/`look` with no
  snapshot id, or with a latest-but-stale id, transparently re-run `see`.
  With no counter available the fallback is a 1.5s TTL.
- **Coordinates (C5).** Everything is global screen points, top-left origin
  (CGEvent space). `(x,y wxh)` in element lines is directly clickable via
  frame center; the only points→pixels conversion lives in the set-of-marks
  renderer.
- **Privacy (C6).** Secure-field values (and any descendant of a secure
  field) are `«redacted»` before they reach the index, FTS, serialization,
  or annotations. The index and PNGs live under
  `<app_data>/perception/{snapshots.db, frames/}` with retention of 20
  snapshots per window and a 24h hard expiry.
- **Grammar.** `SNAP <id> app=<bundle> win="…" <W>x<H>pt scale=<s> n=<total> t=<ms>`,
  optional `FOCUS <eid>`, element lines
  `<eid> <role-word> "<label>" (x,y wxh) [actions] val=… [disabled] [web⊥]`,
  footer `… N more — call read with role/text/region filters`. The `class`
  column / eid prefix letters: B button, T text input, L link, C toggle,
  M menu item, S slider, I image, G generic-actionable, X static.
- **Known deviations from the build spec** (deliberate, documented in phase
  commits): `Screen` scope walks all windows of the frontmost app rather
  than every on-screen app; `read_sql` ships enabled with
  `SCREENIE_PERCEPTION_SQL=0` as the off switch; AXStaticText skips the
  action-names fetch; the hit-test harness validates frame-center
  containment but does not auto-press elements (press-vs-click agreement
  needs supervised manual QA).
