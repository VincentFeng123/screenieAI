# Overlay Window Behavior Research

ScreenieAI's overlay target is a macOS-first non-activating HUD: React renders
the UI, while native AppKit/CoreGraphics code owns selective mouse behavior,
Spaces/fullscreen behavior, and consumed global shortcuts.

Current product decision: only interactive overlay regions are mouse-active.
Buttons, inputs, drag handles, capture rects, edit canvas, dropdowns, popovers,
toolbars, action pills, and chat panels disable passthrough. Transparent/empty
overlay regions re-enable passthrough so hover and click events go to Safari,
Finder, or whichever app is underneath.

## What Is Possible

- **Partial click-through is possible on macOS and is the overlay model.**
  AppKit exposes
  `NSWindow.ignoresMouseEvents`, which makes an entire window transparent to
  mouse events. Because Tauri/WebKit does not expose per-DOM-node native hit
  testing, ScreenieAI uses a measured-region bridge: React sends visible
  interactive rects to native code, and native code toggles
  `ignoresMouseEvents` based on whether the pointer is inside one of them.
- **First-click interaction is possible without programmatic focus.** A
  borderless transparent window needs panel-like defaults, and WebKit/AppKit
  views must accept the initial mouse down instead of treating it as only an
  activation click. ScreenieAI class-swaps the Tauri `NSWindow` and its content
  views to enable non-activating, first-click behavior.
- **A non-activating overlay can be key without making ScreenieAI the normal
  foreground app, but ScreenieAI does not force that.** The overlay is ordered
  front without `NSApp.activate`, Tauri `set_focus`, or `makeKeyAndOrderFront`.
- **Consuming Esc globally requires a CG event tap.** An `NSEvent` global monitor
  can observe keyboard events sent to other apps, but Apple documents that it
  cannot change or prevent those events. An active `CGEventTap` can delete the
  event by returning `NULL`, which is what prevents fullscreen Safari from also
  receiving Esc.

## What Is Not Perfect

- Without Accessibility permission, macOS may refuse the active keyboard event
  tap. ScreenieAI falls back to AppKit event monitors, which can still close the
  overlay in many cases but cannot guarantee suppression of Esc in the
  underlying app.
- Secure Keyboard Entry/password fields, games, remote desktops, and other
  protected input paths can block or bypass event taps. In those cases, the app
  must fall back to best-effort close behavior.
- Tauri's cross-platform window APIs cover transparent/always-on-top windows and
  whole-window cursor-event ignoring, but not this complete macOS HUD behavior.
  Native macOS code is required.

## macOS APIs Used

- `NSWindowStyleMaskNonactivatingPanel`: panel-style window that does not
  activate the owning app.
- `canBecomeKeyWindow = YES`, `canBecomeMainWindow = NO`: panel-compatible
  defaults for real user clicks, without programmatic focus.
- `acceptsFirstMouse`: installed on WebKit/AppKit content views so first mouse
  down is usable instead of only activating.
- `NSWindow.ignoresMouseEvents`: whole-window click-through switch, toggled by
  native hit testing against React-supplied interactive regions.
- `NSWindowCollectionBehaviorCanJoinAllSpaces`,
  `NSWindowCollectionBehaviorFullScreenAuxiliary`,
  `NSWindowCollectionBehaviorTransient`, `NSWindowCollectionBehaviorIgnoresCycle`:
  HUD-like Spaces/fullscreen/Mission Control behavior.
- `NSEvent` local monitor: consumes Esc while ScreenieAI receives the event.
- `NSEvent` global monitor: fallback observation only.
- `CGEventTapCreate` at `kCGSessionEventTap`: active session keyboard filter for
  consuming Esc while the overlay is visible.

## Permissions

- **Screen Recording:** required for screenshot capture and agentic visual
  verification.
- **Accessibility:** required for reliable global keyboard event monitoring,
  Esc suppression, and agentic AX tree inspection. Without it, Esc suppression
  over another active app is best effort only and agentic mode cannot inspect
  app controls.
- **Input control / PostEvent:** required for agentic mouse, keyboard, and scroll
  actions. Overlay-only Esc handling does not depend on it.

## Windows Fallback

Windows uses a layered transparent topmost window with `WS_EX_NOACTIVATE`,
cursor-tracked `WS_EX_TRANSPARENT` outside UI regions, `WM_MOUSEACTIVATE`
returning `MA_NOACTIVATE`, and a low-level keyboard hook for Esc.
`WM_NCHITTEST`/`HTTRANSPARENT` is still useful inside the WebView2 window tree,
but Microsoft documents `HTTRANSPARENT` as same-thread hit-test rerouting, so
cross-app click-through relies on the layered-window `WS_EX_TRANSPARENT` path.

## Sources

- Apple `NSWindowStyleMaskNonactivatingPanel`:
  https://developer.apple.com/documentation/appkit/nswindow/stylemask-swift.struct/nonactivatingpanel
- Apple `NSView.acceptsFirstMouse`:
  https://developer.apple.com/documentation/appkit/nsview/acceptsfirstmouse%28for%3A%29
- Apple `NSWindow.CollectionBehavior`:
  https://developer.apple.com/documentation/appkit/nswindow/collectionbehavior-swift.struct
- Apple `NSWindow.ignoresMouseEvents`:
  https://developer.apple.com/documentation/appkit/nswindow
- Apple `NSView.hitTest`:
  https://developer.apple.com/documentation/appkit/nsview/hittest%28_%3A%29
- Apple `NSEvent.addGlobalMonitorForEvents`:
  https://developer.apple.com/documentation/appkit/nsevent/addglobalmonitorforevents%28matching%3Ahandler%3A%29
- Apple Event Handling Guide, Monitoring Events:
  https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/EventOverview/MonitoringEvents/MonitoringEvents.html
- Apple `CGEventTapCreate`:
  https://developer.apple.com/documentation/coregraphics/cgevent/tapcreate%28tap%3Aplace%3Aoptions%3Aeventsofinterest%3Acallback%3Auserinfo%3A%29
- Apple `CGEventTapCallBack`:
  https://developer.apple.com/documentation/coregraphics/cgeventtapcallback
- Microsoft `WM_NCHITTEST` / `HTTRANSPARENT`:
  https://learn.microsoft.com/en-us/windows/win32/inputdev/wm-nchittest
- Microsoft extended window styles (`WS_EX_LAYERED`, `WS_EX_NOACTIVATE`,
  `WS_EX_TRANSPARENT`, `WS_EX_TOOLWINDOW`):
  https://learn.microsoft.com/en-us/windows/win32/winmsg/extended-window-styles
- Microsoft layered-window hit testing:
  https://learn.microsoft.com/en-us/windows/win32/winmsg/window-features
- Microsoft `WM_MOUSEACTIVATE` / `MA_NOACTIVATE`:
  https://learn.microsoft.com/en-us/windows/win32/inputdev/wm-mouseactivate
- Tauri `WebviewWindow`:
  https://docs.rs/tauri/latest/tauri/webview/struct.WebviewWindow.html
