#import <Cocoa/Cocoa.h>
#import <Vision/Vision.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>
#import <QuartzCore/QuartzCore.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#import <ApplicationServices/ApplicationServices.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

static NSWindow *screenieOverlayWindow = nil;
static id screenieOverlayEscapeLocalMonitor = nil;
static id screenieOverlayEscapeGlobalMonitor = nil;
static CFMachPortRef screenieOverlayEscapeEventTap = NULL;
static CFRunLoopSourceRef screenieOverlayEscapeRunLoopSource = NULL;
static bool (*screenieOverlayEscapeCallback)(void) = NULL;
static id screenieOverlayAppDeactivateObserver = nil;
static id screenieOverlaySpaceObserver = nil;
static id screenieOverlayGestureLocalMonitor = nil;
static id screenieOverlayGestureGlobalMonitor = nil;
static void (*screenieOverlayBackgroundChangedCallback)(void) = NULL;
static id screenieOverlayMouseLocalMonitor = nil;
static id screenieMainWindowResizeObserver = nil;
static id screenieMainWindowLiveResizeObserver = nil;

static void screenie_apply_corner_mask_to_view(NSView *view, CGFloat radius) {
  if (view == nil) {
    return;
  }

  [view setWantsLayer:YES];
  CALayer *layer = [view layer];
  if (layer == nil) {
    return;
  }

  [layer setCornerRadius:radius];
  [layer setMasksToBounds:YES];
  [layer setNeedsDisplayOnBoundsChange:YES];

  if (@available(macOS 10.15, *)) {
    [layer setCornerCurve:kCACornerCurveContinuous];
  }
}

static void screenie_pin_direct_subviews_to_bounds(NSView *view) {
  if (view == nil) {
    return;
  }

  NSRect bounds = [view bounds];
  NSArray<NSView *> *subviews = [[view subviews] copy];
  for (NSView *subview in subviews) {
    [subview setAutoresizingMask:(NSViewWidthSizable | NSViewHeightSizable)];
    if (!NSEqualRects([subview frame], bounds)) {
      [subview setFrame:bounds];
    }
  }
}

static void screenie_refresh_main_window_geometry(NSWindow *window) {
  if (window == nil) {
    return;
  }

  NSView *contentView = [window contentView];
  if (contentView == nil) {
    return;
  }

  screenie_apply_corner_mask_to_view(contentView, 24.0);
  screenie_apply_corner_mask_to_view([contentView superview], 24.0);
  screenie_pin_direct_subviews_to_bounds(contentView);

  [contentView setNeedsLayout:YES];
  [contentView layoutSubtreeIfNeeded];
  [[contentView layer] setNeedsLayout];
  [[contentView layer] setNeedsDisplay];
}

static void screenie_install_main_resize_observers(NSWindow *window) {
  if (window == nil) {
    return;
  }

  NSNotificationCenter *center = [NSNotificationCenter defaultCenter];
  if (screenieMainWindowResizeObserver != nil) {
    [center removeObserver:screenieMainWindowResizeObserver];
    screenieMainWindowResizeObserver = nil;
  }
  if (screenieMainWindowLiveResizeObserver != nil) {
    [center removeObserver:screenieMainWindowLiveResizeObserver];
    screenieMainWindowLiveResizeObserver = nil;
  }

  screenieMainWindowResizeObserver =
      [center addObserverForName:NSWindowDidResizeNotification
                          object:window
                           queue:[NSOperationQueue mainQueue]
                      usingBlock:^(NSNotification *notification) {
                        screenie_refresh_main_window_geometry(
                            (NSWindow *)[notification object]);
                      }];

  screenieMainWindowLiveResizeObserver =
      [center addObserverForName:NSWindowDidEndLiveResizeNotification
                          object:window
                           queue:[NSOperationQueue mainQueue]
                      usingBlock:^(NSNotification *notification) {
                        screenie_refresh_main_window_geometry(
                            (NSWindow *)[notification object]);
                      }];
}
static id screenieOverlayMouseGlobalMonitor = nil;
static id screenieOverlayCaptureDragGlobalMonitor = nil;
static CFMachPortRef screenieOverlayCaptureDragEventTap = NULL;
static CFRunLoopSourceRef screenieOverlayCaptureDragRunLoopSource = NULL;
static NSTimer *screenieOverlayMousePollTimer = nil;
static NSTimer *screenieOverlayBackgroundPollTimer = nil;
static NSTimer *screenieOverlayBackgroundSignalTimer = nil;
static uint64_t screenieOverlayBackgroundFingerprint = 0;
static uint64_t screenieOverlayBackgroundVisualFingerprint = 0;
static uint64_t screenieOverlayBackgroundVisualCandidateFingerprint = 0;
static uint64_t screenieOverlayBackgroundVisualGeneration = 0;
static int screenieOverlayBackgroundVisualCandidateCount = 0;
static bool screenieOverlayBackgroundVisualCaptureInFlight = false;
static bool screenieOverlayPassthroughEnabled = false;
static bool screenieOverlayOutsideClickStarted = false;
static bool screenieOverlayMouseCaptureActive = false;
static bool screenieOverlayClickRelayActive = false;
static uint64_t screenieOverlayClickRelayGeneration = 0;
static bool screenieOverlayCaptureDragEnabled = false;
static bool screenieOverlayCaptureDragTracking = false;
static bool screenieOverlayCaptureDragActive = false;
static bool screenieOverlayCaptureDragSuppressing = false;
static NSInteger screenieOverlayCaptureDragButton = 0;
static NSPoint screenieOverlayCaptureDragStartPoint = {0.0, 0.0};
static void (*screenieOverlayCaptureDragCallback)(double dx, double dy, bool ended) = NULL;

typedef struct {
  double x;
  double y;
  double w;
  double h;
} ScreenieOverlayInteractionRegion;

static ScreenieOverlayInteractionRegion screenieOverlayCaptureDragRegion = {0, 0, 0, 0};

/// Borderless transparent NSWindow won't become key by default. The overlay
/// should never programmatically focus/refocus, but it still needs panel-like
/// defaults when AppKit naturally routes a real user click to WebKit.
@interface ScreenieOverlayWindow : NSWindow
@end

static bool screenie_should_relay_overlay_mouse_down(NSEvent *event);
static void screenie_relay_overlay_mouse_down_event(NSEvent *event);

@implementation ScreenieOverlayWindow
- (BOOL)canBecomeKeyWindow {
  return YES;
}
- (BOOL)canBecomeMainWindow {
  return NO;
}
- (BOOL)acceptsFirstMouse:(NSEvent *)event {
  (void)event;
  return YES;
}
- (BOOL)_isNonactivatingPanel {
  return YES;
}
- (void)sendEvent:(NSEvent *)event {
  if (screenie_should_relay_overlay_mouse_down(event)) {
    screenie_relay_overlay_mouse_down_event(event);
    return;
  }
  [super sendEvent:event];
}
@end

static ScreenieOverlayInteractionRegion *screenieOverlayInteractionRegions = NULL;
static size_t screenieOverlayInteractionRegionCount = 0;

static bool screenie_overlay_mouse_is_inside_region(NSPoint globalPoint);
static bool screenie_overlay_mouse_is_inside_capture_drag_region(NSPoint globalPoint);
static void screenie_update_overlay_mouse_passthrough(void);
static void screenie_signal_overlay_background_changed_debounced(void);
void screenie_uninstall_overlay_escape_monitor(void);

static BOOL screenie_view_accepts_first_mouse(id self, SEL _cmd, NSEvent *event) {
  (void)self;
  (void)_cmd;
  (void)event;
  return YES;
}

static NSMutableSet<NSString *> *screenieFirstMousePatchedClasses = nil;

static void screenie_enable_first_mouse_for_view(NSView *view) {
  if (view == nil) {
    return;
  }

  Class cls = object_getClass(view);
  if (cls != Nil) {
    if (screenieFirstMousePatchedClasses == nil) {
      // `[NSMutableSet set]` returns an autoreleased instance — when the
      // pool drains, the set is freed and this static becomes a dangling
      // pointer. macOS's CFPreferences subsystem then reuses the memory
      // for a `__CFPrefsWeakObservers`, and the next `containsObject:`
      // throws `NSInvalidArgumentException -[__CFPrefsWeakObservers
      // containsObject:]: unrecognized selector`. This file is compiled
      // without ARC (build.rs uses cc with -fblocks + -fobjc-exceptions
      // but no -fobjc-arc), so we need an explicit +1 retained instance.
      screenieFirstMousePatchedClasses = [[NSMutableSet alloc] init];
    }
    const char *className = class_getName(cls);
    NSString *key = className != NULL
        ? [NSString stringWithUTF8String:className]
        : nil;
    if (key != nil && ![screenieFirstMousePatchedClasses containsObject:key]) {
      class_replaceMethod(cls,
                          @selector(acceptsFirstMouse:),
                          (IMP)screenie_view_accepts_first_mouse,
                          "c@:@");
      [screenieFirstMousePatchedClasses addObject:key];
    }
  }

  for (NSView *subview in [view subviews]) {
    screenie_enable_first_mouse_for_view(subview);
  }
}

static void screenie_overlay_sync_prevents_activation(NSWindow *window) {
  if (window == nil) {
    return;
  }
  SEL sel = NSSelectorFromString(@"_setPreventsActivation:");
  if ([window respondsToSelector:sel]) {
    ((void (*)(id, SEL, BOOL))objc_msgSend)(window, sel, YES);
  }
}

static uint64_t screenie_hash_mix(uint64_t hash, uint64_t value) {
  hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
  return hash;
}

static uint64_t screenie_overlay_background_window_fingerprint(void) {
  CFArrayRef windows =
      CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID);
  if (windows == NULL) {
    return 0;
  }

  uint64_t hash = 1469598103934665603ULL;
  pid_t ownPid = getpid();
  CFIndex count = CFArrayGetCount(windows);
  int included = 0;

  for (CFIndex i = 0; i < count; i++) {
    CFDictionaryRef info = (CFDictionaryRef)CFArrayGetValueAtIndex(windows, i);
    if (info == NULL || CFGetTypeID(info) != CFDictionaryGetTypeID()) {
      continue;
    }

    int ownerPid = 0;
    CFNumberRef ownerPidRef =
        (CFNumberRef)CFDictionaryGetValue(info, kCGWindowOwnerPID);
    if (ownerPidRef != NULL) {
      CFNumberGetValue(ownerPidRef, kCFNumberIntType, &ownerPid);
    }
    if (ownerPid == ownPid) {
      continue;
    }

    int layer = 0;
    CFNumberRef layerRef =
        (CFNumberRef)CFDictionaryGetValue(info, kCGWindowLayer);
    if (layerRef != NULL) {
      CFNumberGetValue(layerRef, kCFNumberIntType, &layer);
    }
    if (layer != 0) {
      continue;
    }

    int onscreen = 1;
    CFBooleanRef onscreenRef =
        (CFBooleanRef)CFDictionaryGetValue(info, kCGWindowIsOnscreen);
    if (onscreenRef != NULL) {
      onscreen = CFBooleanGetValue(onscreenRef) ? 1 : 0;
    }
    if (!onscreen) {
      continue;
    }

    double alpha = 1.0;
    CFNumberRef alphaRef =
        (CFNumberRef)CFDictionaryGetValue(info, kCGWindowAlpha);
    if (alphaRef != NULL) {
      CFNumberGetValue(alphaRef, kCFNumberDoubleType, &alpha);
    }
    if (alpha <= 0.01) {
      continue;
    }

    CGRect bounds = CGRectZero;
    CFDictionaryRef boundsRef =
        (CFDictionaryRef)CFDictionaryGetValue(info, kCGWindowBounds);
    if (boundsRef == NULL ||
        !CGRectMakeWithDictionaryRepresentation(boundsRef, &bounds) ||
        bounds.size.width <= 1.0 ||
        bounds.size.height <= 1.0) {
      continue;
    }

    uint32_t windowNumber = 0;
    CFNumberRef numberRef =
        (CFNumberRef)CFDictionaryGetValue(info, kCGWindowNumber);
    if (numberRef != NULL) {
      CFNumberGetValue(numberRef, kCFNumberSInt32Type, &windowNumber);
    }

    included++;
    hash = screenie_hash_mix(hash, (uint64_t)windowNumber);
    hash = screenie_hash_mix(hash, (uint64_t)(uint32_t)ownerPid);
    hash = screenie_hash_mix(hash, (uint64_t)(uint32_t)layer);
    hash = screenie_hash_mix(hash, (uint64_t)llround(bounds.origin.x));
    hash = screenie_hash_mix(hash, (uint64_t)llround(bounds.origin.y));
    hash = screenie_hash_mix(hash, (uint64_t)llround(bounds.size.width));
    hash = screenie_hash_mix(hash, (uint64_t)llround(bounds.size.height));
    hash = screenie_hash_mix(hash, (uint64_t)llround(alpha * 1000.0));
  }

  CFRelease(windows);
  return screenie_hash_mix(hash, (uint64_t)(uint32_t)included);
}

static uint64_t screenie_hash_downsampled_image(CGImageRef image) {
  if (image == NULL) {
    return 0;
  }

  enum { sampleW = 48, sampleH = 48, bytesPerPixel = 4 };
  unsigned char pixels[sampleW * sampleH * bytesPerPixel];
  memset(pixels, 0, sizeof(pixels));

  CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
  if (colorSpace == NULL) {
    return 0;
  }

  CGContextRef context =
      CGBitmapContextCreate(pixels,
                            sampleW,
                            sampleH,
                            8,
                            sampleW * bytesPerPixel,
                            colorSpace,
                            kCGImageAlphaPremultipliedLast |
                                kCGBitmapByteOrder32Big);
  CGColorSpaceRelease(colorSpace);
  if (context == NULL) {
    return 0;
  }

  CGContextSetInterpolationQuality(context, kCGInterpolationLow);
  CGContextDrawImage(context,
                     CGRectMake(0.0, 0.0, (CGFloat)sampleW, (CGFloat)sampleH),
                     image);

  uint64_t hash = 1469598103934665603ULL;
  hash = screenie_hash_mix(hash, (uint64_t)CGImageGetWidth(image));
  hash = screenie_hash_mix(hash, (uint64_t)CGImageGetHeight(image));
  for (size_t i = 0; i < sizeof(pixels); i += bytesPerPixel) {
    uint32_t rgba = ((uint32_t)pixels[i] << 24) |
                    ((uint32_t)pixels[i + 1] << 16) |
                    ((uint32_t)pixels[i + 2] << 8) |
                    (uint32_t)pixels[i + 3];
    hash = screenie_hash_mix(hash, (uint64_t)rgba);
  }

  CGContextRelease(context);
  return hash;
}

static char *screenie_copy_png_base64_from_image(CGImageRef image) {
  if (image == NULL) {
    return NULL;
  }

  NSMutableData *data = [NSMutableData data];
  CGImageDestinationRef dest =
      CGImageDestinationCreateWithData((CFMutableDataRef)data,
                                       CFSTR("public.png"),
                                       1,
                                       NULL);
  if (dest == NULL) {
    return NULL;
  }

  CGImageDestinationAddImage(dest, image, NULL);
  BOOL ok = CGImageDestinationFinalize(dest);
  CFRelease(dest);
  if (!ok || data.length == 0) {
    return NULL;
  }

  NSString *base64 = [data base64EncodedStringWithOptions:0];
  const char *utf8 = [base64 UTF8String];
  return utf8 != NULL ? strdup(utf8) : NULL;
}

static SCDisplay *screenie_find_shareable_display(SCShareableContent *content,
                                                  CGDirectDisplayID displayID) {
  if (content == nil) {
    return nil;
  }

  for (SCDisplay *display in content.displays) {
    if (displayID != 0 && display.displayID == displayID) {
      return display;
    }
  }
  return content.displays.firstObject;
}

static SCRunningApplication *screenie_find_own_shareable_application(
    SCShareableContent *content) {
  if (content == nil) {
    return nil;
  }

  pid_t ownPid = getpid();
  for (SCRunningApplication *application in content.applications) {
    if (application.processID == ownPid) {
      return application;
    }
  }
  return nil;
}

static SCContentFilter *screenie_create_display_filter_excluding_self(
    SCShareableContent *content,
    CGDirectDisplayID displayID) {
  SCDisplay *targetDisplay = screenie_find_shareable_display(content, displayID);
  SCRunningApplication *ownApplication =
      screenie_find_own_shareable_application(content);
  if (targetDisplay == nil || ownApplication == nil) {
    return nil;
  }

  return [[SCContentFilter alloc] initWithDisplay:targetDisplay
                            excludingApplications:@[ ownApplication ]
                                 exceptingWindows:@[]];
}

uint32_t screenie_window_display_id(void *window_ptr) {
  @try {
    if (window_ptr == NULL) {
      return 0;
    }
    NSWindow *window = (NSWindow *)window_ptr;
    NSNumber *screenNumber = [[window screen] deviceDescription][@"NSScreenNumber"];
    return screenNumber != nil ? (uint32_t)[screenNumber unsignedIntValue] : 0;
  } @catch (NSException *exception) {
    NSLog(@"[screenie] display id exception: %@ %@", [exception name],
          [exception reason]);
    return 0;
  }
}

const char *screenie_capture_display_png_excluding_self(uint32_t display_id,
                                                       size_t width,
                                                       size_t height) {
  @autoreleasepool {
    __block char *result = NULL;
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

    [SCShareableContent
        getShareableContentExcludingDesktopWindows:NO
                                onScreenWindowsOnly:YES
                                  completionHandler:^(SCShareableContent *content,
                                                      NSError *error) {
      if (error != nil || content == nil) {
        dispatch_semaphore_signal(semaphore);
        return;
      }

      SCContentFilter *filter =
          screenie_create_display_filter_excluding_self(content,
                                                        (CGDirectDisplayID)display_id);
      if (filter == nil) {
        dispatch_semaphore_signal(semaphore);
        return;
      }

      SCStreamConfiguration *config = [[SCStreamConfiguration alloc] init];
      config.width = width > 0 ? width : 64;
      config.height = height > 0 ? height : 64;
      config.scalesToFit = YES;
      config.showsCursor = NO;
      config.capturesAudio = NO;

      [SCScreenshotManager captureImageWithFilter:filter
                                    configuration:config
                                completionHandler:^(CGImageRef image,
                                                    NSError *captureError) {
        if (captureError == nil && image != NULL) {
          result = screenie_copy_png_base64_from_image(image);
        }
        dispatch_semaphore_signal(semaphore);
      }];
    }];

    dispatch_time_t timeout =
        dispatch_time(DISPATCH_TIME_NOW, (int64_t)(3 * NSEC_PER_SEC));
    if (dispatch_semaphore_wait(semaphore, timeout) != 0) {
      if (result != NULL) {
        free(result);
      }
      return NULL;
    }
    return result;
  }
}

static void screenie_handle_overlay_background_visual_fingerprint(uint64_t nextVisual) {
  if (screenieOverlayWindow == nil ||
      ![screenieOverlayWindow isVisible] ||
      screenieOverlayBackgroundChangedCallback == NULL ||
      nextVisual == 0) {
    return;
  }

  if (screenieOverlayBackgroundVisualFingerprint == 0) {
    screenieOverlayBackgroundVisualFingerprint = nextVisual;
    screenieOverlayBackgroundVisualCandidateFingerprint = 0;
    screenieOverlayBackgroundVisualCandidateCount = 0;
    return;
  }
  if (nextVisual == screenieOverlayBackgroundVisualFingerprint) {
    screenieOverlayBackgroundVisualCandidateFingerprint = 0;
    screenieOverlayBackgroundVisualCandidateCount = 0;
    return;
  }
  screenieOverlayBackgroundVisualFingerprint = nextVisual;
  screenieOverlayBackgroundVisualCandidateFingerprint = 0;
  screenieOverlayBackgroundVisualCandidateCount = 0;
  screenie_signal_overlay_background_changed_debounced();
}

static void screenie_request_overlay_background_visual_fingerprint(void) {
  if (screenieOverlayWindow == nil ||
      ![screenieOverlayWindow isVisible] ||
      screenieOverlayBackgroundChangedCallback == NULL ||
      screenieOverlayBackgroundVisualCaptureInFlight) {
    return;
  }

  NSNumber *screenNumber =
      [[screenieOverlayWindow screen] deviceDescription][@"NSScreenNumber"];
  CGDirectDisplayID displayID = screenNumber != nil
      ? (CGDirectDisplayID)[screenNumber unsignedIntValue]
      : 0;
  pid_t ownPid = getpid();
  uint64_t generation = screenieOverlayBackgroundVisualGeneration;
  screenieOverlayBackgroundVisualCaptureInFlight = true;

  [SCShareableContent
      getShareableContentExcludingDesktopWindows:NO
                              onScreenWindowsOnly:YES
                                completionHandler:^(SCShareableContent *content,
                                                    NSError *error) {
    if (error != nil || content == nil) {
      dispatch_async(dispatch_get_main_queue(), ^{
        if (generation == screenieOverlayBackgroundVisualGeneration) {
          screenieOverlayBackgroundVisualCaptureInFlight = false;
        }
      });
      return;
    }

    (void)ownPid;
    SCContentFilter *filter =
        screenie_create_display_filter_excluding_self(content, displayID);
    if (filter == nil) {
      dispatch_async(dispatch_get_main_queue(), ^{
        if (generation == screenieOverlayBackgroundVisualGeneration) {
          screenieOverlayBackgroundVisualCaptureInFlight = false;
        }
      });
      return;
    }

    SCStreamConfiguration *config = [[SCStreamConfiguration alloc] init];
    config.width = 64;
    config.height = 64;
    config.scalesToFit = YES;
    config.showsCursor = NO;
    config.capturesAudio = NO;

    [SCScreenshotManager captureImageWithFilter:filter
                                  configuration:config
                              completionHandler:^(CGImageRef image,
                                                  NSError *captureError) {
      uint64_t hash = captureError == nil ? screenie_hash_downsampled_image(image) : 0;
      dispatch_async(dispatch_get_main_queue(), ^{
        if (generation != screenieOverlayBackgroundVisualGeneration) {
          return;
        }
        screenieOverlayBackgroundVisualCaptureInFlight = false;
        screenie_handle_overlay_background_visual_fingerprint(hash);
      });
    }];
  }];
}

static void screenie_signal_overlay_background_changed_debounced(void) {
  if (screenieOverlayWindow == nil ||
      screenieOverlayBackgroundChangedCallback == NULL ||
      ![screenieOverlayWindow isVisible]) {
    return;
  }

  if (screenieOverlayBackgroundSignalTimer != nil) {
    [screenieOverlayBackgroundSignalTimer invalidate];
    screenieOverlayBackgroundSignalTimer = nil;
  }
  screenieOverlayBackgroundSignalTimer =
      [NSTimer timerWithTimeInterval:0.05
                              repeats:NO
                                block:^(NSTimer *timer) {
    (void)timer;
    screenieOverlayBackgroundSignalTimer = nil;
    if (screenieOverlayBackgroundChangedCallback != NULL &&
        screenieOverlayWindow != nil &&
        [screenieOverlayWindow isVisible]) {
      screenieOverlayBackgroundChangedCallback();
    }
  }];
  [[NSRunLoop mainRunLoop] addTimer:screenieOverlayBackgroundSignalTimer
                            forMode:NSRunLoopCommonModes];
}

static void screenie_update_overlay_background_fingerprint(void) {
  if (screenieOverlayWindow == nil ||
      ![screenieOverlayWindow isVisible] ||
      screenieOverlayBackgroundChangedCallback == NULL) {
    screenieOverlayBackgroundFingerprint = 0;
    screenieOverlayBackgroundVisualFingerprint = 0;
    screenieOverlayBackgroundVisualCandidateFingerprint = 0;
    screenieOverlayBackgroundVisualCandidateCount = 0;
    return;
  }

  uint64_t next = screenie_overlay_background_window_fingerprint();
  if (screenieOverlayBackgroundFingerprint == 0) {
    screenieOverlayBackgroundFingerprint = next;
    screenieOverlayBackgroundVisualFingerprint = 0;
    screenieOverlayBackgroundVisualCandidateFingerprint = 0;
    screenieOverlayBackgroundVisualCandidateCount = 0;
    screenie_request_overlay_background_visual_fingerprint();
    return;
  }
  if (next != 0 && next != screenieOverlayBackgroundFingerprint) {
    screenieOverlayBackgroundFingerprint = next;
    screenieOverlayBackgroundVisualFingerprint = 0;
    screenieOverlayBackgroundVisualCandidateFingerprint = 0;
    screenieOverlayBackgroundVisualCandidateCount = 0;
    screenie_signal_overlay_background_changed_debounced();
    return;
  }

  screenie_request_overlay_background_visual_fingerprint();
}

static void screenie_start_overlay_background_poll_timer(void) {
  if (screenieOverlayBackgroundPollTimer != nil) {
    return;
  }
  screenieOverlayBackgroundVisualGeneration++;
  screenie_update_overlay_background_fingerprint();
  screenieOverlayBackgroundPollTimer =
      [NSTimer timerWithTimeInterval:0.20
                              repeats:YES
                                block:^(NSTimer *timer) {
    (void)timer;
    screenie_update_overlay_background_fingerprint();
  }];
  [[NSRunLoop mainRunLoop] addTimer:screenieOverlayBackgroundPollTimer
                            forMode:NSRunLoopCommonModes];
}

static void screenie_stop_overlay_background_poll_timer(void) {
  if (screenieOverlayBackgroundPollTimer != nil) {
    [screenieOverlayBackgroundPollTimer invalidate];
    screenieOverlayBackgroundPollTimer = nil;
  }
  if (screenieOverlayBackgroundSignalTimer != nil) {
    [screenieOverlayBackgroundSignalTimer invalidate];
    screenieOverlayBackgroundSignalTimer = nil;
  }
  screenieOverlayBackgroundVisualGeneration++;
  screenieOverlayBackgroundVisualCaptureInFlight = false;
  screenieOverlayBackgroundFingerprint = 0;
  screenieOverlayBackgroundVisualFingerprint = 0;
  screenieOverlayBackgroundVisualCandidateFingerprint = 0;
  screenieOverlayBackgroundVisualCandidateCount = 0;
}

static void screenie_overlay_set_ignores_mouse(bool ignores) {
  if (screenieOverlayWindow == nil) {
    return;
  }
  if ([screenieOverlayWindow ignoresMouseEvents] != ignores) {
    [screenieOverlayWindow setIgnoresMouseEvents:ignores];
  }
}

static bool screenie_event_is_mouse_down(NSEventType type) {
  return type == NSEventTypeLeftMouseDown ||
         type == NSEventTypeRightMouseDown ||
         type == NSEventTypeOtherMouseDown;
}

static bool screenie_event_is_mouse_up(NSEventType type) {
  return type == NSEventTypeLeftMouseUp ||
         type == NSEventTypeRightMouseUp ||
         type == NSEventTypeOtherMouseUp;
}

static bool screenie_cg_event_is_mouse_down(CGEventType type) {
  return type == kCGEventLeftMouseDown ||
         type == kCGEventRightMouseDown ||
         type == kCGEventOtherMouseDown;
}

static bool screenie_cg_event_is_mouse_dragged(CGEventType type) {
  return type == kCGEventLeftMouseDragged ||
         type == kCGEventRightMouseDragged ||
         type == kCGEventOtherMouseDragged;
}

static bool screenie_cg_event_is_mouse_up(CGEventType type) {
  return type == kCGEventLeftMouseUp ||
         type == kCGEventRightMouseUp ||
         type == kCGEventOtherMouseUp;
}

static CGMouseButton screenie_cg_button_from_number(NSInteger buttonNumber) {
  if (buttonNumber == 1) {
    return kCGMouseButtonRight;
  }
  if (buttonNumber == 2) {
    return kCGMouseButtonCenter;
  }
  return kCGMouseButtonLeft;
}

static CGEventType screenie_cg_mouse_down_type(CGMouseButton button) {
  if (button == kCGMouseButtonRight) {
    return kCGEventRightMouseDown;
  }
  if (button == kCGMouseButtonLeft) {
    return kCGEventLeftMouseDown;
  }
  return kCGEventOtherMouseDown;
}

static CGEventType screenie_cg_mouse_up_type(CGMouseButton button) {
  if (button == kCGMouseButtonRight) {
    return kCGEventRightMouseUp;
  }
  if (button == kCGMouseButtonLeft) {
    return kCGEventLeftMouseUp;
  }
  return kCGEventOtherMouseUp;
}

static bool screenie_current_cg_mouse_location(CGPoint *point) {
  if (point == NULL) {
    return false;
  }
  CGEventRef event = CGEventCreate(NULL);
  if (event == NULL) {
    return false;
  }
  *point = CGEventGetLocation(event);
  CFRelease(event);
  return true;
}

static bool screenie_can_post_mouse_events(void) {
  if (@available(macOS 10.15, *)) {
    if (!CGPreflightPostEventAccess()) {
      (void)CGRequestPostEventAccess();
      NSLog(@"[screenie] click-through relay needs Accessibility/Input "
            "Monitoring approval to post the deferred mouse click");
      return false;
    }
  }
  return true;
}

static bool screenie_post_mouse_click(CGPoint point, CGMouseButton button) {
  CGEventType downType = screenie_cg_mouse_down_type(button);
  CGEventType upType = screenie_cg_mouse_up_type(button);
  CGEventSourceRef source =
      CGEventSourceCreate(kCGEventSourceStateHIDSystemState);
  CGEventRef down = CGEventCreateMouseEvent(source, downType, point, button);
  CGEventRef up = CGEventCreateMouseEvent(source, upType, point, button);

  if (source != NULL) {
    CFRelease(source);
  }
  if (down == NULL || up == NULL) {
    if (down != NULL) {
      CFRelease(down);
    }
    if (up != NULL) {
      CFRelease(up);
    }
    return false;
  }

  CGEventSetIntegerValueField(down, kCGMouseEventClickState, 1);
  CGEventSetIntegerValueField(up, kCGMouseEventClickState, 1);
  CGEventSetIntegerValueField(down, kCGMouseEventButtonNumber, button);
  CGEventSetIntegerValueField(up, kCGMouseEventButtonNumber, button);
  CGEventPost(kCGHIDEventTap, down);
  CGEventPost(kCGHIDEventTap, up);
  CFRelease(down);
  CFRelease(up);
  return true;
}

static bool screenie_post_scroll(double deltaX, double deltaY, int phase) {
  if (!screenie_can_post_mouse_events()) {
    return false;
  }
  int32_t wheelY = (int32_t)lrint(-deltaY);
  int32_t wheelX = (int32_t)lrint(-deltaX);
  if (wheelY == 0 && fabs(deltaY) > 0.01) {
    wheelY = deltaY > 0 ? -1 : 1;
  }
  if (wheelX == 0 && fabs(deltaX) > 0.01) {
    wheelX = deltaX > 0 ? -1 : 1;
  }

  CGEventRef scroll =
      CGEventCreateScrollWheelEvent(NULL,
                                    kCGScrollEventUnitPixel,
                                    2,
                                    wheelY,
                                    wheelX);
  if (scroll == NULL) {
    return false;
  }
  // Tag the synthetic event as a continuous (trackpad-style) scroll and
  // stamp the gesture phase JS inferred. Without these the receiving
  // app's compositor treats each event as a discrete mouse-wheel tick:
  // no momentum, no rubber-band, no continuity with the user's real
  // passthrough wheel events that arrive between relays — and the mix
  // of phased real events with phaseless synth events is what reads as
  // choppy. Phase values match kCGScrollPhase (1=Began, 2=Changed,
  // 4=Ended); 0 means "don't stamp a phase".
  CGEventSetIntegerValueField(scroll, kCGScrollWheelEventIsContinuous, 1);
  if (phase != 0) {
    CGEventSetIntegerValueField(scroll, kCGScrollWheelEventScrollPhase, phase);
  }
  CGEventPost(kCGHIDEventTap, scroll);
  CFRelease(scroll);
  return true;
}

/// PID of the topmost layer-0 window under `point` (CG / top-left coords)
/// owned by another app. The click-relay path uses this to activate the
/// underlying app after posting a synthetic click — otherwise our
/// nonactivating panel keeps system-wide key-window status across the
/// relay, and the user's next keystroke goes to our prompt textarea
/// instead of the app they just clicked through to. Returns 0 when
/// nothing qualifying matches (desktop background, dock, our own window).
static pid_t screenie_app_pid_at_cg_point(CGPoint point) {
  CFArrayRef windows =
      CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID);
  if (windows == NULL) {
    return 0;
  }

  pid_t result = 0;
  pid_t ownPid = getpid();
  CFIndex count = CFArrayGetCount(windows);

  for (CFIndex i = 0; i < count; i++) {
    CFDictionaryRef info = (CFDictionaryRef)CFArrayGetValueAtIndex(windows, i);
    if (info == NULL || CFGetTypeID(info) != CFDictionaryGetTypeID()) {
      continue;
    }

    int ownerPid = 0;
    CFNumberRef ownerPidRef =
        (CFNumberRef)CFDictionaryGetValue(info, kCGWindowOwnerPID);
    if (ownerPidRef != NULL) {
      CFNumberGetValue(ownerPidRef, kCFNumberIntType, &ownerPid);
    }
    if (ownerPid <= 0 || ownerPid == ownPid) {
      continue;
    }

    int layer = 0;
    CFNumberRef layerRef =
        (CFNumberRef)CFDictionaryGetValue(info, kCGWindowLayer);
    if (layerRef != NULL) {
      CFNumberGetValue(layerRef, kCFNumberIntType, &layer);
    }
    // Skip non-zero layers — Dock, menu-bar status items, other panels,
    // our own status-level overlay. "Activate the Dock on click-through"
    // is wrong; we only want real app windows.
    if (layer != 0) {
      continue;
    }

    int onscreen = 1;
    CFBooleanRef onscreenRef =
        (CFBooleanRef)CFDictionaryGetValue(info, kCGWindowIsOnscreen);
    if (onscreenRef != NULL) {
      onscreen = CFBooleanGetValue(onscreenRef) ? 1 : 0;
    }
    if (!onscreen) {
      continue;
    }

    double alpha = 1.0;
    CFNumberRef alphaRef =
        (CFNumberRef)CFDictionaryGetValue(info, kCGWindowAlpha);
    if (alphaRef != NULL) {
      CFNumberGetValue(alphaRef, kCFNumberDoubleType, &alpha);
    }
    if (alpha <= 0.01) {
      continue;
    }

    CGRect bounds = CGRectZero;
    CFDictionaryRef boundsRef =
        (CFDictionaryRef)CFDictionaryGetValue(info, kCGWindowBounds);
    if (boundsRef == NULL ||
        !CGRectMakeWithDictionaryRepresentation(boundsRef, &bounds) ||
        bounds.size.width <= 1.0 || bounds.size.height <= 1.0) {
      continue;
    }

    if (CGRectContainsPoint(bounds, point)) {
      // CGWindowList returns front-to-back; the first hit is the topmost.
      result = (pid_t)ownerPid;
      break;
    }
  }

  CFRelease(windows);
  return result;
}

static bool screenie_relay_overlay_click_at_current_mouse(NSInteger buttonNumber) {
  if (screenieOverlayWindow == nil || ![screenieOverlayWindow isVisible]) {
    return false;
  }
  if (!screenie_can_post_mouse_events()) {
    return false;
  }

  CGPoint point;
  if (!screenie_current_cg_mouse_location(&point)) {
    return false;
  }

  // Snapshot the app under the cursor BEFORE posting the click so we can
  // hand keyboard focus to it after the click lands. Without this step,
  // clicking a button in (say) Safari via the overlay's passthrough region
  // presses the button but our nonactivating panel keeps system-wide
  // key-window status — the user's next keystroke goes to our prompt
  // textarea instead of Safari, and they have to click Safari a second
  // time to "really" focus it.
  pid_t targetPid = screenie_app_pid_at_cg_point(point);

  CGMouseButton button = screenie_cg_button_from_number(buttonNumber);
  uint64_t generation = ++screenieOverlayClickRelayGeneration;
  screenieOverlayClickRelayActive = true;
  screenieOverlayMouseCaptureActive = false;
  screenie_overlay_set_ignores_mouse(true);

  // Give WindowServer one turn to apply ignoresMouseEvents before posting the
  // deferred click. Otherwise the synthetic click can hit the overlay again.
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 8 * NSEC_PER_MSEC),
                 dispatch_get_main_queue(), ^{
    (void)screenie_post_mouse_click(point, button);
    if (targetPid > 0) {
      // `activateWithOptions:0` doesn't raise windows — our status-level
      // overlay stays visually on top — it just hands key-window status
      // to the target app's frontmost window so subsequent keystrokes
      // follow the click. Same call shape the Cmd+digit forwarding path
      // uses (`screenie_forward_overlay_key_to_previous_app`).
      NSRunningApplication *target = [NSRunningApplication
          runningApplicationWithProcessIdentifier:targetPid];
      if (target != nil) {
        [target activateWithOptions:0];
      }
    }
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 80 * NSEC_PER_MSEC),
                   dispatch_get_main_queue(), ^{
      if (screenieOverlayClickRelayGeneration == generation) {
        screenieOverlayClickRelayActive = false;
        screenie_update_overlay_mouse_passthrough();
      }
    });
  });
  return true;
}

static bool screenie_relay_overlay_scroll(double deltaX,
                                          double deltaY,
                                          int phase) {
  if (screenieOverlayWindow == nil || ![screenieOverlayWindow isVisible]) {
    return false;
  }
  // P-C-R10: scroll relay uses CGEventPost, which is gated by Accessibility
  // (kAXTrustedCheckOptionPrompt). On a Mac without that grant,
  // screenie_post_scroll returns false and the underlying app gets nothing.
  // If we flip ignoresMouseEvents:YES before checking, every wheel tick
  // flickers the overlay through the click-relay state with no actual
  // scroll being delivered. Preflight; bail before mutating state.
  if (!screenie_can_post_mouse_events()) {
    return false;
  }

  uint64_t generation = ++screenieOverlayClickRelayGeneration;
  screenieOverlayClickRelayActive = true;
  screenieOverlayMouseCaptureActive = false;
  screenie_overlay_set_ignores_mouse(true);

  // Post immediately — the click-relay path waits 1ms for WindowServer to
  // settle `ignoresMouseEvents:YES` before the injected click is
  // hit-tested, but a scroll only needs the cursor position (handled by
  // CGEventPost regardless of overlay state), so the delay just added
  // jitter between events in the gesture stream.
  (void)screenie_post_scroll(deltaX, deltaY, phase);
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 24 * NSEC_PER_MSEC),
                 dispatch_get_main_queue(), ^{
    if (screenieOverlayClickRelayGeneration == generation) {
      screenieOverlayClickRelayActive = false;
      screenie_update_overlay_mouse_passthrough();
    }
  });
  return true;
}

static void screenie_reset_overlay_capture_drag_tracking(void) {
  screenieOverlayCaptureDragTracking = false;
  screenieOverlayCaptureDragActive = false;
  screenieOverlayCaptureDragSuppressing = false;
  screenieOverlayCaptureDragButton = 0;
  screenieOverlayCaptureDragStartPoint = NSMakePoint(0.0, 0.0);
}

static void screenie_handle_overlay_capture_drag_observed_event(
    NSEventType type,
    NSPoint point,
    NSInteger buttonNumber) {
  if (!screenieOverlayCaptureDragEnabled ||
      screenieOverlayCaptureDragCallback == NULL ||
      screenieOverlayWindow == nil ||
      ![screenieOverlayWindow isVisible] ||
      screenieOverlayClickRelayActive) {
    return;
  }

  if (screenie_event_is_mouse_down(type)) {
    if (!screenie_overlay_mouse_is_inside_capture_drag_region(point)) {
      return;
    }
    screenieOverlayCaptureDragTracking = true;
    screenieOverlayCaptureDragActive = false;
    screenieOverlayCaptureDragSuppressing = false;
    screenieOverlayCaptureDragStartPoint = point;
    screenieOverlayCaptureDragButton = buttonNumber;
    return;
  }

  if (!screenieOverlayCaptureDragTracking) {
    return;
  }

  double dx = point.x - screenieOverlayCaptureDragStartPoint.x;
  double dy = screenieOverlayCaptureDragStartPoint.y - point.y;

  if (screenie_event_is_mouse_down(type)) {
    return;
  }

  if (type == NSEventTypeLeftMouseDragged ||
      type == NSEventTypeRightMouseDragged ||
      type == NSEventTypeOtherMouseDragged) {
    if (!screenieOverlayCaptureDragActive) {
      double distance = hypot(dx, dy);
      if (distance < 5.0) {
        return;
      }
      screenieOverlayCaptureDragActive = true;
    }
    screenieOverlayCaptureDragCallback(dx, dy, false);
    return;
  }

  if (screenie_event_is_mouse_up(type)) {
    if (screenieOverlayCaptureDragActive) {
      screenieOverlayCaptureDragCallback(dx, dy, true);
    }
    screenie_reset_overlay_capture_drag_tracking();
  }
}

static CGEventRef screenie_overlay_capture_drag_event_tap(
    CGEventTapProxy proxy,
    CGEventType type,
    CGEventRef event,
    void *refcon) {
  (void)proxy;
  (void)refcon;

  if (type == kCGEventTapDisabledByTimeout ||
      type == kCGEventTapDisabledByUserInput) {
    if (screenieOverlayCaptureDragEventTap != NULL) {
      CGEventTapEnable(screenieOverlayCaptureDragEventTap, true);
    }
    return event;
  }

  if (!screenieOverlayCaptureDragEnabled ||
      screenieOverlayCaptureDragCallback == NULL ||
      screenieOverlayWindow == nil ||
      ![screenieOverlayWindow isVisible] ||
      screenieOverlayClickRelayActive) {
    return event;
  }

  NSPoint point = [NSEvent mouseLocation];

  if (screenie_cg_event_is_mouse_down(type)) {
    if (!screenie_overlay_mouse_is_inside_capture_drag_region(point)) {
      return event;
    }
    // We suppress the initial down only when we can replay it later. That
    // lets a normal click still activate/click the app underneath, while a
    // drag becomes a Screenie move without leaking a partial drag to Safari.
    // If that permission is not present, return the event and let the
    // non-mutating NSEvent global monitor below provide best-effort movement.
    if (!screenie_can_post_mouse_events()) {
      return event;
    }
    screenieOverlayCaptureDragTracking = true;
    screenieOverlayCaptureDragActive = false;
    screenieOverlayCaptureDragSuppressing = true;
    screenieOverlayCaptureDragStartPoint = point;
    screenieOverlayCaptureDragButton =
        (NSInteger)CGEventGetIntegerValueField(event, kCGMouseEventButtonNumber);
    return NULL;
  }

  if (!screenieOverlayCaptureDragTracking) {
    return event;
  }

  if (!screenieOverlayCaptureDragSuppressing) {
    return event;
  }

  double dx = point.x - screenieOverlayCaptureDragStartPoint.x;
  double dy = screenieOverlayCaptureDragStartPoint.y - point.y;

  if (screenie_cg_event_is_mouse_dragged(type)) {
    if (!screenieOverlayCaptureDragActive) {
      double distance = hypot(dx, dy);
      if (distance < 5.0) {
        return NULL;
      }
      screenieOverlayCaptureDragActive = true;
    }
    screenieOverlayCaptureDragCallback(dx, dy, false);
    return NULL;
  }

  if (screenie_cg_event_is_mouse_up(type)) {
    bool wasActive = screenieOverlayCaptureDragActive;
    NSInteger buttonNumber = screenieOverlayCaptureDragButton;
    if (wasActive) {
      screenieOverlayCaptureDragCallback(dx, dy, true);
    }
    screenie_reset_overlay_capture_drag_tracking();
    if (!wasActive) {
      (void)screenie_relay_overlay_click_at_current_mouse(buttonNumber);
    }
    return NULL;
  }

  return event;
}

static bool screenie_install_overlay_capture_drag_event_tap(void) {
  if (screenieOverlayCaptureDragEventTap != NULL) {
    CGEventTapEnable(screenieOverlayCaptureDragEventTap, true);
    return true;
  }

  CGEventMask mask = CGEventMaskBit(kCGEventLeftMouseDown) |
                     CGEventMaskBit(kCGEventLeftMouseDragged) |
                     CGEventMaskBit(kCGEventLeftMouseUp) |
                     CGEventMaskBit(kCGEventRightMouseDown) |
                     CGEventMaskBit(kCGEventRightMouseDragged) |
                     CGEventMaskBit(kCGEventRightMouseUp) |
                     CGEventMaskBit(kCGEventOtherMouseDown) |
                     CGEventMaskBit(kCGEventOtherMouseDragged) |
                     CGEventMaskBit(kCGEventOtherMouseUp);
  screenieOverlayCaptureDragEventTap =
      CGEventTapCreate(kCGSessionEventTap,
                       kCGHeadInsertEventTap,
                       kCGEventTapOptionDefault,
                       mask,
                       screenie_overlay_capture_drag_event_tap,
                       NULL);
  if (screenieOverlayCaptureDragEventTap == NULL) {
    NSLog(@"[screenie] capture drag event tap unavailable; Accessibility/Input "
          "Monitoring permission is likely missing");
    return false;
  }

  screenieOverlayCaptureDragRunLoopSource =
      CFMachPortCreateRunLoopSource(kCFAllocatorDefault,
                                    screenieOverlayCaptureDragEventTap,
                                    0);
  if (screenieOverlayCaptureDragRunLoopSource == NULL) {
    CFRelease(screenieOverlayCaptureDragEventTap);
    screenieOverlayCaptureDragEventTap = NULL;
    return false;
  }

  CFRunLoopAddSource(CFRunLoopGetMain(),
                     screenieOverlayCaptureDragRunLoopSource,
                     kCFRunLoopCommonModes);
  CGEventTapEnable(screenieOverlayCaptureDragEventTap, true);
  return true;
}

static bool screenie_install_overlay_capture_drag_global_monitor(void) {
  if (screenieOverlayCaptureDragGlobalMonitor != nil) {
    return true;
  }

  NSEventMask mask = NSEventMaskLeftMouseDown |
                     NSEventMaskRightMouseDown |
                     NSEventMaskOtherMouseDown |
                     NSEventMaskLeftMouseDragged |
                     NSEventMaskRightMouseDragged |
                     NSEventMaskOtherMouseDragged |
                     NSEventMaskLeftMouseUp |
                     NSEventMaskRightMouseUp |
                     NSEventMaskOtherMouseUp;
  screenieOverlayCaptureDragGlobalMonitor =
      [NSEvent addGlobalMonitorForEventsMatchingMask:mask
                                             handler:^(NSEvent *event) {
    if (event == nil) {
      return;
    }
    screenie_handle_overlay_capture_drag_observed_event(
        [event type], [NSEvent mouseLocation], [event buttonNumber]);
  }];
  return screenieOverlayCaptureDragGlobalMonitor != nil;
}

static void screenie_uninstall_overlay_capture_drag_event_tap(void) {
  if (screenieOverlayCaptureDragEventTap != NULL) {
    CGEventTapEnable(screenieOverlayCaptureDragEventTap, false);
  }
  if (screenieOverlayCaptureDragRunLoopSource != NULL) {
    CFRunLoopRemoveSource(CFRunLoopGetMain(),
                          screenieOverlayCaptureDragRunLoopSource,
                          kCFRunLoopCommonModes);
    CFRelease(screenieOverlayCaptureDragRunLoopSource);
    screenieOverlayCaptureDragRunLoopSource = NULL;
  }
  if (screenieOverlayCaptureDragEventTap != NULL) {
    CFRelease(screenieOverlayCaptureDragEventTap);
    screenieOverlayCaptureDragEventTap = NULL;
  }
}

static void screenie_uninstall_overlay_capture_drag_global_monitor(void) {
  if (screenieOverlayCaptureDragGlobalMonitor != nil) {
    [NSEvent removeMonitor:screenieOverlayCaptureDragGlobalMonitor];
    screenieOverlayCaptureDragGlobalMonitor = nil;
  }
}

static bool screenie_should_relay_overlay_mouse_down(NSEvent *event) {
  if (event == nil ||
      screenieOverlayWindow == nil ||
      ![screenieOverlayWindow isVisible] ||
      !screenieOverlayPassthroughEnabled ||
      screenieOverlayMouseCaptureActive ||
      screenieOverlayClickRelayActive ||
      [event window] != screenieOverlayWindow ||
      !screenie_event_is_mouse_down([event type])) {
    return false;
  }

  return !screenie_overlay_mouse_is_inside_region([NSEvent mouseLocation]);
}

static void screenie_relay_overlay_mouse_down_event(NSEvent *event) {
  NSInteger buttonNumber = event != nil ? [event buttonNumber] : 0;
  (void)screenie_relay_overlay_click_at_current_mouse(buttonNumber);
}

static void screenie_note_global_mouse_click_for_refresh(NSEvent *event) {
  if (event == nil ||
      screenieOverlayWindow == nil ||
      ![screenieOverlayWindow isVisible] ||
      screenieOverlayBackgroundChangedCallback == NULL) {
    screenieOverlayOutsideClickStarted = false;
    return;
  }

  NSEventType type = [event type];
  if (!screenie_event_is_mouse_down(type) && !screenie_event_is_mouse_up(type)) {
    return;
  }

  NSPoint point = [NSEvent mouseLocation];
  bool insideOverlayUi = screenie_overlay_mouse_is_inside_region(point);

  if (screenie_event_is_mouse_down(type)) {
    screenieOverlayOutsideClickStarted = !insideOverlayUi;
    return;
  }

  bool shouldRefresh = screenieOverlayOutsideClickStarted && !insideOverlayUi;
  screenieOverlayOutsideClickStarted = false;
  if (shouldRefresh) {
    screenie_signal_overlay_background_changed_debounced();
  }
}

/// When a mouseDown happens through the overlay's empty-space passthrough
/// (cursor outside every interaction region, `ignoresMouseEvents` already
/// YES), the click already reached the underlying app — but our
/// nonactivating panel keeps system-wide key-window status, so the next
/// keystroke goes to our overlay's prompt textarea instead of the app the
/// user just clicked. Mirror the explicit click-relay path: snapshot the
/// PID under the cursor and `activateWithOptions:0` the target so keyboard
/// focus follows the click. Without this, switching browser tabs through
/// the overlay required a second click before keyboard shortcuts worked.
static void screenie_activate_app_on_passthrough_click(NSEvent *event) {
  if (event == nil ||
      screenieOverlayWindow == nil ||
      ![screenieOverlayWindow isVisible] ||
      !screenieOverlayPassthroughEnabled ||
      screenieOverlayMouseCaptureActive ||
      screenieOverlayClickRelayActive) {
    return;
  }
  if (!screenie_event_is_mouse_down([event type])) {
    return;
  }
  // Inside an overlay region → overlay handles the click via the local
  // monitor; nothing for us to do. Belt-and-braces: the global monitor
  // only fires for events delivered to OTHER apps, so reaching here
  // already implies the click went through.
  if (screenie_overlay_mouse_is_inside_region([NSEvent mouseLocation])) {
    return;
  }

  CGPoint cg;
  if (!screenie_current_cg_mouse_location(&cg)) {
    return;
  }
  pid_t targetPid = screenie_app_pid_at_cg_point(cg);
  if (targetPid <= 0 || targetPid == getpid()) {
    return;
  }
  NSRunningApplication *target =
      [NSRunningApplication runningApplicationWithProcessIdentifier:targetPid];
  if (target == nil || [target isActive]) {
    return;
  }
  // `activateWithOptions:0` does NOT raise the target's windows — our
  // status-level overlay stays visually on top. It just hands key-window
  // status to the target's frontmost window so subsequent keystrokes
  // follow the click. Same call shape as the explicit click-relay path
  // (`screenie_relay_overlay_click_at_current_mouse`).
  [target activateWithOptions:0];
}

static void screenie_enable_overlay_mouse_for_interaction(void) {
  if (screenieOverlayWindow == nil || ![screenieOverlayWindow isVisible]) {
    return;
  }
  screenie_overlay_set_ignores_mouse(false);
}

static bool screenie_overlay_mouse_is_inside_region(NSPoint globalPoint) {
  if (screenieOverlayWindow == nil || screenieOverlayInteractionRegionCount == 0) {
    return false;
  }

  NSRect frame = [screenieOverlayWindow frame];
  double x = globalPoint.x - NSMinX(frame);
  double y = NSMaxY(frame) - globalPoint.y;
  // Keep this exact: padding creates invisible click-blocking bands around
  // handles/buttons, which is especially noticeable inside the capture rect.
  const double pad = 0.0;

  for (size_t i = 0; i < screenieOverlayInteractionRegionCount; i++) {
    ScreenieOverlayInteractionRegion r = screenieOverlayInteractionRegions[i];
    if (r.w <= 0.0 || r.h <= 0.0) {
      continue;
    }
    if (x >= r.x - pad && x <= r.x + r.w + pad &&
        y >= r.y - pad && y <= r.y + r.h + pad) {
      return true;
    }
  }
  return false;
}

static bool screenie_overlay_mouse_is_inside_capture_drag_region(NSPoint globalPoint) {
  if (screenieOverlayWindow == nil ||
      !screenieOverlayCaptureDragEnabled ||
      screenieOverlayCaptureDragRegion.w <= 0.0 ||
      screenieOverlayCaptureDragRegion.h <= 0.0) {
    return false;
  }

  NSRect frame = [screenieOverlayWindow frame];
  double x = globalPoint.x - NSMinX(frame);
  double y = NSMaxY(frame) - globalPoint.y;
  ScreenieOverlayInteractionRegion r = screenieOverlayCaptureDragRegion;
  return x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h;
}

static void screenie_update_overlay_mouse_passthrough(void) {
  if (screenieOverlayWindow == nil || ![screenieOverlayWindow isVisible] ||
      !screenieOverlayPassthroughEnabled) {
    screenie_overlay_set_ignores_mouse(false);
    return;
  }

  if (screenieOverlayClickRelayActive) {
    screenie_overlay_set_ignores_mouse(true);
    return;
  }

  if (screenieOverlayMouseCaptureActive) {
    screenie_overlay_set_ignores_mouse(false);
    return;
  }

  if (screenieOverlayInteractionRegionCount == 0) {
    screenie_overlay_set_ignores_mouse(true);
    return;
  }

  NSPoint mouse = [NSEvent mouseLocation];
  bool insideRegion = screenie_overlay_mouse_is_inside_region(mouse);
  if (insideRegion) {
    screenie_enable_overlay_mouse_for_interaction();
  } else {
    screenie_overlay_set_ignores_mouse(true);
  }
}

static void screenie_start_overlay_mouse_poll_timer(void) {
  if (screenieOverlayMousePollTimer != nil) {
    return;
  }

  screenieOverlayMousePollTimer =
      [NSTimer timerWithTimeInterval:0.016
                              repeats:YES
                                block:^(NSTimer *timer) {
    (void)timer;
    screenie_update_overlay_mouse_passthrough();
  }];
  [[NSRunLoop mainRunLoop] addTimer:screenieOverlayMousePollTimer
                            forMode:NSRunLoopCommonModes];
}

static void screenie_install_overlay_mouse_monitors(void) {
  if (screenieOverlayWindow != nil) {
    [screenieOverlayWindow setAcceptsMouseMovedEvents:YES];
  }

  NSEventMask mask = NSEventMaskMouseMoved |
                     NSEventMaskLeftMouseDragged |
                     NSEventMaskRightMouseDragged |
                     NSEventMaskOtherMouseDragged |
                     NSEventMaskLeftMouseDown |
                     NSEventMaskRightMouseDown |
                     NSEventMaskOtherMouseDown |
                     NSEventMaskLeftMouseUp |
                     NSEventMaskRightMouseUp |
                     NSEventMaskOtherMouseUp |
                     NSEventMaskScrollWheel;

  if (screenieOverlayMouseLocalMonitor == nil) {
    screenieOverlayMouseLocalMonitor =
        [NSEvent addLocalMonitorForEventsMatchingMask:mask
                                              handler:^NSEvent *(NSEvent *event) {
      NSEventType type = [event type];
      if (screenie_event_is_mouse_down(type)) {
        screenieOverlayOutsideClickStarted = false;
        if (screenie_overlay_mouse_is_inside_region([NSEvent mouseLocation])) {
          screenieOverlayMouseCaptureActive = true;
        }
      } else if (screenie_event_is_mouse_up(type)) {
        screenieOverlayMouseCaptureActive = false;
      }
      screenie_update_overlay_mouse_passthrough();
      return event;
    }];
  }

  if (screenieOverlayMouseGlobalMonitor == nil) {
    screenieOverlayMouseGlobalMonitor =
        [NSEvent addGlobalMonitorForEventsMatchingMask:mask
                                               handler:^(NSEvent *event) {
      screenie_update_overlay_mouse_passthrough();
      screenie_note_global_mouse_click_for_refresh(event);
      // Hand key-window status to whichever app the user just clicked
      // through to. Without this, our nonactivating panel keeps key
      // status and the next keystroke goes to the overlay textarea
      // instead of the app the user is now interacting with.
      screenie_activate_app_on_passthrough_click(event);
    }];
  }

  screenie_start_overlay_mouse_poll_timer();
}

static void screenie_clear_overlay_mouse_passthrough(void) {
  screenieOverlayPassthroughEnabled = false;
  screenieOverlayMouseCaptureActive = false;
  screenieOverlayClickRelayActive = false;
  screenieOverlayClickRelayGeneration++;
  screenieOverlayCaptureDragEnabled = false;
  screenieOverlayCaptureDragCallback = NULL;
  screenieOverlayCaptureDragRegion =
      (ScreenieOverlayInteractionRegion){0, 0, 0, 0};
  screenie_reset_overlay_capture_drag_tracking();
  screenie_uninstall_overlay_capture_drag_event_tap();
  screenie_uninstall_overlay_capture_drag_global_monitor();
  if (screenieOverlayInteractionRegions != NULL) {
    free(screenieOverlayInteractionRegions);
    screenieOverlayInteractionRegions = NULL;
  }
  screenieOverlayInteractionRegionCount = 0;
  screenieOverlayOutsideClickStarted = false;

  if (screenieOverlayMouseLocalMonitor != nil) {
    [NSEvent removeMonitor:screenieOverlayMouseLocalMonitor];
    screenieOverlayMouseLocalMonitor = nil;
  }
  if (screenieOverlayMouseGlobalMonitor != nil) {
    [NSEvent removeMonitor:screenieOverlayMouseGlobalMonitor];
    screenieOverlayMouseGlobalMonitor = nil;
  }
  if (screenieOverlayMousePollTimer != nil) {
    [screenieOverlayMousePollTimer invalidate];
    screenieOverlayMousePollTimer = nil;
  }
  screenie_overlay_set_ignores_mouse(false);
}

bool screenie_set_overlay_mouse_capture(void *window_ptr, bool active) {
  @try {
    if (window_ptr == NULL) {
      screenieOverlayMouseCaptureActive = false;
      screenie_update_overlay_mouse_passthrough();
      return false;
    }
    screenieOverlayWindow = (NSWindow *)window_ptr;
    if (active) {
      screenieOverlayClickRelayActive = false;
      screenieOverlayClickRelayGeneration++;
    }
    screenieOverlayMouseCaptureActive = active;
    screenie_update_overlay_mouse_passthrough();
    return true;
  } @catch (NSException *exception) {
    NSLog(@"[screenie] set mouse capture exception: %@ %@",
          [exception name], [exception reason]);
    screenieOverlayMouseCaptureActive = false;
    screenie_update_overlay_mouse_passthrough();
    return false;
  }
}

bool screenie_relay_overlay_click(void *window_ptr, int buttonNumber) {
  @try {
    if (window_ptr == NULL) {
      return false;
    }
    screenieOverlayWindow = (NSWindow *)window_ptr;
    return screenie_relay_overlay_click_at_current_mouse((NSInteger)buttonNumber);
  } @catch (NSException *exception) {
    NSLog(@"[screenie] relay overlay click exception: %@ %@",
          [exception name], [exception reason]);
    screenieOverlayClickRelayActive = false;
    screenie_update_overlay_mouse_passthrough();
    return false;
  }
}

bool screenie_relay_overlay_wheel(void *window_ptr,
                                  double deltaX,
                                  double deltaY,
                                  int phase) {
  @try {
    if (window_ptr == NULL) {
      return false;
    }
    screenieOverlayWindow = (NSWindow *)window_ptr;
    return screenie_relay_overlay_scroll(deltaX, deltaY, phase);
  } @catch (NSException *exception) {
    NSLog(@"[screenie] relay overlay wheel exception: %@ %@",
          [exception name], [exception reason]);
    screenieOverlayClickRelayActive = false;
    screenie_update_overlay_mouse_passthrough();
    return false;
  }
}

bool screenie_set_overlay_capture_drag_region(
    void *window_ptr,
    const ScreenieOverlayInteractionRegion *region,
    bool enabled,
    void (*callback)(double dx, double dy, bool ended)) {
  @try {
    if (!enabled || window_ptr == NULL || region == NULL) {
      screenieOverlayCaptureDragEnabled = false;
      screenieOverlayCaptureDragCallback = NULL;
      screenieOverlayCaptureDragRegion =
          (ScreenieOverlayInteractionRegion){0, 0, 0, 0};
      screenie_reset_overlay_capture_drag_tracking();
      screenie_uninstall_overlay_capture_drag_event_tap();
      screenie_uninstall_overlay_capture_drag_global_monitor();
      return window_ptr != NULL;
    }

    screenieOverlayWindow = (NSWindow *)window_ptr;
    screenieOverlayCaptureDragRegion = *region;
    screenieOverlayCaptureDragCallback = callback;
    screenieOverlayCaptureDragEnabled =
        region->w > 0.0 && region->h > 0.0 && callback != NULL;

    if (!screenieOverlayCaptureDragEnabled) {
      screenie_reset_overlay_capture_drag_tracking();
      screenie_uninstall_overlay_capture_drag_event_tap();
      return true;
    }

    bool globalInstalled = screenie_install_overlay_capture_drag_global_monitor();
    bool eventTapInstalled = screenie_install_overlay_capture_drag_event_tap();
    return globalInstalled || eventTapInstalled;
  } @catch (NSException *exception) {
    NSLog(@"[screenie] set capture drag region exception: %@ %@",
          [exception name], [exception reason]);
    screenieOverlayCaptureDragEnabled = false;
    screenieOverlayCaptureDragCallback = NULL;
    screenie_reset_overlay_capture_drag_tracking();
    screenie_uninstall_overlay_capture_drag_event_tap();
    screenie_uninstall_overlay_capture_drag_global_monitor();
    return false;
  }
}

static bool screenie_handle_overlay_escape_event(NSEvent *event,
                                                 bool require_active_window) {
  if ([event keyCode] != 53 || screenieOverlayEscapeCallback == NULL ||
      screenieOverlayWindow == nil || ![screenieOverlayWindow isVisible]) {
    return false;
  }

  // For local monitors: confirm the event was actually delivered to our
  // overlay panel (not some other window of ours). The local monitor only
  // fires when our app receives the event in the first place, which
  // implies our app was active.
  if (require_active_window) {
    if (![NSApp isActive]) {
      return false;
    }
    NSWindow *eventWindow = [event window];
    NSWindow *activeWindow = eventWindow != nil ? eventWindow : [NSApp keyWindow];
    if (activeWindow != screenieOverlayWindow) {
      return false;
    }
  }
  // For global monitors: fire on any Esc press while the overlay is
  // visible, even when our app is inactive — which is the common case
  // with the nonactivating-panel mask (the user has Cmd+Tabbed to
  // another app but the overlay is still up). The global monitor doesn't
  // intercept events, so the user's currently-active app still receives
  // its Esc; we just also close the overlay alongside it. Without this,
  // the previous `[NSApp isActive]` gate silently swallowed every Esc
  // press the moment our app lost focus.

  return screenieOverlayEscapeCallback();
}

// P-C-B3: defined in lib.rs as `#[no_mangle] pub extern "C"`. True while
// `refresh_overlay_capture` is mid-flight — the overlay is intentionally
// hidden for ~300-500 ms but the user's intent is still "this Esc closes
// the overlay", so the tap must keep consuming it. Without this gate, an
// Esc landing in the hide window leaks to the app underneath (fullscreen
// Safari unfullscreens, slide presentations exit).
extern bool screenie_is_overlay_refresh_in_flight(void);

static CGEventRef screenie_overlay_escape_event_tap(CGEventTapProxy proxy,
                                                    CGEventType type,
                                                    CGEventRef event,
                                                    void *refcon) {
  (void)proxy;
  (void)refcon;

  if (type == kCGEventTapDisabledByTimeout ||
      type == kCGEventTapDisabledByUserInput) {
    if (screenieOverlayEscapeEventTap != NULL) {
      CGEventTapEnable(screenieOverlayEscapeEventTap, true);
    }
    return event;
  }

  // The window-visibility gate was the original short-circuit so the tap
  // wouldn't consume Esc after the overlay had been dismissed. But
  // `refresh_overlay_capture` hides the overlay for the duration of the
  // recapture — during that window the visibility gate fires falsely and
  // lets Esc through to the underlying app. Allow the tap to proceed
  // whenever the overlay is visible OR a refresh is mid-flight.
  bool overlay_active =
      screenieOverlayWindow != nil &&
      ([screenieOverlayWindow isVisible] ||
       screenie_is_overlay_refresh_in_flight());
  if (type != kCGEventKeyDown ||
      !overlay_active ||
      screenieOverlayEscapeCallback == NULL) {
    return event;
  }

  int64_t keyCode =
      CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode);
  if (keyCode != 53) {
    return event;
  }

  // NSEvent global monitors can only observe events targeted at other apps;
  // returning NULL from an active CGEventTap is the path that prevents Esc
  // from also reaching fullscreen Safari/Chrome while the overlay is up.
  return screenieOverlayEscapeCallback() ? NULL : event;
}

static bool screenieAccessibilityPrompted = false;

static bool screenie_install_overlay_escape_event_tap(void) {
  if (screenieOverlayEscapeEventTap != NULL) {
    CGEventTapEnable(screenieOverlayEscapeEventTap, true);
    return true;
  }

  CGEventMask mask = CGEventMaskBit(kCGEventKeyDown);
  screenieOverlayEscapeEventTap =
      CGEventTapCreate(kCGSessionEventTap,
                       kCGHeadInsertEventTap,
                       kCGEventTapOptionDefault,
                       mask,
                       screenie_overlay_escape_event_tap,
                       NULL);
  if (screenieOverlayEscapeEventTap == NULL) {
    NSLog(@"[screenie] escape event tap unavailable; Accessibility permission "
          "is likely missing");
    // The tap is the ONLY path that prevents Esc from also reaching the
    // app under the overlay. Without it, Esc still closes our overlay
    // (the NSEvent global monitor's observe-only path fires the
    // callback) but Esc ALSO reaches the active app — fullscreen
    // Safari/Chrome unfullscreens, slide presentations exit, etc. Surface
    // the macOS Accessibility prompt once per launch so the user has a
    // one-click path to grant. The OS no-ops when already trusted, and
    // dedupes the dialog if the user has already dismissed it. Gating
    // with `screenieAccessibilityPrompted` just keeps us from invoking
    // the API on every overlay show — the OS would suppress the dialog
    // anyway, but skipping the call is tidier.
    if (!screenieAccessibilityPrompted) {
      screenieAccessibilityPrompted = true;
      NSDictionary *options = @{
        (__bridge id)kAXTrustedCheckOptionPrompt: @YES,
      };
      (void)AXIsProcessTrustedWithOptions((__bridge CFDictionaryRef)options);
    }
    return false;
  }

  screenieOverlayEscapeRunLoopSource =
      CFMachPortCreateRunLoopSource(kCFAllocatorDefault,
                                    screenieOverlayEscapeEventTap,
                                    0);
  if (screenieOverlayEscapeRunLoopSource == NULL) {
    CFRelease(screenieOverlayEscapeEventTap);
    screenieOverlayEscapeEventTap = NULL;
    return false;
  }

  CFRunLoopAddSource(CFRunLoopGetMain(),
                     screenieOverlayEscapeRunLoopSource,
                     kCFRunLoopCommonModes);
  CGEventTapEnable(screenieOverlayEscapeEventTap, true);
  return true;
}

bool screenie_configure_main_window(void *window_ptr) {
  @try {
    if (window_ptr == NULL) {
      NSLog(@"[screenie] configure main: null NSWindow");
      return false;
    }

    NSWindow *window = (NSWindow *)window_ptr;
    NSWindowStyleMask mask = [window styleMask];
    mask |= NSWindowStyleMaskFullSizeContentView;
    [window setStyleMask:mask];

    if (@available(macOS 11.0, *)) {
      [window setTitlebarSeparatorStyle:(NSTitlebarSeparatorStyle)1];
    }
    [window setTitlebarAppearsTransparent:YES];
    [window setOpaque:NO];
    [window setBackgroundColor:[NSColor clearColor]];

    // Tauri's windowEffects radius can briefly redraw square while macOS is
    // live-resizing the transparent window. Keep a stable layer mask and pin
    // the WebView/material children to the content bounds during resize, so
    // the page does not briefly trail behind the native panel edge.
    screenie_refresh_main_window_geometry(window);
    screenie_install_main_resize_observers(window);

    return true;
  } @catch (NSException *exception) {
    NSLog(@"[screenie] configure main exception: %@ %@", [exception name],
          [exception reason]);
    return false;
  }
}

bool screenie_configure_overlay_window(void *window_ptr) {
  @try {
    if (window_ptr == NULL) {
      NSLog(@"[screenie] configure overlay: null NSWindow");
      return false;
    }

    NSWindow *window = (NSWindow *)window_ptr;
    screenieOverlayWindow = window;

    // Class-swap to ScreenieOverlayWindow so the borderless transparent
    // window can become key (default NSWindow returns NO for borderless)
    // and so first-click activation doesn't get eaten by AppKit. Without
    // this swap clicks on overlay buttons land on a window that can't
    // become key, the WebView never becomes first responder, and the
    // user-visible symptom is "buttons don't work / can't type".
    if (![window isKindOfClass:[ScreenieOverlayWindow class]]) {
      object_setClass(window, [ScreenieOverlayWindow class]);
    }
    screenie_enable_first_mouse_for_view([window contentView]);

    NSWindowStyleMask mask = [window styleMask];
    mask |= NSWindowStyleMaskNonactivatingPanel;
    mask |= NSWindowStyleMaskFullSizeContentView;
    [window setStyleMask:mask];
    // The nonactivating style bit is only fully honored when AppKit also sets
    // the WindowServer prevents-activation tag. Tauri gives us an existing
    // NSWindow, so re-sync the tag after mutating the style mask.
    screenie_overlay_sync_prevents_activation(window);

    [window setLevel:NSStatusWindowLevel];
    [window setHidesOnDeactivate:NO];
    [window setReleasedWhenClosed:NO];
    [window setAcceptsMouseMovedEvents:YES];

    NSWindowCollectionBehavior behavior = [window collectionBehavior];
    behavior &= ~NSWindowCollectionBehaviorStationary;
    behavior |= NSWindowCollectionBehaviorCanJoinAllSpaces;
    behavior |= NSWindowCollectionBehaviorFullScreenAuxiliary;
    behavior |= NSWindowCollectionBehaviorIgnoresCycle;
    // Hide the overlay from the Mission Control / Exposé thumbnail grid —
    // otherwise the borderless transparent overlay shows up as an empty
    // tile next to the user's real windows.
    behavior |= NSWindowCollectionBehaviorTransient;
    [window setCollectionBehavior:behavior];

    return true;
  } @catch (NSException *exception) {
    NSLog(@"[screenie] configure overlay exception: %@ %@", [exception name],
          [exception reason]);
    return false;
  }
}

/// Local OCR via Apple's Vision framework. Takes raw PNG bytes, runs
/// `VNRecognizeTextRequest` synchronously on the calling thread, and
/// returns the recognized text as a UTF-8 C string allocated with
/// `strdup`. The Rust caller must release it with
/// `screenie_free_string`. Returns NULL on any failure (bad PNG,
/// Vision error, no text found is treated as an empty string).
///
/// All work runs offline using Apple's on-device text recognition model
/// — no network round-trip, no cloud API key, no token cost. Available
/// on macOS 10.15+ (covers every version Tauri 2 supports).
const char *screenie_ocr_png(const unsigned char *png_bytes, size_t png_len) {
  if (png_bytes == NULL || png_len == 0) {
    return NULL;
  }

  @autoreleasepool {
    @try {
      CFDataRef cfData = CFDataCreate(NULL, png_bytes, (CFIndex)png_len);
      if (cfData == NULL) {
        NSLog(@"[screenie] ocr: CFDataCreate failed");
        return NULL;
      }
      CGImageSourceRef src = CGImageSourceCreateWithData(cfData, NULL);
      CFRelease(cfData);
      if (src == NULL) {
        NSLog(@"[screenie] ocr: CGImageSourceCreate failed");
        return NULL;
      }
      CGImageRef cgImage = CGImageSourceCreateImageAtIndex(src, 0, NULL);
      CFRelease(src);
      if (cgImage == NULL) {
        NSLog(@"[screenie] ocr: CGImageSource produced no image");
        return NULL;
      }

      VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
      request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
      request.usesLanguageCorrection = YES;
      // Top observation per region — that's the candidate Vision is most
      // confident in. We could ask for more candidates and rank, but for
      // copy-to-clipboard the top pick is the right tradeoff.

      VNImageRequestHandler *handler =
          [[VNImageRequestHandler alloc] initWithCGImage:cgImage options:@{}];
      CGImageRelease(cgImage);

      NSError *error = nil;
      BOOL ok = [handler performRequests:@[ request ] error:&error];
      if (!ok) {
        NSLog(@"[screenie] ocr: performRequests failed: %@", error);
        return NULL;
      }

      // Concatenate observations top-to-bottom, joined by newlines.
      // Vision orders results in roughly reading order; we keep that.
      NSMutableString *result = [NSMutableString string];
      for (VNRecognizedTextObservation *obs in request.results) {
        NSArray<VNRecognizedText *> *candidates = [obs topCandidates:1];
        if (candidates.count == 0) {
          continue;
        }
        NSString *line = [candidates[0] string];
        if (line.length == 0) {
          continue;
        }
        if (result.length > 0) {
          [result appendString:@"\n"];
        }
        [result appendString:line];
      }

      const char *utf8 = [result UTF8String];
      if (utf8 == NULL) {
        return strdup("");
      }
      return strdup(utf8);
    } @catch (NSException *exception) {
      NSLog(@"[screenie] ocr exception: %@ %@", [exception name],
            [exception reason]);
      return NULL;
    }
  }
}

void screenie_free_string(const char *ptr) {
  if (ptr != NULL) {
    free((void *)ptr);
  }
}

// ============================================================================
// Keystroke forwarding to the previously-frontmost app
//
// Once the user clicks the prompt textarea, the overlay panel becomes the
// key window. After that, any unrelated shortcut they press (Cmd+1 to
// switch tabs, Cmd+T for a new tab, etc.) is delivered to our WKWebView
// instead of to whichever app they were just using — and our WebView has
// no binding for it, so it dies.
//
// This block remembers the user's previously-frontmost app at the moment
// the overlay opens, and the local NSEvent monitor (in
// screenie_install_overlay_escape_monitor below) forwards any keystroke
// the overlay does NOT actively handle back to that app: activate it,
// then re-post the same CGEvent into its event queue. Esc, Cmd+L, and
// ALL keystrokes while a text input is focused are exempt — the overlay
// handles those itself.
// ============================================================================

static NSRunningApplication *screenieOverlayPreviousApp = nil;
static bool screenieOverlayTextInputFocused = false;

void screenie_remember_previous_app(void) {
  @autoreleasepool {
    @try {
      NSRunningApplication *app =
          [[NSWorkspace sharedWorkspace] frontmostApplication];
      if (app == nil) return;
      if ([app processIdentifier] == getpid()) return;
      [screenieOverlayPreviousApp release];
      screenieOverlayPreviousApp = [app retain];
    } @catch (NSException *exception) {
      NSLog(@"[screenie] remember previous app exception: %@ %@",
            [exception name], [exception reason]);
    }
  }
}

void screenie_forget_previous_app(void) {
  @autoreleasepool {
    @try {
      [screenieOverlayPreviousApp release];
      screenieOverlayPreviousApp = nil;
      screenieOverlayTextInputFocused = false;
    } @catch (NSException *exception) {
      NSLog(@"[screenie] forget previous app exception: %@ %@",
            [exception name], [exception reason]);
    }
  }
}

void screenie_set_overlay_text_input_focused(bool focused) {
  screenieOverlayTextInputFocused = focused;
}

/// Called by the local NSEvent monitor for any keystroke that isn't
/// handled by the overlay itself. Activates the previously-frontmost
/// app and re-posts the keystroke directly into its process event
/// queue — the original event is consumed by the caller (returns nil
/// from the monitor handler).
///
/// Returns true when the forward was attempted (caller should consume
/// the original event), false when we have no previous app on record
/// and the event should fall through unmodified.
static bool screenie_forward_overlay_key_to_previous_app(NSEvent *event) {
  @autoreleasepool {
    @try {
      if (event == nil || screenieOverlayPreviousApp == nil) return false;
      pid_t pid = [screenieOverlayPreviousApp processIdentifier];
      if (pid <= 0) return false;
      // Activate first so the receiving app actually acts on key
      // events (most apps ignore key events while inactive).
      [screenieOverlayPreviousApp activateWithOptions:0];
      // Re-post the underlying CGEvent into the receiving process's
      // event queue. CGEventPostToPid targets a specific PID
      // regardless of which app is currently key, so this works even
      // before the activation finishes propagating.
      CGEventRef cge = [event CGEvent];
      if (cge == NULL) return true;
      CGEventRef copy = CGEventCreateCopy(cge);
      if (copy == NULL) return true;
      CGEventPostToPid(pid, copy);
      CFRelease(copy);
      return true;
    } @catch (NSException *exception) {
      NSLog(@"[screenie] forward key exception: %@ %@",
            [exception name], [exception reason]);
      return false;
    }
  }
}

bool screenie_install_overlay_escape_monitor(bool (*callback)(void)) {
  @try {
    screenieOverlayEscapeCallback = callback;
    bool eventTapInstalled = screenie_install_overlay_escape_event_tap();
    if (screenieOverlayEscapeLocalMonitor != nil &&
        screenieOverlayEscapeGlobalMonitor != nil) {
      return true;
    }

    if (screenieOverlayEscapeLocalMonitor == nil) {
      screenieOverlayEscapeLocalMonitor =
        [NSEvent addLocalMonitorForEventsMatchingMask:NSEventMaskKeyDown
                                             handler:^NSEvent *(NSEvent *event) {
      // Esc is the overlay's primary dismiss shortcut — handle it
      // unconditionally regardless of focus.
      if (screenie_handle_overlay_escape_event(event, true)) {
        return nil;
      }
      // While a text input is focused (prompt textarea, dropdown
      // search field, edit-canvas text annotation), every keystroke
      // is "typing" — pass them all through so characters, Cmd+A,
      // Cmd+C, arrow keys etc. work normally.
      if (screenieOverlayTextInputFocused) {
        return event;
      }
      // Forward ONLY Cmd+digit to the previously-frontmost app.
      // Earlier we forwarded every unhandled shortcut, but that made
      // ordinary combos (Cmd+W close-tab, Cmd+H hide, Cmd+M minimize)
      // act on the receiving app in surprising ways — including
      // showing the desktop / Finder when the receiving app had no
      // visible windows. Cmd+digit is unambiguously a tab-switch in
      // every browser-style app, so it's the one combo that's safe
      // to forward without "did the wrong thing" risk.
      //
      // The check requires the Command modifier ALONE — Cmd+Shift+1,
      // Cmd+Alt+1 etc. are screenshot / power-user shortcuts and are
      // left for the OS / WKWebView to handle.
      NSEventModifierFlags mods = [event modifierFlags] &
                                  NSEventModifierFlagDeviceIndependentFlagsMask;
      unsigned short keyCode = [event keyCode];
      bool cmdOnly = (mods == NSEventModifierFlagCommand);
      bool isDigit =
          keyCode == 18 || keyCode == 19 || keyCode == 20 ||
          keyCode == 21 || keyCode == 22 || keyCode == 23 ||
          keyCode == 25 || keyCode == 26 || keyCode == 28 ||
          keyCode == 29; // 1-9, then 0
      if (cmdOnly && isDigit &&
          screenie_forward_overlay_key_to_previous_app(event)) {
        return nil;
      }
      // Everything else (Cmd+L provider toggle, other shortcuts the
      // user might have bound, browser defaults like Cmd+R) goes to
      // the WKWebView normally — overlay's JS handlers run for the
      // ones we care about, the rest no-op inside WebKit.
      return event;
    }];
    }

    if (screenieOverlayEscapeGlobalMonitor == nil) {
      screenieOverlayEscapeGlobalMonitor =
          [NSEvent addGlobalMonitorForEventsMatchingMask:NSEventMaskKeyDown
                                                 handler:^(NSEvent *event) {
        (void)screenie_handle_overlay_escape_event(event, false);
      }];
    }

    return eventTapInstalled ||
           screenieOverlayEscapeLocalMonitor != nil ||
           screenieOverlayEscapeGlobalMonitor != nil;
  } @catch (NSException *exception) {
    NSLog(@"[screenie] install escape monitor exception: %@ %@",
          [exception name], [exception reason]);
    return false;
  }
}

void screenie_uninstall_overlay_escape_monitor(void) {
  @try {
    if (screenieOverlayEscapeLocalMonitor != nil) {
      [NSEvent removeMonitor:screenieOverlayEscapeLocalMonitor];
      screenieOverlayEscapeLocalMonitor = nil;
    }
    if (screenieOverlayEscapeGlobalMonitor != nil) {
      [NSEvent removeMonitor:screenieOverlayEscapeGlobalMonitor];
      screenieOverlayEscapeGlobalMonitor = nil;
    }
    if (screenieOverlayEscapeEventTap != NULL) {
      CGEventTapEnable(screenieOverlayEscapeEventTap, false);
    }
    if (screenieOverlayEscapeRunLoopSource != NULL) {
      CFRunLoopRemoveSource(CFRunLoopGetMain(),
                            screenieOverlayEscapeRunLoopSource,
                            kCFRunLoopCommonModes);
      CFRelease(screenieOverlayEscapeRunLoopSource);
      screenieOverlayEscapeRunLoopSource = NULL;
    }
    if (screenieOverlayEscapeEventTap != NULL) {
      CFRelease(screenieOverlayEscapeEventTap);
      screenieOverlayEscapeEventTap = NULL;
    }
    screenieOverlayEscapeCallback = NULL;
  } @catch (NSException *exception) {
    NSLog(@"[screenie] uninstall escape monitor exception: %@ %@",
          [exception name], [exception reason]);
  }
}

bool screenie_order_overlay_window(void *window_ptr) {
  @try {
    if (window_ptr == NULL) {
      NSLog(@"[screenie] order overlay: null NSWindow");
      return false;
    }

    NSWindow *window = (NSWindow *)window_ptr;
    screenieOverlayWindow = window;
    screenie_overlay_set_ignores_mouse(false);
    screenie_overlay_sync_prevents_activation(window);
    screenie_enable_first_mouse_for_view([window contentView]);
    [window setAlphaValue:1.0];
    // Re-assert level on every show; AppKit can drop the window below
    // other status-level windows after a Space change.
    [window setLevel:NSStatusWindowLevel];
    [window orderFrontRegardless];

    return true;
  } @catch (NSException *exception) {
    NSLog(@"[screenie] order overlay exception: %@ %@", [exception name],
          [exception reason]);
    return false;
  }
}

bool screenie_set_overlay_interaction_regions(
    void *window_ptr,
    const ScreenieOverlayInteractionRegion *regions,
    size_t count,
    bool passthrough_enabled) {
  @try {
    if (window_ptr == NULL) {
      screenie_clear_overlay_mouse_passthrough();
      return false;
    }

    screenieOverlayWindow = (NSWindow *)window_ptr;
    screenieOverlayPassthroughEnabled = passthrough_enabled;

    if (screenieOverlayInteractionRegions != NULL) {
      free(screenieOverlayInteractionRegions);
      screenieOverlayInteractionRegions = NULL;
    }
    screenieOverlayInteractionRegionCount = 0;

    if (passthrough_enabled) {
      size_t bytes = sizeof(ScreenieOverlayInteractionRegion) * count;
      if (regions != NULL && count > 0) {
        screenieOverlayInteractionRegions =
            (ScreenieOverlayInteractionRegion *)malloc(bytes);
        if (screenieOverlayInteractionRegions == NULL) {
          screenieOverlayPassthroughEnabled = false;
          screenie_overlay_set_ignores_mouse(false);
          return false;
        }
        memcpy(screenieOverlayInteractionRegions, regions, bytes);
        screenieOverlayInteractionRegionCount = count;
      }
      screenie_install_overlay_mouse_monitors();
    } else {
      screenie_clear_overlay_mouse_passthrough();
    }

    screenie_update_overlay_mouse_passthrough();
    return true;
  } @catch (NSException *exception) {
    NSLog(@"[screenie] set interaction regions exception: %@ %@",
          [exception name], [exception reason]);
    screenie_clear_overlay_mouse_passthrough();
    return false;
  }
}

/// Install native observers for actual background transitions.
/// Ordinary app deactivation is intentionally ignored: the overlay can now
/// pass pointer events through empty space, so activating the app underneath
/// is expected and should not dismiss the Screenie UI.
bool screenie_install_overlay_deactivate_hider(void (*callback)(void)) {
  @try {
    screenieOverlayBackgroundChangedCallback = callback;
    if (screenieOverlayWindow == nil) {
      return false;
    }

    if (screenieOverlayAppDeactivateObserver != nil) {
      [[NSNotificationCenter defaultCenter]
          removeObserver:screenieOverlayAppDeactivateObserver];
      screenieOverlayAppDeactivateObserver = nil;
    }

    if (screenieOverlaySpaceObserver != nil) {
      [[[NSWorkspace sharedWorkspace] notificationCenter]
          removeObserver:screenieOverlaySpaceObserver];
      screenieOverlaySpaceObserver = nil;
    }
    // The overlay follows the user across Spaces via
    // NSWindowCollectionBehaviorMoveToActiveSpace, so a Space change just
    // means the bitmap behind the frosted panels is stale. Emit the
    // bg-changed callback so the JS side refreshes via ScreenCaptureKit's
    // exclude-self capture — silent, no hide. The earlier behaviour
    // (orderOut:nil on every Space change) read as "the overlay vanishes
    // when I switch Spaces / tabs" and is what the user reported.
    screenieOverlaySpaceObserver =
        [[[NSWorkspace sharedWorkspace] notificationCenter]
            addObserverForName:NSWorkspaceActiveSpaceDidChangeNotification
                        object:nil
                         queue:[NSOperationQueue mainQueue]
                    usingBlock:^(NSNotification *note) {
                      (void)note;
                      if (screenieOverlayBackgroundChangedCallback != NULL &&
                          screenieOverlayWindow != nil &&
                          [screenieOverlayWindow isVisible]) {
                        screenieOverlayBackgroundChangedCallback();
                      }
                    }];

    // Gesture-based hiding (3-finger / horizontal-scroll swipes) was
    // removed: it triggered on ordinary in-app gestures (browser tab
    // navigation, horizontal page scrolls) and produced the visible
    // "overlay disappears when I swipe" bug. The visual fingerprint poll
    // installed below still detects desktop changes for bitmap freshness
    // without ever hiding the overlay window.

    screenie_start_overlay_background_poll_timer();

    return true;
  } @catch (NSException *exception) {
    NSLog(@"[screenie] install deactivate hider exception: %@ %@",
          [exception name], [exception reason]);
    return false;
  }
}

void screenie_uninstall_overlay_deactivate_hider(void) {
  @try {
    if (screenieOverlayAppDeactivateObserver != nil) {
      [[NSNotificationCenter defaultCenter]
          removeObserver:screenieOverlayAppDeactivateObserver];
      screenieOverlayAppDeactivateObserver = nil;
    }
    if (screenieOverlaySpaceObserver != nil) {
      [[[NSWorkspace sharedWorkspace] notificationCenter]
          removeObserver:screenieOverlaySpaceObserver];
      screenieOverlaySpaceObserver = nil;
    }
    if (screenieOverlayGestureLocalMonitor != nil) {
      [NSEvent removeMonitor:screenieOverlayGestureLocalMonitor];
      screenieOverlayGestureLocalMonitor = nil;
    }
    if (screenieOverlayGestureGlobalMonitor != nil) {
      [NSEvent removeMonitor:screenieOverlayGestureGlobalMonitor];
      screenieOverlayGestureGlobalMonitor = nil;
    }
    screenieOverlayBackgroundChangedCallback = NULL;
    screenie_stop_overlay_background_poll_timer();
    screenie_clear_overlay_mouse_passthrough();
  } @catch (NSException *exception) {
    NSLog(@"[screenie] uninstall deactivate hider exception: %@ %@",
          [exception name], [exception reason]);
  }
}

// ============================================================================
// Per-panel vibrancy regions — live frosted glass for the overlay panels.
//
// We mount NSVisualEffectViews as subviews of the overlay window's
// contentView, positioned BEHIND the WKWebView in z-order. The WebView is
// transparent everywhere except where it renders the panel content (text,
// buttons, borders). The vibrancy view shows live-blurred desktop pixels
// underneath, so moving windows behind the overlay updates the frost in
// real time — no React renders, no IPC roundtrips, no SCK captures.
//
// React drives the position+size of each region via
// `screenie_set_overlay_vibrancy_regions`, called every layout pass with
// the current viewport rects of every frosted panel. The native side
// pools / reuses NSVisualEffectViews so we don't churn the view hierarchy
// every render.
//
// Each region is given a corner radius so the vibrancy clips to match the
// React-side rounded panel.
// ============================================================================

typedef struct {
  double x;       // viewport left in CSS pixels (top-left origin)
  double y;       // viewport top in CSS pixels
  double w;       // width in CSS pixels
  double h;       // height in CSS pixels
  double radius;  // corner radius in CSS pixels (uniform)
} ScreenieOverlayVibrancyRegion;

static NSMutableArray<NSVisualEffectView *> *screenieOverlayVibrancyViews = nil;

bool screenie_set_overlay_vibrancy_regions(
    void *window_ptr,
    const ScreenieOverlayVibrancyRegion *regions,
    size_t count) {
  @autoreleasepool {
  @try {
    if (window_ptr == NULL) {
      return false;
    }
    NSWindow *window = (NSWindow *)window_ptr;
    NSView *contentView = [window contentView];
    if (contentView == nil) {
      return false;
    }
    if (screenieOverlayVibrancyViews == nil) {
      // MRC: `[NSMutableArray array]` is autoreleased and would be freed
      // at the next runloop iteration, leaving the static pointer
      // dangling — a future call would crash or, worse, send a setFrame:
      // to whatever object reused that memory (we hit a
      // `_CFPasteboardEntry setFrame:` exception with `array` here).
      // `[[NSMutableArray alloc] init]` keeps the +1 retain forever,
      // which is what we want for an app-lifetime singleton.
      screenieOverlayVibrancyViews = [[NSMutableArray alloc] init];
    }

    // Pool: ensure exactly `count` views exist. NSVisualEffectViews
    // added directly to contentView (no wrapper layer between them
    // and the host window — `wantsLayer = YES` on a wrapper breaks
    // the system's `behindWindow` vibrancy composition).
    while (screenieOverlayVibrancyViews.count < count) {
      NSVisualEffectView *v =
          [[NSVisualEffectView alloc] initWithFrame:NSZeroRect];
      // Real vibrancy materials (the ones that actually blur the
      // desktop with `behindWindow` blending on a transparent host
      // window): `sidebar`, `popover`, `menu`, `hudWindow`. Order
      // roughly heaviest → lightest: hudWindow > popover > sidebar >
      // menu. The "lighter" materials in NSVisualEffectMaterial
      // (titlebar, contentBackground, underWindowBackground) are
      // window-content fills, NOT vibrancy backers — they render as
      // opaque solid panels here. `sidebar` is the working baseline.
      // HUDWindow = the heaviest practical vibrancy material on macOS.
      // Matches the ChatGPT / Linear / Notion floating-panel look — colors
      // bleed through but content is gaussian-blurred to soft blobs.
      v.material = NSVisualEffectMaterialHUDWindow;
      // `behindWindow` blurs what is BEHIND the window in the OS
      // compositor (the desktop + other apps' windows).
      v.blendingMode = NSVisualEffectBlendingModeBehindWindow;
      // `Active` keeps the vibrancy on regardless of window focus.
      // `Inactive` literally turns the effect OFF (despite the name)
      // and the view falls back to a solid NSView. Don't use it.
      v.state = NSVisualEffectStateActive;
      v.wantsLayer = YES;
      v.layer.masksToBounds = YES;
      // `NSWindowBelow` with a nil reference puts the view at the
      // bottom of the contentView's subview stack — behind the
      // WKWebView Tauri added earlier. WebView stays on top.
      [contentView addSubview:v positioned:NSWindowBelow relativeTo:nil];
      [screenieOverlayVibrancyViews addObject:v];
      // MRC: balance the +1 from `alloc`. Array + contentView each
      // retain.
      [v release];
    }
    while (screenieOverlayVibrancyViews.count > count) {
      NSView *v = [screenieOverlayVibrancyViews lastObject];
      [v removeFromSuperview];
      [screenieOverlayVibrancyViews removeLastObject];
    }

    // Update geometry. JS rects use top-left origin in CSS pixels; the
    // contentView's coordinate space depends on `isFlipped`. WKWebView's
    // host view is typically NOT flipped, so we Y-flip.
    CGFloat ch = NSHeight([contentView bounds]);
    BOOL flipped = [contentView isFlipped];

    for (size_t i = 0; i < count; i++) {
      ScreenieOverlayVibrancyRegion r = regions[i];
      NSView *v = screenieOverlayVibrancyViews[i];
      if (!isfinite(r.x) || !isfinite(r.y) || !isfinite(r.w) ||
          !isfinite(r.h) || r.w <= 0.0 || r.h <= 0.0) {
        // Park off-screen instead of removing from the pool, so we
        // don't churn the view hierarchy on every edge case.
        [v setFrame:NSMakeRect(-1, -1, 1, 1)];
        continue;
      }
      CGFloat y = flipped ? r.y : (ch - r.y - r.h);
      NSRect frame = NSMakeRect(r.x, y, r.w, r.h);
      [v setFrame:frame];
      // Clamp corner radius to half the shorter side; values like
      // 9999 (pill) are intentional shorthand for "fully rounded".
      double maxRadius = MIN(r.w, r.h) / 2.0;
      double radius = MAX(0.0, MIN(r.radius, maxRadius));
      v.layer.cornerRadius = radius;
    }
    return true;
  } @catch (NSException *exception) {
    NSLog(@"[screenie] set vibrancy regions exception: %@ %@",
          [exception name], [exception reason]);
    return false;
  }
  } // @autoreleasepool
}

void screenie_clear_overlay_vibrancy_regions(void) {
  @autoreleasepool {
  @try {
    if (screenieOverlayVibrancyViews == nil) return;
    for (NSVisualEffectView *v in screenieOverlayVibrancyViews) {
      [v removeFromSuperview];
    }
    [screenieOverlayVibrancyViews removeAllObjects];
  } @catch (NSException *exception) {
    NSLog(@"[screenie] clear vibrancy exception: %@ %@",
          [exception name], [exception reason]);
  }
  } // @autoreleasepool
}
