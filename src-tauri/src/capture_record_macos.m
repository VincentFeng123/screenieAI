// capture_record_macos.m — ScreenCaptureKit capture/recording bridge.
//
// Engine choice (per docs/ + plan): hand-written ObjC compiled by `cc` in
// build.rs, matching macos_window.m / voice_macos.m — NOT objc2 crate
// bindings, which would introduce a second SCK access pattern alongside this
// repo's established bridge. Recording uses SCStream → AVAssetWriter (H.264),
// which is uniformly available on macOS 12.3+; SCRecordingOutput (15+) is
// deliberately avoided so there is exactly one recorder code path. Single
// stills (screenie_capture_target_png) use SCScreenshotManager and are gated
// on macOS 14+; Rust falls back to the `screencapture` CLI below that.
//
// Threading/state contract (same as voice_macos.m): this file is mechanism
// only. Logical session state — which session is active, duration/disk caps,
// timers — lives in Rust's AppState. Each recording is one retained opaque
// ScreenieRecorder handle; Rust polls state/error, then calls stop + release.
// Strings returned as malloc'd UTF-8 are freed with the existing
// screenie_free_string from macos_window.m (one binary, one malloc zone).

#import <AppKit/AppKit.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

// Defined in macos_window.m (non-static).
extern void screenie_free_string(const char *ptr);

// SCShareableContent enumeration can be slow right after a fresh TCC grant.
static const int64_t kScreenieContentTimeoutSeconds = 5;
static const int64_t kScreenieStreamStartTimeoutSeconds = 5;
static const int64_t kScreenieStreamStopTimeoutSeconds = 3;
static const int64_t kScreenieWriterFinalizeTimeoutSeconds = 8;

#pragma mark - Probes

bool screenie_screenshot_api_available(void) {
  if (@available(macOS 14.0, *)) {
    return true;
  }
  return false;
}

// Non-prompting Accessibility probe (macos_window.m only has the prompting
// variant). Used by the capture_permission map.
bool screenie_has_accessibility_access(void) {
  return AXIsProcessTrusted();
}

#pragma mark - Shareable content helpers

// Synchronously fetch shareable content. Returns nil on error/timeout and
// fills *error_text (malloc'd) when provided.
static SCShareableContent *screenie_fetch_shareable_content(char **error_text) {
  __block SCShareableContent *result = nil;
  __block char *errText = NULL;
  dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

  [SCShareableContent
      getShareableContentExcludingDesktopWindows:NO
                              onScreenWindowsOnly:YES
                                completionHandler:^(SCShareableContent *content,
                                                    NSError *error) {
    if (error != nil || content == nil) {
      NSString *msg = error != nil ? [error localizedDescription]
                                   : @"no shareable content";
      const char *utf8 = [msg UTF8String];
      errText = utf8 != NULL ? strdup(utf8) : strdup("shareable content failed");
    } else {
      result = content;
    }
    dispatch_semaphore_signal(semaphore);
  }];

  dispatch_time_t timeout = dispatch_time(
      DISPATCH_TIME_NOW, kScreenieContentTimeoutSeconds * NSEC_PER_SEC);
  if (dispatch_semaphore_wait(semaphore, timeout) != 0) {
    if (error_text != NULL) {
      *error_text = strdup("shareable content enumeration timed out");
    }
    return nil;
  }
  if (result == nil && error_text != NULL) {
    *error_text = errText;
  } else if (errText != NULL) {
    free(errText);
  }
  return result;
}

static SCDisplay *screenie_record_find_display(SCShareableContent *content,
                                               CGDirectDisplayID displayID) {
  for (SCDisplay *display in content.displays) {
    if (displayID != 0 && display.displayID == displayID) {
      return display;
    }
  }
  return content.displays.firstObject;
}

static SCWindow *screenie_record_find_window(SCShareableContent *content,
                                             uint32_t windowID) {
  for (SCWindow *window in content.windows) {
    if ((uint32_t)window.windowID == windowID) {
      return window;
    }
  }
  return nil;
}

static SCRunningApplication *screenie_record_find_self(
    SCShareableContent *content) {
  pid_t ownPid = getpid();
  for (SCRunningApplication *application in content.applications) {
    if (application.processID == ownPid) {
      return application;
    }
  }
  return nil;
}

// Display filter, optionally excluding this app's windows (overlay, tooltip)
// so recordings show the user's screen rather than our own chrome.
static SCContentFilter *screenie_record_display_filter(
    SCShareableContent *content, CGDirectDisplayID displayID,
    bool excludeSelf) {
  SCDisplay *display = screenie_record_find_display(content, displayID);
  if (display == nil) {
    return nil;
  }
  if (excludeSelf) {
    SCRunningApplication *own = screenie_record_find_self(content);
    if (own != nil) {
      return [[SCContentFilter alloc] initWithDisplay:display
                                excludingApplications:@[ own ]
                                     exceptingWindows:@[]];
    }
  }
  return [[SCContentFilter alloc] initWithDisplay:display
                            excludingApplications:@[]
                                 exceptingWindows:@[]];
}

#pragma mark - Target enumeration

// malloc'd JSON:
// {"displays":[{"id":u32,"x":pt,"y":pt,"width":pt,"height":pt,
//               "pixelWidth":px,"pixelHeight":px}],
//  "windows":[{"id":u32,"title":str,"app":str,"pid":i32,
//              "x":pt,"y":pt,"width":pt,"height":pt,
//              "onScreen":bool,"layer":i32}]}
// Coordinates are global logical points (CG coordinate space, origin top-left
// of the main display) — the same space as capture::capture_rect.
const char *screenie_capture_list_targets(void) {
  @autoreleasepool {
    SCShareableContent *content = screenie_fetch_shareable_content(NULL);
    if (content == nil) {
      return NULL;
    }

    NSMutableArray *displays = [NSMutableArray array];
    for (SCDisplay *display in content.displays) {
      CGRect frame = display.frame;
      size_t pixelW = 0;
      size_t pixelH = 0;
      CGDisplayModeRef mode = CGDisplayCopyDisplayMode(display.displayID);
      if (mode != NULL) {
        pixelW = CGDisplayModeGetPixelWidth(mode);
        pixelH = CGDisplayModeGetPixelHeight(mode);
        CGDisplayModeRelease(mode);
      }
      [displays addObject:@{
        @"id" : @(display.displayID),
        @"x" : @(frame.origin.x),
        @"y" : @(frame.origin.y),
        @"width" : @(frame.size.width),
        @"height" : @(frame.size.height),
        @"pixelWidth" : @(pixelW),
        @"pixelHeight" : @(pixelH),
      }];
    }

    NSMutableArray *windows = [NSMutableArray array];
    for (SCWindow *window in content.windows) {
      CGRect frame = window.frame;
      if (frame.size.width <= 1.0 || frame.size.height <= 1.0) {
        continue;
      }
      NSString *title = window.title != nil ? window.title : @"";
      SCRunningApplication *app = window.owningApplication;
      NSString *appName = app != nil ? app.applicationName : @"";
      int32_t pid = app != nil ? (int32_t)app.processID : 0;
      [windows addObject:@{
        @"id" : @((uint32_t)window.windowID),
        @"title" : title,
        @"app" : appName,
        @"pid" : @(pid),
        @"x" : @(frame.origin.x),
        @"y" : @(frame.origin.y),
        @"width" : @(frame.size.width),
        @"height" : @(frame.size.height),
        @"onScreen" : @(window.isOnScreen),
        @"layer" : @((int32_t)window.windowLayer),
      }];
    }

    NSDictionary *payload = @{ @"displays" : displays, @"windows" : windows };
    NSError *jsonError = nil;
    NSData *json = [NSJSONSerialization dataWithJSONObject:payload
                                                   options:0
                                                     error:&jsonError];
    if (json == nil || jsonError != nil) {
      return NULL;
    }
    NSString *str = [[NSString alloc] initWithData:json
                                          encoding:NSUTF8StringEncoding];
    const char *utf8 = [str UTF8String];
    return utf8 != NULL ? strdup(utf8) : NULL;
  }
}

#pragma mark - Single still (macOS 14+)

// ARC sibling of macos_window.m's static screenie_copy_png_base64_from_image.
static char *screenie_capture_copy_png_base64(CGImageRef image) {
  if (image == NULL) {
    return NULL;
  }
  NSMutableData *data = [NSMutableData data];
  CGImageDestinationRef dest = CGImageDestinationCreateWithData(
      (__bridge CFMutableDataRef)data, CFSTR("public.png"), 1, NULL);
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

// One still of a display (window_id == 0) or a window. Returns malloc'd
// base64 PNG, or NULL on failure / below macOS 14 (callers check
// screenie_screenshot_api_available first and use the screencapture-CLI
// fallback). max_dimension > 0 downscales the long edge at the SCK layer —
// GPU-side, so native 5K pixels never cross the FFI boundary.
const char *screenie_capture_target_png(uint32_t display_id, uint32_t window_id,
                                        uint32_t max_dimension,
                                        bool exclude_self, bool show_cursor) {
  if (@available(macOS 14.0, *)) {
    @autoreleasepool {
      SCShareableContent *content = screenie_fetch_shareable_content(NULL);
      if (content == nil) {
        return NULL;
      }

      SCContentFilter *filter = nil;
      if (window_id != 0) {
        SCWindow *window = screenie_record_find_window(content, window_id);
        if (window == nil) {
          return NULL;
        }
        filter =
            [[SCContentFilter alloc] initWithDesktopIndependentWindow:window];
      } else {
        filter = screenie_record_display_filter(
            content, (CGDirectDisplayID)display_id, exclude_self);
      }
      if (filter == nil) {
        return NULL;
      }

      CGSize contentSize = filter.contentRect.size;
      double scale = (double)filter.pointPixelScale;
      double nativeW = contentSize.width * scale;
      double nativeH = contentSize.height * scale;
      if (nativeW < 1.0 || nativeH < 1.0) {
        return NULL;
      }
      double outW = nativeW;
      double outH = nativeH;
      if (max_dimension > 0) {
        double longEdge = MAX(nativeW, nativeH);
        if (longEdge > (double)max_dimension) {
          double s = (double)max_dimension / longEdge;
          outW = MAX(1.0, floor(nativeW * s));
          outH = MAX(1.0, floor(nativeH * s));
        }
      }

      SCStreamConfiguration *config = [[SCStreamConfiguration alloc] init];
      config.width = (size_t)outW;
      config.height = (size_t)outH;
      config.scalesToFit = YES;
      config.showsCursor = show_cursor;

      __block char *result = NULL;
      dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
      [SCScreenshotManager
          captureImageWithFilter:filter
                   configuration:config
               completionHandler:^(CGImageRef image, NSError *error) {
        if (error == nil && image != NULL) {
          result = screenie_capture_copy_png_base64(image);
        }
        dispatch_semaphore_signal(semaphore);
      }];
      dispatch_time_t timeout =
          dispatch_time(DISPATCH_TIME_NOW, 3 * NSEC_PER_SEC);
      if (dispatch_semaphore_wait(semaphore, timeout) != 0) {
        // Same accepted trade-off as macos_window.m: on the (rare) timeout a
        // late completion writes into block storage the block itself owns;
        // the strdup'd string leaks but nothing dangles.
        return NULL;
      }
      return result;
    }
  }
  return NULL;
}

#pragma mark - Recording

typedef struct ScreenieRecordConfig {
  uint32_t display_id;  // 0 = main/first display
  uint32_t window_id;   // 0 = display mode
  // SCStreamConfiguration.sourceRect in POINTS, relative to the display
  // origin. src_w <= 0 -> full display (no sourceRect).
  double src_x;
  double src_y;
  double src_w;
  double src_h;
  uint32_t fps;        // minimumFrameInterval = 1/fps
  uint32_t out_width;  // output PIXELS; Rust pre-computes (even, scaled)
  uint32_t out_height;
  bool show_cursor;
  bool exclude_self;  // display mode only; hide our own overlay windows
} ScreenieRecordConfig;

// GIF-mode frame tap. Fires on the SCStream sample queue — the Rust side MUST
// copy the pixels and return immediately; blocking here stalls capture.
typedef void (*ScreenieFrameCallback)(const uint8_t *bgra, size_t len,
                                      uint32_t width, uint32_t height,
                                      size_t bytes_per_row, double pts_seconds,
                                      void *ctx);

typedef NS_ENUM(int, ScreenieRecorderState) {
  ScreenieRecorderStateRecording = 0,
  ScreenieRecorderStateStopped = 1,
  ScreenieRecorderStateFailed = 2,
};

@interface ScreenieRecorder : NSObject <SCStreamOutput, SCStreamDelegate>
@property(nonatomic, strong) SCStream *stream;
@property(nonatomic, strong) AVAssetWriter *writer;
@property(nonatomic, strong) AVAssetWriterInput *videoInput;
@property(nonatomic, strong) dispatch_queue_t sampleQueue;
@property(atomic) int state;
@property(atomic) uint64_t frameCount;
// Accessed only on sampleQueue.
@property(nonatomic) BOOL sessionStarted;
@property(nonatomic) ScreenieFrameCallback frameCb;
@property(nonatomic) void *frameCtx;
@property(atomic) BOOL stopRequested;
@end

@implementation ScreenieRecorder {
  NSString *_errorText;  // guarded by @synchronized(self)
}

- (void)setErrorText:(NSString *)text {
  @synchronized(self) {
    if (_errorText == nil) {
      _errorText = [text copy];
    }
  }
}

- (NSString *)errorTextCopy {
  @synchronized(self) {
    return _errorText;
  }
}

- (void)failWith:(NSString *)text {
  [self setErrorText:text];
  self.state = ScreenieRecorderStateFailed;
}

- (void)stream:(SCStream *)stream
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
                   ofType:(SCStreamOutputType)type {
  if (type != SCStreamOutputTypeScreen ||
      self.state != ScreenieRecorderStateRecording || self.stopRequested) {
    return;
  }
  if (!CMSampleBufferDataIsReady(sampleBuffer)) {
    return;
  }

  // SCK delivers idle/blank status frames; only complete frames carry pixels.
  CFArrayRef attachmentsArray =
      CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, false);
  if (attachmentsArray == NULL || CFArrayGetCount(attachmentsArray) == 0) {
    return;
  }
  CFDictionaryRef attachments = CFArrayGetValueAtIndex(attachmentsArray, 0);
  CFTypeRef statusRef = CFDictionaryGetValue(
      attachments, (__bridge CFStringRef)SCStreamFrameInfoStatus);
  if (statusRef == NULL) {
    return;
  }
  int status = -1;
  if (!CFNumberGetValue((CFNumberRef)statusRef, kCFNumberIntType, &status) ||
      status != SCFrameStatusComplete) {
    return;
  }

  if (self.writer != nil) {
    if (self.writer.status == AVAssetWriterStatusFailed) {
      NSString *msg = self.writer.error != nil
                          ? [self.writer.error localizedDescription]
                          : @"asset writer failed";
      [self failWith:msg];
      return;
    }
    if (!self.sessionStarted) {
      [self.writer startSessionAtSourceTime:CMSampleBufferGetPresentationTimeStamp(
                                                sampleBuffer)];
      self.sessionStarted = YES;
    }
    // Real-time policy: drop frames rather than block the SCK queue.
    if (self.videoInput.readyForMoreMediaData) {
      if ([self.videoInput appendSampleBuffer:sampleBuffer]) {
        self.frameCount += 1;
      } else {
        NSString *msg = self.writer.error != nil
                            ? [self.writer.error localizedDescription]
                            : @"appendSampleBuffer failed";
        [self failWith:msg];
      }
    }
    return;
  }

  if (self.frameCb != NULL) {
    CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (pixelBuffer == NULL) {
      return;
    }
    if (CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly) !=
        kCVReturnSuccess) {
      return;
    }
    const uint8_t *base = CVPixelBufferGetBaseAddress(pixelBuffer);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
    size_t width = CVPixelBufferGetWidth(pixelBuffer);
    size_t height = CVPixelBufferGetHeight(pixelBuffer);
    if (base != NULL && width > 0 && height > 0) {
      double pts = CMTimeGetSeconds(
          CMSampleBufferGetPresentationTimeStamp(sampleBuffer));
      self.frameCb(base, bytesPerRow * height, (uint32_t)width,
                   (uint32_t)height, bytesPerRow, pts, self.frameCtx);
      self.frameCount += 1;
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
  }
}

- (void)stream:(SCStream *)stream didStopWithError:(NSError *)error {
  if (self.state != ScreenieRecorderStateRecording || self.stopRequested) {
    return;
  }
  NSString *msg;
  if (error != nil && error.code == SCStreamErrorUserStopped) {
    msg = @"recording stopped via the system screen-sharing indicator";
  } else if (error != nil) {
    msg = [NSString stringWithFormat:@"capture stream stopped: %@ (%ld)",
                                     [error localizedDescription],
                                     (long)error.code];
  } else {
    msg = @"capture stream stopped unexpectedly";
  }
  [self failWith:msg];
}

@end

static void screenie_record_set_error(char **error_out, NSString *message) {
  if (error_out == NULL) {
    return;
  }
  const char *utf8 = [message UTF8String];
  *error_out = utf8 != NULL ? strdup(utf8) : strdup("recording error");
}

// Start a recording. Exactly one of out_path_utf8 (mp4 mode) / frame_cb (gif
// frame-tap mode) must be non-NULL. Returns a RETAINED opaque handle, or NULL
// with *error_out set (malloc'd; free with screenie_free_string).
void *screenie_recording_start(const ScreenieRecordConfig *cfg,
                               const char *out_path_utf8,
                               ScreenieFrameCallback frame_cb, void *frame_ctx,
                               char **error_out) {
  @autoreleasepool {
    if (cfg == NULL || cfg->fps == 0 || cfg->out_width == 0 ||
        cfg->out_height == 0 ||
        ((out_path_utf8 == NULL) == (frame_cb == NULL))) {
      screenie_record_set_error(error_out, @"invalid recording configuration");
      return NULL;
    }

    char *contentError = NULL;
    SCShareableContent *content =
        screenie_fetch_shareable_content(&contentError);
    if (content == nil) {
      if (error_out != NULL && contentError != NULL) {
        *error_out = contentError;
      } else {
        if (contentError != NULL) {
          free(contentError);
        }
        screenie_record_set_error(error_out, @"shareable content unavailable");
      }
      return NULL;
    }

    SCContentFilter *filter = nil;
    if (cfg->window_id != 0) {
      SCWindow *window = screenie_record_find_window(content, cfg->window_id);
      if (window == nil) {
        screenie_record_set_error(
            error_out, @"target window not found (it may have closed)");
        return NULL;
      }
      filter = [[SCContentFilter alloc] initWithDesktopIndependentWindow:window];
    } else {
      filter = screenie_record_display_filter(
          content, (CGDirectDisplayID)cfg->display_id, cfg->exclude_self);
    }
    if (filter == nil) {
      screenie_record_set_error(error_out, @"no capture target available");
      return NULL;
    }

    SCStreamConfiguration *config = [[SCStreamConfiguration alloc] init];
    config.width = cfg->out_width;
    config.height = cfg->out_height;
    config.pixelFormat = kCVPixelFormatType_32BGRA;
    config.minimumFrameInterval = CMTimeMake(1, (int32_t)cfg->fps);
    config.queueDepth = 8;
    config.showsCursor = cfg->show_cursor;
    config.scalesToFit = YES;
    if (cfg->window_id == 0 && cfg->src_w > 0.0 && cfg->src_h > 0.0) {
      config.sourceRect =
          CGRectMake(cfg->src_x, cfg->src_y, cfg->src_w, cfg->src_h);
    }

    ScreenieRecorder *recorder = [[ScreenieRecorder alloc] init];
    recorder.sampleQueue = dispatch_queue_create(
        "com.screenieai.capture.samples", DISPATCH_QUEUE_SERIAL);
    recorder.state = ScreenieRecorderStateRecording;
    recorder.frameCb = frame_cb;
    recorder.frameCtx = frame_ctx;

    if (out_path_utf8 != NULL) {
      NSString *path = [NSString stringWithUTF8String:out_path_utf8];
      if (path == nil) {
        screenie_record_set_error(error_out, @"invalid output path");
        return NULL;
      }
      NSURL *url = [NSURL fileURLWithPath:path];
      NSError *writerError = nil;
      AVAssetWriter *writer = [[AVAssetWriter alloc] initWithURL:url
                                                        fileType:AVFileTypeMPEG4
                                                           error:&writerError];
      if (writer == nil || writerError != nil) {
        NSString *msg = writerError != nil
                            ? [writerError localizedDescription]
                            : @"failed to create asset writer";
        screenie_record_set_error(error_out, msg);
        return NULL;
      }

      // Bitrate heuristic: ~0.1 bit/pixel/frame keeps screen content crisp
      // without ballooning files (1920x1080@10fps ≈ 2.0 Mbit/s).
      double bitrate = (double)cfg->out_width * (double)cfg->out_height *
                       (double)cfg->fps * 0.1;
      if (bitrate < 500000.0) {
        bitrate = 500000.0;
      }
      NSDictionary *settings = @{
        AVVideoCodecKey : AVVideoCodecTypeH264,
        AVVideoWidthKey : @(cfg->out_width),
        AVVideoHeightKey : @(cfg->out_height),
        AVVideoCompressionPropertiesKey : @{
          AVVideoAverageBitRateKey : @((NSInteger)bitrate),
          AVVideoMaxKeyFrameIntervalKey : @((NSInteger)(cfg->fps * 2)),
        },
      };
      AVAssetWriterInput *input =
          [[AVAssetWriterInput alloc] initWithMediaType:AVMediaTypeVideo
                                         outputSettings:settings];
      input.expectsMediaDataInRealTime = YES;
      if (![writer canAddInput:input]) {
        screenie_record_set_error(error_out, @"writer rejected video input");
        return NULL;
      }
      [writer addInput:input];
      if (![writer startWriting]) {
        NSString *msg = writer.error != nil
                            ? [writer.error localizedDescription]
                            : @"startWriting failed";
        screenie_record_set_error(error_out, msg);
        return NULL;
      }
      recorder.writer = writer;
      recorder.videoInput = input;
    }

    NSError *streamInitError = nil;
    SCStream *stream = [[SCStream alloc] initWithFilter:filter
                                          configuration:config
                                               delegate:recorder];
    if (stream == nil) {
      [recorder.writer cancelWriting];
      screenie_record_set_error(error_out, @"failed to create capture stream");
      return NULL;
    }
    if (![stream addStreamOutput:recorder
                            type:SCStreamOutputTypeScreen
              sampleHandlerQueue:recorder.sampleQueue
                           error:&streamInitError]) {
      [recorder.writer cancelWriting];
      NSString *msg = streamInitError != nil
                          ? [streamInitError localizedDescription]
                          : @"failed to attach stream output";
      screenie_record_set_error(error_out, msg);
      return NULL;
    }
    recorder.stream = stream;

    __block NSError *startError = nil;
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
    [stream startCaptureWithCompletionHandler:^(NSError *error) {
      startError = error;
      dispatch_semaphore_signal(semaphore);
    }];
    dispatch_time_t timeout = dispatch_time(
        DISPATCH_TIME_NOW, kScreenieStreamStartTimeoutSeconds * NSEC_PER_SEC);
    if (dispatch_semaphore_wait(semaphore, timeout) != 0) {
      [recorder.writer cancelWriting];
      screenie_record_set_error(error_out, @"capture start timed out");
      return NULL;
    }
    if (startError != nil) {
      [recorder.writer cancelWriting];
      screenie_record_set_error(
          error_out,
          [NSString stringWithFormat:@"capture start failed: %@ (%ld)",
                                     [startError localizedDescription],
                                     (long)startError.code]);
      return NULL;
    }

    return (void *)CFBridgingRetain(recorder);
  }
}

// 0 = recording, 1 = stopped, 2 = failed.
int screenie_recording_state(void *handle) {
  if (handle == NULL) {
    return ScreenieRecorderStateFailed;
  }
  ScreenieRecorder *recorder = (__bridge ScreenieRecorder *)handle;
  return recorder.state;
}

// malloc'd copy of the failure message, or NULL. Free with
// screenie_free_string.
const char *screenie_recording_error(void *handle) {
  if (handle == NULL) {
    return NULL;
  }
  ScreenieRecorder *recorder = (__bridge ScreenieRecorder *)handle;
  NSString *text = [recorder errorTextCopy];
  if (text == nil) {
    return NULL;
  }
  const char *utf8 = [text UTF8String];
  return utf8 != NULL ? strdup(utf8) : NULL;
}

uint64_t screenie_recording_frame_count(void *handle) {
  if (handle == NULL) {
    return 0;
  }
  ScreenieRecorder *recorder = (__bridge ScreenieRecorder *)handle;
  return recorder.frameCount;
}

// Stop capture and finalize the output. Blocking but bounded (~3s stream stop
// + ~8s writer finalize). Idempotent; safe after a native failure. After this
// returns, no further frame callbacks fire (the sample queue is drained), so
// the caller may reclaim frame_ctx once stop AND release have returned.
bool screenie_recording_stop(void *handle, char **error_out) {
  if (handle == NULL) {
    screenie_record_set_error(error_out, @"null recording handle");
    return false;
  }
  @autoreleasepool {
    ScreenieRecorder *recorder = (__bridge ScreenieRecorder *)handle;
    bool alreadyStopping = recorder.stopRequested;
    recorder.stopRequested = YES;

    if (!alreadyStopping && recorder.stream != nil) {
      dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
      [recorder.stream stopCaptureWithCompletionHandler:^(NSError *error) {
        // "Already stopped" errors are expected when the system or a native
        // failure tore the stream down first; the writer result decides.
        (void)error;
        dispatch_semaphore_signal(semaphore);
      }];
      dispatch_time_t timeout = dispatch_time(
          DISPATCH_TIME_NOW, kScreenieStreamStopTimeoutSeconds * NSEC_PER_SEC);
      dispatch_semaphore_wait(semaphore, timeout);
    }

    // Drain in-flight sample callbacks so no frame_cb fires after stop and
    // the writer sees no appends after markAsFinished.
    if (recorder.sampleQueue != nil) {
      dispatch_sync(recorder.sampleQueue, ^{
        recorder.frameCb = NULL;
        recorder.frameCtx = NULL;
      });
    }

    bool wasFailed = recorder.state == ScreenieRecorderStateFailed;
    if (!wasFailed) {
      recorder.state = ScreenieRecorderStateStopped;
    }

    AVAssetWriter *writer = recorder.writer;
    if (writer == nil) {
      // GIF frame-tap mode: nothing to finalize natively. A stream that died
      // early is still salvageable (Rust decides via the encoder's frame
      // count); pass the cause through error_out either way.
      if (wasFailed) {
        NSString *text = [recorder errorTextCopy];
        screenie_record_set_error(
            error_out, text != nil ? text : @"recording failed");
      }
      return true;
    }

    __block BOOL sessionStarted = NO;
    dispatch_sync(recorder.sampleQueue, ^{
      sessionStarted = recorder.sessionStarted;
    });

    if (writer.status == AVAssetWriterStatusFailed) {
      NSString *msg = writer.error != nil
                          ? [writer.error localizedDescription]
                          : @"asset writer failed";
      screenie_record_set_error(error_out, msg);
      return false;
    }
    if (!sessionStarted) {
      [writer cancelWriting];
      screenie_record_set_error(
          error_out,
          @"no frames were captured (zero complete frames delivered)");
      return false;
    }

    [recorder.videoInput markAsFinished];
    __block BOOL finalized = NO;
    dispatch_semaphore_t finishSemaphore = dispatch_semaphore_create(0);
    [writer finishWritingWithCompletionHandler:^{
      finalized = writer.status == AVAssetWriterStatusCompleted;
      dispatch_semaphore_signal(finishSemaphore);
    }];
    dispatch_time_t finishTimeout =
        dispatch_time(DISPATCH_TIME_NOW,
                      kScreenieWriterFinalizeTimeoutSeconds * NSEC_PER_SEC);
    if (dispatch_semaphore_wait(finishSemaphore, finishTimeout) != 0) {
      screenie_record_set_error(error_out, @"writer finalize timed out");
      return false;
    }
    if (!finalized) {
      NSString *msg = writer.error != nil
                          ? [writer.error localizedDescription]
                          : @"writer did not complete";
      screenie_record_set_error(error_out, msg);
      return false;
    }

    if (wasFailed) {
      // The file finalized, but the stream died early (display unplugged,
      // user hit the system stop button). Salvageable partial clip: report
      // success WITH the cause in error_out so Rust can surface it as the
      // stop reason rather than discarding the file.
      NSString *text = [recorder errorTextCopy];
      screenie_record_set_error(error_out,
                                text != nil ? text : @"recording interrupted");
    }
    return true;
  }
}

// Balance the retain from screenie_recording_start. Call only after stop has
// returned; afterwards no callbacks fire and frame_ctx may be reclaimed.
void screenie_recording_release(void *handle) {
  if (handle != NULL) {
    CFBridgingRelease(handle);
  }
}
