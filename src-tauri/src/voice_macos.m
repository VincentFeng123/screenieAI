// Microphone-permission preflight for voice command mode.
//
// cpal/CoreAudio delivers *silent zeros* (not an error) when microphone
// permission is denied, so the Rust side must consult TCC explicitly before
// starting a listening session. Consumed by the extern "C" block in
// src/voice/capture.rs. Kept stateless, matching the macos_window.m
// convention: all logical state lives in Rust's AppState.

#import <AVFoundation/AVFoundation.h>
#import <stdbool.h>

// AVAuthorizationStatus mapped to a plain int for FFI:
// 0 = notDetermined, 1 = restricted, 2 = denied, 3 = authorized.
int screenie_voice_mic_auth_status(void) {
  AVAuthorizationStatus status =
      [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
  switch (status) {
    case AVAuthorizationStatusNotDetermined:
      return 0;
    case AVAuthorizationStatusRestricted:
      return 1;
    case AVAuthorizationStatusDenied:
      return 2;
    case AVAuthorizationStatusAuthorized:
      return 3;
  }
  return 2;
}

// Triggers the TCC prompt when status is notDetermined. The completion
// handler runs on an arbitrary AVFoundation queue; the Rust trampoline owns
// `ctx` (a boxed channel sender) and consumes it exactly once.
void screenie_voice_request_mic_access(void (*cb)(bool granted, void *ctx),
                                       void *ctx) {
  [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio
                           completionHandler:^(BOOL granted) {
                             if (cb != NULL) {
                               cb(granted ? true : false, ctx);
                             }
                           }];
}
