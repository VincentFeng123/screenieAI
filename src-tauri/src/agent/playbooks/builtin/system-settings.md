---
name: system-settings
apps: com.apple.systempreferences, System Settings
triggers: settings, preferences, enable, turn on, turn off, wifi, bluetooth, display, sound, notifications, focus, do not disturb
---
Open a specific System Settings pane in ONE step with openUrl and an
x-apple.systempreferences: URL instead of clicking through the sidebar.

## Pane URLs
- Wi-Fi: x-apple.systempreferences:com.apple.wifi-settings-extension
- Bluetooth: x-apple.systempreferences:com.apple.BluetoothSettings
- Displays: x-apple.systempreferences:com.apple.Displays-Settings.extension
- Sound: x-apple.systempreferences:com.apple.Sound-Settings.extension
- Notifications: x-apple.systempreferences:com.apple.Notifications-Settings.extension
- Focus: x-apple.systempreferences:com.apple.Focus-Settings.extension
- Privacy & Security: x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension
- Screen Recording permission: x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture
- Accessibility permission: x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility
- General > Login Items: x-apple.systempreferences:com.apple.LoginItems-Settings.extension

## Inside a pane
After the pane opens, use the observation's toggles and controls directly.
If the setting is not visible, findUi its name before scrolling blindly; the
sidebar search field also accepts a typed setting name.
