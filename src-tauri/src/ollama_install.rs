//! One-click Ollama installer for macOS.
//!
//! Strategy: pull the official `Ollama-darwin.zip`, unzip it, drop the
//! `Ollama.app` into `/Applications` (or `~/Applications` if the system
//! Applications folder isn't writable), then launch it. macOS will pop the
//! usual one-time admin prompt for Ollama's own CLI / daemon setup — that's
//! Ollama's problem and we can't bypass it.
//!
//! Progress is streamed over a Tauri `Channel<InstallStatus>` so the UI can
//! render a download bar.

use futures_util::StreamExt;
use serde::Serialize;
use std::path::{Path, PathBuf};
use tauri::ipc::Channel;
use tokio::io::AsyncWriteExt;

const DOWNLOAD_URL: &str = "https://ollama.com/download/Ollama-darwin.zip";
#[cfg(target_os = "windows")]
const WINDOWS_DOWNLOAD_URL: &str = "https://ollama.com/download/OllamaSetup.exe";
const OLLAMA_PULL_URL: &str = "http://localhost:11434/api/pull";
const ALLOWED_PULL_MODELS: &[&str] = &["llama3.2-vision"];

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum InstallStatus {
    Downloading { percent: u8 },
    Extracting,
    Installing,
    Launching,
    Done,
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum PullStatus {
    Connecting,
    Pulling { percent: u8, phase: String },
    Verifying,
    Done,
}

#[derive(Debug, thiserror::Error)]
pub enum InstallError {
    #[allow(dead_code)]
    #[error("automatic install isn't supported on this platform yet")]
    Unsupported,
    #[error("io: {0}")]
    Io(String),
    #[error("download: {0}")]
    Download(String),
    #[error("install: {0}")]
    Other(String),
}

impl serde::Serialize for InstallError {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

impl From<std::io::Error> for InstallError {
    fn from(e: std::io::Error) -> Self {
        InstallError::Io(e.to_string())
    }
}

struct TempInstallPaths {
    zip_path: PathBuf,
    extract_dir: PathBuf,
}

impl TempInstallPaths {
    fn new() -> Self {
        let id = uuid::Uuid::new_v4();
        let base = std::env::temp_dir();
        Self {
            zip_path: base.join(format!("screenie-ollama-install-{id}.zip")),
            extract_dir: base.join(format!("screenie-ollama-extract-{id}")),
        }
    }
}

impl Drop for TempInstallPaths {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.zip_path);
        let _ = std::fs::remove_dir_all(&self.extract_dir);
    }
}

#[cfg(target_os = "macos")]
pub async fn install(on_progress: Channel<InstallStatus>) -> Result<(), InstallError> {
    use tokio::process::Command;

    // 1. Already installed? Just launch it. Lets the user use this button as
    //    a "make sure Ollama is running" shortcut too.
    let system_app = Path::new("/Applications/Ollama.app");
    let user_app = home_apps_path().map(|p| p.join("Ollama.app"));
    let already = system_app.exists()
        || user_app
            .as_ref()
            .map(|p| p.exists())
            .unwrap_or(false);
    if already {
        let _ = on_progress.send(InstallStatus::Launching);
        let _ = Command::new("/usr/bin/open").args(["-a", "Ollama"]).spawn();
        let _ = on_progress.send(InstallStatus::Done);
        return Ok(());
    }

    // 2. Download.
    let _ = on_progress.send(InstallStatus::Downloading { percent: 0 });
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .read_timeout(std::time::Duration::from_secs(60))
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| InstallError::Download(e.to_string()))?;
    let response = client
        .get(DOWNLOAD_URL)
        .send()
        .await
        .map_err(|e| InstallError::Download(e.to_string()))?;
    if !response.status().is_success() {
        return Err(InstallError::Download(format!(
            "download failed: HTTP {}",
            response.status()
        )));
    }
    let total = response.content_length().unwrap_or(0);

    let temps = TempInstallPaths::new();
    let zip_path = &temps.zip_path;
    let mut file = tokio::fs::File::create(&zip_path).await?;
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last_pct: i32 = -1;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| InstallError::Download(e.to_string()))?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        let pct = if total > 0 {
            ((downloaded * 100) / total).min(100) as i32
        } else {
            0
        };
        // Throttle UI updates: only emit on integer-percent changes.
        if pct > last_pct {
            last_pct = pct;
            let _ = on_progress.send(InstallStatus::Downloading {
                percent: pct as u8,
            });
        }
    }
    file.flush().await?;
    drop(file);

    // 3. Unzip.
    let _ = on_progress.send(InstallStatus::Extracting);
    let extract_dir = &temps.extract_dir;
    tokio::fs::create_dir_all(&extract_dir).await?;
    let status = Command::new("/usr/bin/unzip")
        .args([
            "-q",
            "-o",
            zip_path.to_str().unwrap_or(""),
            "-d",
            extract_dir.to_str().unwrap_or(""),
        ])
        .status()
        .await?;
    if !status.success() {
        return Err(InstallError::Other(format!(
            "unzip exited with {:?}",
            status.code()
        )));
    }

    // 4. Move Ollama.app into an Applications directory we can write to.
    let _ = on_progress.send(InstallStatus::Installing);
    let src = extract_dir.join("Ollama.app");
    if !src.exists() {
        return Err(InstallError::Other(
            "Ollama.app not found in download".into(),
        ));
    }
    verify_macos_app_signature(&src).await?;

    let dest_parent: PathBuf = if writable_dir(Path::new("/Applications")) {
        PathBuf::from("/Applications")
    } else if let Some(user_apps) = home_apps_path() {
        tokio::fs::create_dir_all(&user_apps).await?;
        user_apps
    } else {
        return Err(InstallError::Other(
            "no writable Applications directory".into(),
        ));
    };
    let dest = dest_parent.join("Ollama.app");
    if dest.exists() {
        let _ = tokio::fs::remove_dir_all(&dest).await;
    }
    let cp_status = Command::new("/bin/cp")
        .args([
            "-R",
            src.to_str().unwrap_or(""),
            dest_parent.to_str().unwrap_or(""),
        ])
        .status()
        .await?;
    if !cp_status.success() {
        return Err(InstallError::Other(format!(
            "cp exited with {:?}",
            cp_status.code()
        )));
    }

    // 5. Launch and wait briefly for the daemon. macOS will pop Ollama's own
    //    admin prompt for helper installation; if the user approves it the
    //    daemon comes up within a few seconds and we're done. If they dismiss
    //    it the wait times out — surface that as an error so the UI can show
    //    a recovery hint instead of misleading the user with a green "Done".
    let _ = on_progress.send(InstallStatus::Launching);
    let _ = Command::new("/usr/bin/open").args(["-a", "Ollama"]).spawn();
    let daemon_up = wait_for_ollama(std::time::Duration::from_secs(20)).await;

    if !daemon_up {
        return Err(InstallError::Other(
            "Ollama installed, but the daemon didn't start. Open Ollama.app \
             from Applications and approve the macOS admin prompt — that \
             prompt installs the helper / daemon. Then try the next step \
             again."
                .into(),
        ));
    }

    let _ = on_progress.send(InstallStatus::Done);
    Ok(())
}

#[cfg(target_os = "windows")]
pub async fn install(on_progress: Channel<InstallStatus>) -> Result<(), InstallError> {
    // 1. Already installed? Just launch it. Same fast path the macOS branch
    //    uses — lets the user retry this button as a "make sure Ollama is
    //    running" shortcut.
    if is_installed_on_disk() {
        let _ = on_progress.send(InstallStatus::Launching);
        try_launch_ollama();
        let daemon_up = wait_for_ollama(std::time::Duration::from_secs(30)).await;
        if !daemon_up {
            return Err(InstallError::Other(
                "Ollama is installed, but the daemon didn't start within 30 seconds. \
                 Check the Ollama tray icon in your system tray, then try the next \
                 step again."
                    .into(),
            ));
        }
        let _ = on_progress.send(InstallStatus::Done);
        return Ok(());
    }

    // 2. Download OllamaSetup.exe.
    let _ = on_progress.send(InstallStatus::Downloading { percent: 0 });
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .read_timeout(std::time::Duration::from_secs(60))
        .timeout(std::time::Duration::from_secs(900))
        .build()
        .map_err(|e| InstallError::Download(e.to_string()))?;
    let response = client
        .get(WINDOWS_DOWNLOAD_URL)
        .send()
        .await
        .map_err(|e| InstallError::Download(e.to_string()))?;
    if !response.status().is_success() {
        return Err(InstallError::Download(format!(
            "download failed: HTTP {}",
            response.status()
        )));
    }
    let total = response.content_length().unwrap_or(0);

    let temps = TempInstallPaths::new();
    // Reuse the .zip_path slot for the installer .exe path — its `Drop`
    // cleans up the file either way.
    let installer_path = temps.zip_path.with_extension("exe");
    let mut file = tokio::fs::File::create(&installer_path).await?;
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last_pct: i32 = -1;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| InstallError::Download(e.to_string()))?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        let pct = if total > 0 {
            ((downloaded * 100) / total).min(100) as i32
        } else {
            0
        };
        if pct > last_pct {
            last_pct = pct;
            let _ = on_progress.send(InstallStatus::Downloading {
                percent: pct as u8,
            });
        }
    }
    file.flush().await?;
    drop(file);

    // 3. Launch the installer interactively. Windows will pop a UAC prompt
    //    via ShellExecute → the user accepts it and walks through the
    //    standard Ollama installer. We can't trivially silently install
    //    OllamaSetup.exe — even with `/SILENT` it still requires admin
    //    elevation, and showing the prompt under our own progress bar
    //    risks the user not noticing the elevation dialog at all.
    let _ = on_progress.send(InstallStatus::Installing);
    let installer_path_for_spawn = installer_path.clone();
    let launched = tokio::task::spawn_blocking(move || -> bool {
        windows_shell_execute_installer(&installer_path_for_spawn)
    })
    .await
    .map_err(|e| InstallError::Other(format!("spawn_blocking: {e}")))?;

    if !launched {
        return Err(InstallError::Other(
            "Couldn't launch OllamaSetup.exe — the download finished but \
             ShellExecute returned an error (commonly: user dismissed the \
             UAC prompt). Try downloading the installer manually from \
             ollama.com."
                .into(),
        ));
    }

    // 4. Poll for the install path appearing. The user might take a couple
    //    of minutes to walk through the installer; budget 10 minutes total
    //    before giving up. The Ollama installer drops `ollama.exe` into
    //    %LOCALAPPDATA%\Programs\Ollama\ as one of the first steps, well
    //    before it finishes — but we wait until the daemon is reachable
    //    (next step) before declaring success.
    let install_deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
    let mut installed = false;
    while std::time::Instant::now() < install_deadline {
        if is_installed_on_disk() {
            installed = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
    }
    if !installed {
        return Err(InstallError::Other(
            "Didn't see Ollama appear at the expected install path. If you \
             cancelled or are still walking through the installer, finish \
             that first, then try the next step again."
                .into(),
        ));
    }

    // 5. Make sure the daemon is up. The Ollama installer registers a Windows
    //    service + tray app that should start automatically — but if it
    //    didn't, give it a nudge via `try_launch_ollama`.
    let _ = on_progress.send(InstallStatus::Launching);
    try_launch_ollama();
    let daemon_up = wait_for_ollama(std::time::Duration::from_secs(30)).await;
    if !daemon_up {
        return Err(InstallError::Other(
            "Ollama installed, but the daemon didn't start within 30 seconds. \
             Open the Ollama tray icon (or launch Ollama from the Start menu) \
             then try the next step again."
                .into(),
        ));
    }

    let _ = on_progress.send(InstallStatus::Done);
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub async fn install(_on_progress: Channel<InstallStatus>) -> Result<(), InstallError> {
    Err(InstallError::Unsupported)
}

/// Invoke ShellExecuteW on the freshly-downloaded `OllamaSetup.exe`. The
/// installer requires admin rights, so the OS surfaces a UAC prompt. Returns
/// true on a successful launch (regardless of whether the user accepts the
/// elevation — that's a downstream poll on disk).
#[cfg(target_os = "windows")]
fn windows_shell_execute_installer(path: &Path) -> bool {
    use windows::core::{w, HSTRING, PCWSTR};
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::Shell::ShellExecuteW;
    use windows::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

    let path_h = HSTRING::from(path.as_os_str());
    let path_w = PCWSTR::from_raw(path_h.as_ptr());
    let inst = unsafe {
        ShellExecuteW(
            // HWND in windows-rs 0.58 is `pub struct HWND(*mut c_void)` —
            // a literal 0 won't coerce. Use a null pointer for "no owner."
            HWND(core::ptr::null_mut()),
            // "runas" forces an elevated launch. Plain "open" would
            // sometimes attach the installer to our process tree without
            // the elevation prompt, leaving the installer stuck at
            // "Access denied" when it tries to write to Program Files.
            w!("runas"),
            path_w,
            PCWSTR::null(),
            PCWSTR::null(),
            SW_SHOWNORMAL,
        )
    };
    // ShellExecuteW returns an HINSTANCE that's actually an error code when
    // <= 32 (legacy MS-DOS convention). > 32 = success.
    inst.0 as isize > 32
}

#[cfg(target_os = "macos")]
fn home_apps_path() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(format!("{}/Applications", h)))
}

#[cfg(target_os = "macos")]
async fn verify_macos_app_signature(app: &Path) -> Result<(), InstallError> {
    use tokio::process::Command;

    let codesign = Command::new("/usr/bin/codesign")
        .args(["--verify", "--deep", "--strict", "--verbose=2"])
        .arg(app)
        .output()
        .await?;
    if !codesign.status.success() {
        return Err(InstallError::Other(format!(
            "Ollama.app signature verification failed: {}",
            String::from_utf8_lossy(&codesign.stderr).trim()
        )));
    }

    let spctl = Command::new("/usr/sbin/spctl")
        .args(["--assess", "--type", "execute", "--verbose"])
        .arg(app)
        .output()
        .await?;
    if !spctl.status.success() {
        return Err(InstallError::Other(format!(
            "Ollama.app Gatekeeper assessment failed: {}",
            String::from_utf8_lossy(&spctl.stderr).trim()
        )));
    }

    Ok(())
}

/// Disk-only "is Ollama.app present anywhere we know about?" check. The
/// onboarding flow uses this to distinguish "user hasn't installed Ollama
/// yet" from "installed but daemon isn't running" — `check_ollama` only
/// answers the second question.
#[cfg(target_os = "macos")]
pub fn is_installed_on_disk() -> bool {
    if Path::new("/Applications/Ollama.app").exists() {
        return true;
    }
    if let Some(p) = home_apps_path() {
        if p.join("Ollama.app").exists() {
            return true;
        }
    }
    false
}

#[cfg(target_os = "windows")]
pub fn is_installed_on_disk() -> bool {
    windows_ollama_exe_path().is_some()
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub fn is_installed_on_disk() -> bool {
    false
}

/// Look up the default Ollama install location on Windows. The official
/// installer writes to `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`; if the
/// user did a system-wide install instead, fall back to
/// `%PROGRAMFILES%\Ollama\ollama.exe`.
#[cfg(target_os = "windows")]
fn windows_ollama_exe_path() -> Option<PathBuf> {
    for env_key in ["LOCALAPPDATA", "ProgramFiles", "ProgramW6432"] {
        if let Ok(base) = std::env::var(env_key) {
            let candidate = if env_key == "LOCALAPPDATA" {
                PathBuf::from(&base)
                    .join("Programs")
                    .join("Ollama")
                    .join("ollama.exe")
            } else {
                PathBuf::from(&base).join("Ollama").join("ollama.exe")
            };
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn writable_dir(p: &Path) -> bool {
    if !p.exists() {
        return false;
    }
    let probe = p.join(format!(".screenie-write-test-{}", uuid::Uuid::new_v4()));
    match std::fs::File::create(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// Best-effort: ping the Ollama daemon to see if it answers.
async fn is_ollama_reachable() -> bool {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(800))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    client
        .get("http://localhost:11434/api/tags")
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// Poll for the daemon up to `timeout`, returning whether it ever came up.
async fn wait_for_ollama(timeout: std::time::Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if is_ollama_reachable().await {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    false
}

/// Wake Ollama.app via `open -a` (macOS only). No-op elsewhere.
#[cfg(target_os = "macos")]
fn try_launch_ollama() {
    let _ = std::process::Command::new("/usr/bin/open")
        .args(["-a", "Ollama"])
        .spawn();
}

/// Launch the Ollama GUI helper (which spawns the daemon + tray icon).
/// Falls back to `ollama.exe serve` if the GUI binary is missing.
#[cfg(target_os = "windows")]
fn try_launch_ollama() {
    let local = match std::env::var("LOCALAPPDATA") {
        Ok(v) => v,
        Err(_) => return,
    };
    let dir = std::path::PathBuf::from(local).join("Programs").join("Ollama");
    // The Windows installer ships `ollama app.exe` as the system-tray
    // helper which keeps the daemon running. If that's missing, fall back
    // to `ollama.exe serve` which runs the daemon in the foreground.
    let app_path = dir.join("ollama app.exe");
    let cli_path = dir.join("ollama.exe");
    if app_path.exists() {
        let _ = std::process::Command::new(&app_path).spawn();
    } else if cli_path.exists() {
        let _ = std::process::Command::new(&cli_path)
            .arg("serve")
            .spawn();
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn try_launch_ollama() {}

/// Pull a model into Ollama, forwarding progress over the given channel.
/// Talks to the Ollama daemon's `/api/pull` endpoint, which streams NDJSON
/// status lines including byte-level totals so we can render a progress bar.
///
/// If the daemon isn't running we proactively try to launch Ollama.app and
/// wait briefly for the daemon to come up, since the most common failure
/// after auto-install is that the user dismissed the macOS admin prompt and
/// the helper / daemon never finished setting up.
pub async fn pull_model(
    model: String,
    on_progress: Channel<PullStatus>,
) -> Result<(), InstallError> {
    if !ALLOWED_PULL_MODELS.contains(&model.as_str()) {
        return Err(InstallError::Other("model pull is not allowed".into()));
    }
    let _ = on_progress.send(PullStatus::Connecting);

    if !is_ollama_reachable().await {
        try_launch_ollama();
        if !wait_for_ollama(std::time::Duration::from_secs(15)).await {
            return Err(InstallError::Other(
                "Couldn't reach the Ollama daemon at localhost:11434. Open \
                 Ollama.app from /Applications and approve the macOS admin \
                 prompt (it sets up the local helper). Then try pulling again."
                    .into(),
            ));
        }
    }

    let client = reqwest::Client::builder()
        // Pull can take many minutes on slow links; avoid a total timeout.
        .connect_timeout(std::time::Duration::from_secs(3))
        .build()
        .map_err(|e| InstallError::Other(e.to_string()))?;

    let body = serde_json::json!({ "model": model });
    let resp = client
        .post(OLLAMA_PULL_URL)
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            InstallError::Download(format!("ollama not reachable at localhost:11434 ({})", e))
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(InstallError::Other(format!(
            "pull failed: HTTP {} — {}",
            status, text
        )));
    }

    let mut stream = resp.bytes_stream();
    // Buffer raw bytes — Ollama's pull lines stay ASCII in practice, but the
    // robust fix from the chat-stream code paths costs nothing extra here.
    let mut buf: Vec<u8> = Vec::new();
    let mut last_pct: i32 = -1;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| InstallError::Download(e.to_string()))?;
        buf.extend_from_slice(&chunk);

        while let Some(end) = buf.iter().position(|&b| b == b'\n') {
            let line_bytes = buf.drain(..end + 1).collect::<Vec<u8>>();
            let line_bytes = &line_bytes[..end];
            let line = std::str::from_utf8(line_bytes)
                .map_err(|e| InstallError::Other(e.to_string()))?
                .trim();
            if line.is_empty() {
                continue;
            }

            let v: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // The Ollama daemon may return errors like {"error":"..."}.
            if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
                return Err(InstallError::Other(err.to_string()));
            }

            let status = v
                .get("status")
                .and_then(|s| s.as_str())
                .unwrap_or("");

            if status == "success" {
                let _ = on_progress.send(PullStatus::Done);
                return Ok(());
            }
            if status.starts_with("verifying") {
                let _ = on_progress.send(PullStatus::Verifying);
                continue;
            }

            // Any line that has both `total` and `completed` is a download
            // progress update; emit a percentage.
            let total = v.get("total").and_then(|t| t.as_u64()).unwrap_or(0);
            let completed = v.get("completed").and_then(|c| c.as_u64()).unwrap_or(0);
            if total > 0 {
                let pct = ((completed * 100) / total).min(100) as i32;
                if pct != last_pct {
                    last_pct = pct;
                    let _ = on_progress.send(PullStatus::Pulling {
                        percent: pct as u8,
                        phase: status.to_string(),
                    });
                }
            }
        }
    }

    let _ = on_progress.send(PullStatus::Done);
    Ok(())
}
