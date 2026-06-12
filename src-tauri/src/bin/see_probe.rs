//! Integration probe for the perception layer.
//!
//! `cargo run --bin see_probe -- --app Finder` walks the app's focused
//! window and prints what the index would ingest. Grows the serialized
//! block, marked-PNG output, and perf gates in later phases.

#[cfg(target_os = "macos")]
fn main() {
    use screenieai_lib::perception::ax::{ffi, walker, WalkConfig};
    use std::process::Command;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pid: Option<i32> = None;
    let mut app_name: Option<String> = None;
    let mut budget: usize = 1500;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--pid" => {
                index += 1;
                pid = args.get(index).and_then(|v| v.parse().ok());
            }
            "--app" => {
                index += 1;
                app_name = args.get(index).cloned();
            }
            "--budget" => {
                index += 1;
                budget = args.get(index).and_then(|v| v.parse().ok()).unwrap_or(1500);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
        index += 1;
    }

    if !ffi::process_trusted() {
        eprintln!(
            "see_probe: Accessibility permission missing for this process; grant it to the \
             terminal app running the probe (System Settings → Privacy & Security → Accessibility)"
        );
        std::process::exit(3);
    }

    let pid = pid.or_else(|| {
        let name = app_name.as_deref()?;
        let output = Command::new("pgrep").args(["-ix", name]).output().ok()?;
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()?
            .trim()
            .parse()
            .ok()
    });
    let Some(pid) = pid else {
        eprintln!("usage: see_probe --app <Name> | --pid <pid> [--budget N]");
        std::process::exit(2);
    };

    let config = WalkConfig {
        element_budget: budget,
        ..WalkConfig::default()
    };
    let started = Instant::now();
    let Some(outcome) = walker::walk_app_focused_window(pid, &config) else {
        eprintln!("see_probe: no AX app/window for pid {pid} (app gone or AX returns nothing)");
        std::process::exit(4);
    };
    let total_ms = started.elapsed().as_millis();

    println!(
        "pid={pid} window={:?} window_id={:?} frame=({},{} {}x{})pt n={} partial={} walk={}ms total={}ms",
        outcome.window_title,
        outcome.window_id,
        outcome.window_frame.x,
        outcome.window_frame.y,
        outcome.window_frame.w,
        outcome.window_frame.h,
        outcome.elements.len(),
        outcome.partial,
        outcome.took_ms,
        total_ms,
    );
    let actionable = outcome.elements.iter().filter(|e| e.actionable).count();
    let boundaries = outcome
        .elements
        .iter()
        .filter(|e| e.is_web_boundary)
        .count();
    println!("actionable={actionable} web_boundaries={boundaries}");
    for element in outcome.elements.iter().take(40) {
        println!(
            "  d{} {} {:?} ({},{} {}x{}) actions={:?}{}{}",
            element.depth,
            element.role,
            element
                .title
                .as_deref()
                .or(element.descr.as_deref())
                .unwrap_or(""),
            element.frame.x,
            element.frame.y,
            element.frame.w,
            element.frame.h,
            element.actions,
            if element.focused { " FOCUS" } else { "" },
            if element.is_web_boundary {
                " web⊥"
            } else {
                ""
            },
        );
    }
    if outcome.elements.len() > 40 {
        println!("  … {} more", outcome.elements.len() - 40);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("see_probe is macOS-only");
    std::process::exit(1);
}
