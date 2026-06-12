//! Integration probe for the perception layer: drives the real
//! see → index → read / look pipeline and prints what the agent would get.
//!
//! ```text
//! cargo run --bin see_probe -- --app Finder
//! cargo run --bin see_probe -- --app Safari --look --sql "SELECT class, count(*) FROM elements GROUP BY class"
//! ```
//!
//! Perf gates (Phase 6) print at the end of every run.

#[cfg(target_os = "macos")]
fn main() {
    use screenieai_lib::perception::commands::{look_impl, read_impl, see_impl};
    use screenieai_lib::perception::index::{self, ElementQuery};
    use screenieai_lib::perception::{SeeOptions, SeeScope};
    use std::process::Command;
    use std::time::Instant;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pid: Option<i32> = None;
    let mut app_name: Option<String> = None;
    let mut budget: usize = 1500;
    let mut do_look = false;
    let mut do_hit_test = false;
    let mut do_unset_enhanced = false;
    let mut sql: Option<String> = None;
    let mut index_arg = 0;
    while index_arg < args.len() {
        match args[index_arg].as_str() {
            "--pid" => {
                index_arg += 1;
                pid = args.get(index_arg).and_then(|v| v.parse().ok());
            }
            "--app" => {
                index_arg += 1;
                app_name = args.get(index_arg).cloned();
            }
            "--budget" => {
                index_arg += 1;
                budget = args
                    .get(index_arg)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1500);
            }
            "--look" => do_look = true,
            "--hit-test" => do_hit_test = true,
            // Maintenance: clear a stray AXEnhancedUserInterface=true (the
            // VoiceOver flag) left on an app — it makes WebKit walks ~20x
            // slower while set.
            "--unset-enhanced-ui" => do_unset_enhanced = true,
            "--sql" => {
                index_arg += 1;
                sql = args.get(index_arg).cloned();
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
        index_arg += 1;
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
        eprintln!(
            "usage: see_probe --app <Name> | --pid <pid> [--budget N] [--look] [--sql <select>]"
        );
        std::process::exit(2);
    };

    // Bundle id from the binary path (…/Name.app/Contents/MacOS/bin) so the
    // browser/web-boundary decision matches the in-app frontmost path.
    let bundle = Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "comm="])
        .output()
        .ok()
        .and_then(|output| {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let app_root = path.split(".app/").next().map(|p| format!("{p}.app"))?;
            let plist = format!("{app_root}/Contents/Info");
            let value = Command::new("defaults")
                .args(["read", &plist, "CFBundleIdentifier"])
                .output()
                .ok()?;
            let bundle = String::from_utf8_lossy(&value.stdout).trim().to_string();
            (!bundle.is_empty()).then_some(bundle)
        });
    if let Some(bundle) = &bundle {
        println!("(bundle: {bundle})");
    }

    if do_unset_enhanced {
        use screenieai_lib::perception::ax::ffi;
        match ffi::app_element(pid, 1.0) {
            Some(app) => {
                let cleared = ffi::set_bool_attribute(&app, "AXEnhancedUserInterface", false);
                println!("AXEnhancedUserInterface=false on pid {pid}: {cleared}");
            }
            None => eprintln!("no app element for pid {pid}"),
        }
        return;
    }

    let base = std::env::temp_dir().join("screenie-see-probe");
    let opts = SeeOptions {
        scope: SeeScope::App { pid, bundle },
        element_budget: budget,
        ..SeeOptions::default()
    };

    // --- see ---------------------------------------------------------------
    let started = Instant::now();
    let seen = match see_impl(&base, &opts) {
        Ok(seen) => seen,
        Err(err) => {
            eprintln!("see failed: {err}");
            std::process::exit(3);
        }
    };
    let see_total_ms = started.elapsed().as_millis();
    println!(
        "=== see ({}ms walk, {}ms total incl. index) ===",
        seen.took_ms, see_total_ms
    );
    println!("{}", seen.top_elements);
    println!(
        "(block: {} bytes, partial={}, window_id={:?})",
        seen.top_elements.len(),
        seen.partial,
        seen.window_id
    );

    // --- read (structured, p50 over 20 runs) --------------------------------
    let query = ElementQuery {
        snapshot_id: Some(seen.snapshot_id.clone()),
        actionable_only: true,
        ..ElementQuery::default()
    };
    let mut read_times = Vec::with_capacity(20);
    let mut last_read = None;
    for _ in 0..20 {
        let read_started = Instant::now();
        match read_impl(&base, &query) {
            Ok(result) => {
                read_times.push(read_started.elapsed().as_secs_f64() * 1000.0);
                last_read = Some(result);
            }
            Err(err) => {
                eprintln!("read failed: {err}");
                std::process::exit(4);
            }
        }
    }
    read_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = read_times[read_times.len() / 2];
    if let Some(read) = &last_read {
        println!(
            "\n=== read actionable_only (p50 {:.2}ms over 20 runs) ===",
            p50
        );
        let mut lines = read.text.lines();
        for line in lines.by_ref().take(12) {
            println!("{line}");
        }
        if read.text.lines().count() > 12 {
            println!(
                "…(probe display truncated; {} rows returned of {})",
                read.returned, read.total
            );
        }
    }

    // --- read_sql ------------------------------------------------------------
    if let Some(sql) = sql {
        match index::init(&base).and_then(|idx| idx.reader.read_sql(&sql)) {
            Ok(rows) => {
                println!("\n=== read_sql ===\n{}", rows.columns.join(" | "));
                for row in rows.rows.iter().take(20) {
                    let rendered: Vec<String> = row.iter().map(|v| v.to_string()).collect();
                    println!("{}", rendered.join(" | "));
                }
            }
            Err(err) => println!("\nread_sql failed: {err}"),
        }
    }

    // --- look ----------------------------------------------------------------
    if do_look {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let look_started = Instant::now();
        match runtime.block_on(look_impl(&base, Some(&seen.snapshot_id), None)) {
            Ok(look) => {
                println!(
                    "\n=== look ({}ms) ===\nmarked: {} ({}x{} px, {} marks, {} capped)\nraw:    {}",
                    look_started.elapsed().as_millis(),
                    look.marked_png,
                    look.width,
                    look.height,
                    look.marks_drawn,
                    look.marks_capped,
                    look.raw_png,
                );
                for entry in look.legend.iter().take(10) {
                    println!("  {} → {}", entry.eid, entry.label);
                }
            }
            Err(err) => println!("\nlook failed: {err}"),
        }
    }

    // --- hit-test harness -------------------------------------------------------
    // Resolves sampled actionable elements through the same resolve() path
    // the executor uses, then AX hit-tests each click point and checks the
    // hit element's frame still contains it. Read-only: nothing is pressed
    // (press-vs-click agreement needs supervised manual QA — it mutates the
    // target app).
    if do_hit_test {
        use screenieai_lib::perception::ax::ffi;
        use screenieai_lib::perception::ids;

        let read = read_impl(
            &base,
            &ElementQuery {
                snapshot_id: Some(seen.snapshot_id.clone()),
                actionable_only: true,
                limit: 200,
                ..ElementQuery::default()
            },
        )
        .expect("hit-test read");
        let eids: Vec<String> = read
            .text
            .lines()
            .skip(1)
            .filter(|line| line.starts_with("ax:"))
            .filter_map(|line| line.split_whitespace().next().map(|eid| eid.to_string()))
            .take(20)
            .collect();
        let mut hits = 0usize;
        let mut total = 0usize;
        for eid in &eids {
            let Ok(resolved) = ids::resolve(eid, &seen.snapshot_id, true) else {
                continue;
            };
            total += 1;
            let (cx, cy) = resolved.click_point;
            let contained = ffi::hit_test(cx, cy)
                .and_then(|hit| {
                    let attrs = ffi::copy_attributes(&hit, &["AXPosition", "AXSize"], 0).ok()?;
                    let mut position = None;
                    let mut size = None;
                    for (index, attr) in attrs.into_iter().enumerate() {
                        match (index, attr) {
                            (0, ffi::AttrValue::Point(x, y)) => position = Some((x, y)),
                            (1, ffi::AttrValue::Size(w, h)) => size = Some((w, h)),
                            _ => {}
                        }
                    }
                    let (x, y) = position?;
                    let (w, h) = size?;
                    Some(cx >= x && cx < x + w && cy >= y && cy < y + h)
                })
                .unwrap_or(false);
            if contained {
                hits += 1;
            } else {
                println!("  miss: {eid} center=({cx:.0},{cy:.0})");
            }
        }
        println!(
            "\n=== hit-test ({}/{} centers land on a hit element containing them) ===",
            hits, total
        );
    }

    // --- perf gates -----------------------------------------------------------
    println!("\n=== perf gates ===");
    println!(
        "see walk: {}ms (gate: <400ms System Settings, <1500ms heavy Electron)",
        seen.took_ms
    );
    println!("read p50: {p50:.2}ms (gate: <5ms)");
    println!(
        "top-40 block: {} bytes (gate: ~2KB typical)",
        seen.top_elements.len()
    );
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("see_probe is macOS-only");
    std::process::exit(1);
}
