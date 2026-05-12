fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rerun-if-changed=src/macos_window.m");

        cc::Build::new()
            .file("src/macos_window.m")
            .flag("-fblocks")
            .flag("-fobjc-exceptions")
            .compile("screenie_macos_window");
        println!("cargo:rustc-link-lib=framework=AppKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Vision");
        println!("cargo:rustc-link-lib=framework=ImageIO");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        println!("cargo:rustc-link-lib=framework=ScreenCaptureKit");

        // Embed Info.plist into the binary's __TEXT,__info_plist section.
        // `tauri dev` runs the raw binary at target/debug/screenieai (NOT
        // a .app bundle), so without this, macOS sees a "naked" binary
        // with no usage description. TCC then re-prompts every launch and
        // every grant evaporates on the next rebuild because the binary's
        // hash changes. Embedding the plist makes the binary self-describing
        // — TCC reads NSScreenCaptureUsageDescription / LSUIElement directly
        // from the Mach-O section, which matches what bundled .app installs
        // expose, so the dev binary and the bundled release behave the same.
        println!("cargo:rerun-if-changed=Info.plist");
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR is always set by cargo");
        let plist_path = format!("{manifest_dir}/Info.plist");
        if std::path::Path::new(&plist_path).exists() {
            println!("cargo:rustc-link-arg=-Wl,-sectcreate,__TEXT,__info_plist,{plist_path}");
        }
    }

    tauri_build::build()
}
