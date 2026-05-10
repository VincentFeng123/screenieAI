fn main() {
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rerun-if-changed=src/macos_window.m");

        cc::Build::new()
            .file("src/macos_window.m")
            .flag("-fblocks")
            .flag("-fobjc-exceptions")
            .compile("screenie_macos_window");
        println!("cargo:rustc-link-lib=framework=AppKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
        // Vision (text recognition) + ImageIO (CGImageSource).
        println!("cargo:rustc-link-lib=framework=Vision");
        println!("cargo:rustc-link-lib=framework=ImageIO");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        println!("cargo:rustc-link-lib=framework=ScreenCaptureKit");
    }

    tauri_build::build()
}
