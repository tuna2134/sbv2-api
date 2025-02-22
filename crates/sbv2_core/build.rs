use std::env;
use std::fs;
use std::io::copy;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR")?;
    let out_path = Path::new(&out_dir).join("all.bin");
    if !out_path.exists() {
        println!("cargo:warning=Downloading dictionary file...");
        let mut response = reqwest::blocking::get(
            "https://huggingface.co/neody/sbv2-api-assets/resolve/main/dic/all.bin",
        )?;
        let mut file = fs::File::create(&out_path)?;
        copy(&mut response, &mut file)?;
    }
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
