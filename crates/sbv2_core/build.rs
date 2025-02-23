use std::env;
use std::fs;
use std::io::copy;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(&env::var("OUT_DIR").unwrap());
    let out_path = out_dir.join("all.bin");
    if !out_path.exists() {
        println!("cargo:warning=Downloading dictionary file...");
        let mut response =
            ureq::get("https://huggingface.co/neody/sbv2-api-assets/resolve/main/dic/all.bin")
                .call()?;
        let mut response = response.body_mut().as_reader();
        let mut file = fs::File::create(&out_path)?;
        copy(&mut response, &mut file)?;
    }
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
