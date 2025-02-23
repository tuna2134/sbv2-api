use std::fs;
use std::io::copy;

use home_dir::HomeDirExt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = "~/.cache/sbv2-api-assets".expand_home().unwrap();
    let out_path = out_dir.join("all.bin");
    if !out_path.exists() {
        if !out_dir.exists() {
            fs::create_dir_all(out_dir).unwrap();
        }
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
