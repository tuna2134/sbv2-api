use std::env;
use std::fs::{self, hard_link};
use std::io::copy;
use std::path::PathBuf;

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
    hard_link(
        out_path,
        PathBuf::from(&env::var("OUT_DIR").unwrap()).join("out.bin"),
    )
    .unwrap();
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
