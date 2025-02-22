fn main() {
    if cfg!(feature = "coreml") {
        println!("cargo:rustc-link-arg=-fapple-link-rtlib");
    }
}
