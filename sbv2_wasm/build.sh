wasm-pack build --target web sbv2_wasm
wasm-opt -O3 -o sbv2_wasm/pkg/sbv2_wasm_bg.wasm sbv2_wasm/pkg/sbv2_wasm_bg.wasm