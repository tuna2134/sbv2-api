#!/bin/sh
wasm-pack build --target web ./crates/sbv2_wasm --release
wasm-opt -O3 -o ./crates/sbv2_wasm/pkg/sbv2_wasm_bg.wasm ./crates/sbv2_wasm/pkg/sbv2_wasm_bg.wasm
wasm-strip ./crates/sbv2_wasm/pkg/sbv2_wasm_bg.wasm
mkdir -p ./crates/sbv2_wasm/dist
cp ./crates/sbv2_wasm/pkg/sbv2_wasm_bg.wasm ./crates/sbv2_wasm/dist/sbv2_wasm_bg.wasm
cd ./crates/sbv2_wasm
pnpm build