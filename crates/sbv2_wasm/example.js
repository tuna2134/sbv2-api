import { ModelHolder } from "./dist/index.js";
import fs from "node:fs/promises";

ModelHolder.globalInit(await fs.readFile("./dist/sbv2_wasm_bg.wasm"));
const holder = await ModelHolder.create(
	(await fs.readFile("../../models/tokenizer.json")).toString("utf-8"),
	await fs.readFile("../../models/deberta.onnx"),
);
await holder.load(
	"tsukuyomi",
	await fs.readFile("../../models/tsukuyomi.sbv2"),
);
await fs.writeFile("out.wav", await holder.synthesize("tsukuyomi", "おはよう"));
holder.unload("tsukuyomi");
