import * as wasm from "../pkg/sbv2_wasm.js";
import { InferenceSession, Tensor } from "onnxruntime-web";

export class ModelHolder {
	private models: Map<string, [InferenceSession, wasm.StyleVectorWrap]> =
		new Map();
	constructor(
		private tok: wasm.TokenizerWrap,
		private deberta: InferenceSession,
	) {}
	public static async globalInit(buf: ArrayBufferLike) {
		await wasm.default(buf);
	}
	public static async create(tok: string, deberta: ArrayBufferLike) {
		return new ModelHolder(
			wasm.load_tokenizer(tok),
			await InferenceSession.create(deberta, {
				executionProviders: ["webnn", "webgpu", "wasm", "cpu"],
				graphOptimizationLevel: "all",
			}),
		);
	}
	public async synthesize(
		name: string,
		text: string,
		style_id: number = 0,
		style_weight: number = 1.0,
		sdp_ratio: number = 0.4,
		speed: number = 1.0,
	) {
		const mod = this.models.get(name);
		if (!mod) throw new Error(`No model named ${name}`);
		const [vits2, style] = mod;
		return wasm.synthesize(
			text,
			this.tok,
			async (a: BigInt64Array, b: BigInt64Array) => {
				try {
					const res = (
						await this.deberta.run({
							input_ids: new Tensor("int64", a, [1, a.length]),
							attention_mask: new Tensor("int64", b, [1, b.length]),
						})
					)["output"];
					return [new Uint32Array(res.dims), await res.getData(true)];
				} catch (e) {
					console.warn(e);
					throw e;
				}
			},
			async (
				[a_shape, a_array]: any,
				b_d: any,
				c_d: any,
				d_d: any,
				e_d: any,
				f: number,
				g: number,
			) => {
				try {
					const a = new Tensor("float32", a_array, [1, ...a_shape]);
					const b = new Tensor("int64", b_d, [1, b_d.length]);
					const c = new Tensor("int64", c_d, [1, c_d.length]);
					const d = new Tensor("int64", d_d, [1, d_d.length]);
					const e = new Tensor("float32", e_d, [1, e_d.length]);
					const res = (
						await vits2.run({
							x_tst: b,
							x_tst_lengths: new Tensor("int64", [b_d.length]),
							sid: new Tensor("int64", [0]),
							tones: c,
							language: d,
							bert: a,
							style_vec: e,
							sdp_ratio: new Tensor("float32", [f]),
							length_scale: new Tensor("float32", [g]),
							noise_scale: new Tensor("float32", [0.677]),
							noise_scale_w: new Tensor("float32", [0.8]),
						})
					).output;
					return [new Uint32Array(res.dims), await res.getData(true)];
				} catch (e) {
					console.warn(e);
					throw e;
				}
			},
			sdp_ratio,
			1.0 / speed,
			style_id,
			style_weight,
			style,
		);
	}
	public async load(name: string, b: Uint8Array) {
		const [style, vits2_b] = wasm.load_sbv2file(b);
		const vits2 = await InferenceSession.create(vits2_b as Uint8Array, {
			executionProviders: ["webnn", "webgpu", "wasm", "cpu"],
			graphOptimizationLevel: "all",
		});
		this.models.set(name, [vits2, style]);
	}
	public async unload(name: string) {
		return this.models.delete(name);
	}
	public modelList() {
		return this.models.keys();
	}
}
