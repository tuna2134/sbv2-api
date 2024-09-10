use sbv2_core::{error, tts};

fn main() -> error::Result<()> {
    let text = "眠たい";

    let tts_model = tts::TTSModel::new(
        "models/debert.onnx",
        "models/model_opt.onnx",
        "models/style_vectors.json",
    )?;

    let (bert_ori, phones, tones, lang_ids) = tts_model.parse_text(text)?;

    let style_vector = tts_model.get_style_vector(0, 1.0)?;
    let data = tts_model.synthesize(bert_ori.to_owned(), phones, tones, lang_ids, style_vector)?;

    std::fs::write("output.wav", data)?;

    Ok(())
}
