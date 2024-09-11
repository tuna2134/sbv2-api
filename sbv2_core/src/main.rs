use std::{fs, time::Instant};

use sbv2_core::{error, tts};

fn main() -> error::Result<()> {
    let text = "眠たい";

    let tts_model = tts::TTSModel::new(
        fs::read("models/debert.onnx")?,
        fs::read("models/model_opt.onnx")?,
        fs::read("models/style_vectors.json")?,
        fs::read("models/tokenizer.json")?,
    )?;

    let (bert_ori, phones, tones, lang_ids) = tts_model.parse_text(text)?;

    let style_vector = tts_model.get_style_vector(0, 1.0)?;
    let data = tts_model.synthesize(
        bert_ori.to_owned(),
        phones.clone(),
        tones.clone(),
        lang_ids.clone(),
        style_vector.clone(),
    )?;
    std::fs::write("output.wav", data)?;
    let now = Instant::now();
    for _ in 0..10 {
        tts_model.synthesize(
            bert_ori.to_owned(),
            phones.clone(),
            tones.clone(),
            lang_ids.clone(),
            style_vector.clone(),
        )?;
    }
    println!("Time taken: {}", now.elapsed().as_millis());
    Ok(())
}
