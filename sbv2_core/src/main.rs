use std::{fs, time::Instant};

use sbv2_core::tts;
use std::env;

fn main() -> anyhow::Result<()> {
    dotenvy::dotenv_override().ok();
    env_logger::init();
    let text = "眠たい";
    let ident = "aaa";
    let mut tts_holder = tts::TTSModelHolder::new(
        &fs::read(env::var("BERT_MODEL_PATH")?)?,
        &fs::read(env::var("TOKENIZER_PATH")?)?,
    )?;
    tts_holder.load_sbv2file(ident, fs::read(env::var("MODEL_PATH")?)?)?;

    let (bert_ori, phones, tones, lang_ids) = tts_holder.parse_text(text)?;

    let style_vector = tts_holder.get_style_vector(ident, 0, 1.0)?;
    let data = tts_holder.synthesize(
        ident,
        bert_ori.to_owned(),
        phones.clone(),
        tones.clone(),
        lang_ids.clone(),
        style_vector.clone(),
        0.0,
        0.5,
    )?;
    std::fs::write("output.wav", data)?;
    let now = Instant::now();
    for _ in 0..10 {
        tts_holder.parse_text(text)?;
    }
    println!(
        "Time taken(parse_text): {}ms/it",
        now.elapsed().as_millis() / 10
    );
    let now = Instant::now();
    for _ in 0..10 {
        tts_holder.synthesize(
            ident,
            bert_ori.to_owned(),
            phones.clone(),
            tones.clone(),
            lang_ids.clone(),
            style_vector.clone(),
            0.0,
            1.0,
        )?;
    }
    println!(
        "Time taken(synthesize): {}ms/it",
        now.elapsed().as_millis() / 10
    );
    Ok(())
}
