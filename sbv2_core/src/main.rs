use std::fs;

use sbv2_core::tts;
use std::env;

fn main() -> anyhow::Result<()> {
    dotenvy::dotenv_override().ok();
    env_logger::init();
    let text = fs::read_to_string("content.txt")?;
    let ident = "aaa";
    let mut tts_holder = tts::TTSModelHolder::new(
        &fs::read(env::var("BERT_MODEL_PATH")?)?,
        &fs::read(env::var("TOKENIZER_PATH")?)?,
    )?;
    tts_holder.load_sbv2file(ident, fs::read(env::var("MODEL_PATH")?)?)?;

    let audio = tts_holder.easy_synthesize(ident, &text, 0, tts::SynthesizeOptions::default())?;
    fs::write("output.wav", &audio)?;

    Ok(())
}
