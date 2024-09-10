use sbv2_core::{bert, error, jtalk, nlp, norm};

fn main() -> error::Result<()> {
    let text = "こんにちは,世界!";

    let normalized_text = norm::normalize_text(text);
    println!("{}", normalized_text);

    let jtalk = jtalk::JTalk::new()?;
    let (phones, tones, word2ph) = jtalk.g2p(&normalized_text)?;
    let (phones, tones, lang_ids) = nlp::cleaned_text_to_sequence(phones, tones);

    let tokenizer = jtalk::get_tokenizer()?;
    println!("{:?}", tokenizer);

    let (token_ids, attention_masks) = jtalk::tokenize(&normalized_text, &tokenizer)?;
    println!("{:?}", token_ids);

    let session = bert::load_model()?;

    bert::predict(&session, token_ids, attention_masks)?;

    Ok(())
}
