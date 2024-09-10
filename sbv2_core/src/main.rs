use sbv2_core::{bert, error, jtalk};

fn main() -> error::Result<()> {
    let text = "こんにちは,世界!";

    let normalized_text = jtalk::normalize_text(text);
    println!("{}", normalized_text);

    let jtalk = jtalk::JTalk::new()?;
    let (phones, tones, _) = jtalk.g2p(&normalized_text)?;
    println!("{:?}", tones);

    let tokenizer = jtalk::get_tokenizer()?;
    println!("{:?}", tokenizer);

    let (token_ids, attention_masks) = jtalk::tokenize(&normalized_text, &tokenizer)?;
    println!("{:?}", token_ids);

    let session = bert::load_model()?;

    bert::predict(&session, token_ids, attention_masks)?;

    Ok(())
}
