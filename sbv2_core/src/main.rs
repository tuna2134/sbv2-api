use sbv2_core::{bert, error, text};

fn main() -> error::Result<()> {
    let text = "こんにちは,世界!";

    let normalized_text = text::normalize_text(text);
    println!("{}", normalized_text);

    let jtalk = text::JTalk::new()?;
    jtalk.g2p(&normalized_text)?;
    println!("{:?}", ());

    let tokenizer = text::get_tokenizer()?;
    println!("{:?}", tokenizer);

    let (token_ids, attention_masks) = text::tokenize(&normalized_text, &tokenizer)?;
    println!("{:?}", token_ids);

    let session = bert::load_model()?;

    bert::predict(&session, token_ids, attention_masks)?;

    Ok(())
}
