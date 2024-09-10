use sbv2_core::{bert, error, jtalk, nlp, norm, utils};

fn main() -> error::Result<()> {
    let text = "こんにちは,世界!";

    let normalized_text = norm::normalize_text(text);
    println!("{}", normalized_text);

    let jtalk = jtalk::JTalk::new()?;
    let (phones, tones, mut word2ph) = jtalk.g2p(&normalized_text)?;
    let (phones, tones, lang_ids) = nlp::cleaned_text_to_sequence(phones, tones);

    // add black
    let phones = utils::intersperse(&phones, 0);
    let tones = utils::intersperse(&tones, 0);
    let lang_ids = utils::intersperse(&lang_ids, 0);
    for i in 0..word2ph.len() {
        word2ph[i] *= 2;
    }
    word2ph[0] += 1;

    let tokenizer = jtalk::get_tokenizer()?;
    let (token_ids, attention_masks) = jtalk::tokenize(&normalized_text, &tokenizer)?;

    let session = bert::load_model()?;
    let bert_content = bert::predict(&session, token_ids, attention_masks)?;

    println!("{:?}", word2ph);

    assert!(
        word2ph.len() == normalized_text.chars().count() + 2,
        "{} {}",
        word2ph.len(),
        normalized_text.chars().count()
    );

    Ok(())
}
