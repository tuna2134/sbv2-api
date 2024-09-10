use crate::norm::SYMBOLS;
use once_cell::sync::Lazy;
use std::collections::HashMap;

static SYMBOL_TO_ID: Lazy<HashMap<String, i32>> = Lazy::new(|| {
    let mut map = HashMap::new();
    for (i, symbols) in SYMBOLS.iter().enumerate() {
        map.insert(symbols.to_string(), i as i32);
    }
    map
});

pub fn cleaned_text_to_sequence(
    cleaned_phones: Vec<String>,
    tones: Vec<i32>,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let phones: Vec<i32> = cleaned_phones
        .iter()
        .map(|phone| SYMBOL_TO_ID.get(phone).unwrap())
        .collect();
    let tones: Vec<i32> = tones.iter().map(|tone| tone + 6).collect();
    let lang_ids: Vec<i32> = vec![1; phones.len()];
    (phones, tones, lang_ids)
}
