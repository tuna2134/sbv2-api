use crate::error::{Error, Result};
use crate::norm::{replace_punctuation, PUNCTUATIONS};
use jpreprocess::*;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;
use std::sync::Arc;
use tokenizers::Tokenizer;

type JPreprocessType = JPreprocess<DefaultFetcher>;

fn get_jtalk() -> Result<JPreprocessType> {
    let config = JPreprocessConfig {
        dictionary: SystemDictionaryConfig::Bundled(kind::JPreprocessDictionaryKind::NaistJdic),
        user_dictionary: None,
    };
    let jpreprocess = JPreprocess::from_config(config)?;
    Ok(jpreprocess)
}

static JTALK_G2P_G_A1_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"/A:([0-9\-]+)\+").unwrap());
static JTALK_G2P_G_A2_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\+(\d+)\+").unwrap());
static JTALK_G2P_G_A3_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\+(\d+)/").unwrap());
static JTALK_G2P_G_E3_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"!(\d+)_").unwrap());
static JTALK_G2P_G_F1_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"/F:(\d+)_").unwrap());
static JTALK_G2P_G_P3_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\-(.*?)\+").unwrap());

fn numeric_feature_by_regex(regex: &Regex, text: &str) -> i32 {
    if let Some(mat) = regex.captures(text) {
        mat[1].parse::<i32>().unwrap()
    } else {
        -50
    }
}

macro_rules! hash_set {
    ($($elem:expr),* $(,)?) => {{
        let mut set = HashSet::new();
        $(
            set.insert($elem);
        )*
        set
    }};
}

pub struct JTalk {
    pub jpreprocess: Arc<JPreprocessType>,
}

impl JTalk {
    pub fn new() -> Result<Self> {
        let jpreprocess = Arc::new(get_jtalk()?);
        Ok(Self { jpreprocess })
    }

    pub fn g2p(&self, text: &str) -> Result<()> {
        let parsed = self.jpreprocess.run_frontend(text)?;
        let jtalk_process = JTalkProcess::new(Arc::clone(&self.jpreprocess), parsed);
        jtalk_process.g2p()?;
        Ok(())
    }
}

struct JTalkProcess {
    jpreprocess: Arc<JPreprocessType>,
    parsed: Vec<String>,
}

impl JTalkProcess {
    fn new(jpreprocess: Arc<JPreprocessType>, parsed: Vec<String>) -> Self {
        Self {
            jpreprocess,
            parsed,
        }
    }

    fn fix_phone_tone(&self, phone_tone_list: Vec<(String, i32)>) -> Result<Vec<(String, i32)>> {
        let tone_values: HashSet<i32> = phone_tone_list
            .iter()
            .map(|(_letter, tone)| *tone)
            .collect();
        if tone_values.len() == 1 {
            assert!(tone_values == hash_set![0], "{:?}", tone_values);
            Ok(phone_tone_list)
        } else if tone_values.len() == 2 {
            if tone_values == hash_set![0, 1] {
                return Ok(phone_tone_list);
            } else if tone_values == hash_set![-1, 0] {
                return Ok(phone_tone_list
                    .iter()
                    .map(|x| {
                        let new_tone = if x.1 == -1 { 0 } else { x.1 };
                        (x.0.clone(), new_tone)
                    })
                    .collect());
            } else {
                return Err(Error::ValueError("Invalid tone values 0".to_string()));
            }
        } else {
            return Err(Error::ValueError("Invalid tone values 1".to_string()));
        }
    }

    pub fn g2p(&self) -> Result<()> {
        let phone_tone_list_wo_punct = self.g2phone_tone_wo_punct()?;
        let (seq_text, seq_kata) = self.text_to_seq_kata()?;
        Ok(())
    }

    fn text_to_seq_kata(&self) -> Result<(Vec<String>, Vec<String>)> {
        let mut seq_kata = vec![];
        let mut seq_text = vec![];

        for parts in &self.parsed {
            let (string, pron) = self.parse_to_string_and_pron(parts.clone());
            let mut yomi = pron.replace('’', "");
            let word = replace_punctuation(string);
            assert!(!yomi.is_empty(), "Empty yomi: {}", word);
            if yomi == "、" {
                if !word
                    .chars()
                    .all(|x| PUNCTUATIONS.contains(&x.to_string().as_str()))
                {
                    yomi = "'".repeat(word.len());
                } else {
                    yomi = word.clone();
                }
            } else if yomi == "？" {
                assert!(word == "?", "yomi `？` comes from: {}", word);
                yomi = "?".to_string();
            }
            seq_text.push(word);
            seq_kata.push(yomi);
        }
        Ok((seq_text, seq_kata))
    }

    fn parse_to_string_and_pron(&self, parts: String) -> (String, String) {
        let part_lists: Vec<String> = parts.split(',').map(|x| x.to_string()).collect();
        (part_lists[0].clone(), part_lists[9].clone())
    }

    fn g2phone_tone_wo_punct(&self) -> Result<Vec<(String, i32)>> {
        let prosodies = self.g2p_prosody()?;

        let mut results: Vec<(String, i32)> = Vec::new();
        let mut current_phrase: Vec<(String, i32)> = Vec::new();
        let mut current_tone = 0;

        for (i, letter) in prosodies.iter().enumerate() {
            if letter == "^" {
                assert!(i == 0);
            } else if ["$", "?", "_", "#"].contains(&letter.as_str()) {
                results.extend(self.fix_phone_tone(current_phrase.clone())?);
                if ["$", "?"].contains(&letter.as_str()) {
                    assert!(i == prosodies.len() - 1);
                }
                current_phrase = Vec::new();
                current_tone = 0;
            } else if letter == "[" {
                current_tone += 1;
            } else if letter == "]" {
                current_tone -= 1;
            } else {
                let new_letter = if letter == "cl" {
                    "q".to_string()
                } else {
                    letter.clone()
                };
                current_phrase.push((new_letter, current_tone));
            }
        }

        Ok(results)
    }

    fn g2p_prosody(&self) -> Result<Vec<String>> {
        let labels = self.jpreprocess.make_label(self.parsed.clone());

        let mut phones: Vec<String> = Vec::new();
        for (i, label) in labels.iter().enumerate() {
            let mut p3 = {
                let label_text = label.to_string();
                let mattched = JTALK_G2P_G_P3_PATTERN.captures(&label_text).unwrap();
                mattched[1].to_string()
            };
            if "AIUEO".contains(&p3) {
                // 文字をlowerする
                p3 = p3.to_lowercase();
            }
            if p3 == "sil" {
                assert!(i == 0 || i == labels.len() - 1);
                if i == 0 {
                    phones.push("^".to_string());
                } else if i == labels.len() - 1 {
                    let e3 = numeric_feature_by_regex(&JTALK_G2P_G_E3_PATTERN, &label.to_string());
                    if e3 == 0 {
                        phones.push("$".to_string());
                    } else if e3 == 1 {
                        phones.push("?".to_string());
                    }
                }
                continue;
            } else if p3 == "pau" {
                phones.push("_".to_string());
                continue;
            } else {
                phones.push(p3.clone());
            }

            let a1 = numeric_feature_by_regex(&JTALK_G2P_G_A1_PATTERN, &label.to_string());
            let a2 = numeric_feature_by_regex(&JTALK_G2P_G_A2_PATTERN, &label.to_string());
            let a3 = numeric_feature_by_regex(&JTALK_G2P_G_A3_PATTERN, &label.to_string());

            let f1 = numeric_feature_by_regex(&JTALK_G2P_G_F1_PATTERN, &label.to_string());

            let a2_next =
                numeric_feature_by_regex(&JTALK_G2P_G_A2_PATTERN, &labels[i + 1].to_string());

            if a3 == 1 && a2_next == 1 && "aeiouAEIOUNcl".contains(&p3) {
                phones.push("#".to_string());
            } else if a1 == 0 && a2_next == a2 + 1 && a2 != f1 {
                phones.push("]".to_string());
            } else if a2 == 1 && a2_next == 2 {
                phones.push("[".to_string());
            }
        }

        Ok(phones)
    }
}

pub fn normalize_text(text: &str) -> String {
    // 日本語のテキストを正規化する
    let text = text.replace('~', "ー");
    let text = text.replace('～', "ー");

    text.replace('〜', "ー")
}

pub fn get_tokenizer() -> Result<Tokenizer> {
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;
    Ok(tokenizer)
}

pub fn tokenize(text: &str, tokenizer: &Tokenizer) -> Result<(Vec<i64>, Vec<i64>)> {
    let mut token_ids = vec![1];
    let mut attention_masks = vec![1];
    for content in text.chars() {
        let token = tokenizer.encode(content.to_string(), false)?;
        let ids = token.get_ids();
        token_ids.extend(ids.iter().map(|&x| x as i64));
        attention_masks.extend(token.get_attention_mask().iter().map(|&x| x as i64));
    }
    token_ids.push(2);
    attention_masks.push(1);
    Ok((token_ids, attention_masks))
}
