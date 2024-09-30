use crate::error::{Error, Result};
use crate::{jtalk, model, style, tokenizer, tts_util};
use ndarray::{concatenate, Array1, Array2, Array3, Axis};
use ort::Session;
use tokenizers::Tokenizer;

#[derive(PartialEq, Eq, Clone)]
pub struct TTSIdent(String);

impl std::fmt::Display for TTSIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)?;
        Ok(())
    }
}

impl<S> From<S> for TTSIdent
where
    S: AsRef<str>,
{
    fn from(value: S) -> Self {
        TTSIdent(value.as_ref().to_string())
    }
}

pub struct TTSModel {
    vits2: Session,
    style_vectors: Array2<f32>,
    ident: TTSIdent,
}

/// High-level Style-Bert-VITS2's API
pub struct TTSModelHolder {
    tokenizer: Tokenizer,
    bert: Session,
    models: Vec<TTSModel>,
    jtalk: jtalk::JTalk,
}

impl TTSModelHolder {
    /// Initialize a new TTSModelHolder
    ///
    /// # Examples
    ///
    /// ```rs
    /// let mut tts_holder = TTSModelHolder::new(std::fs::read("deberta.onnx")?, std::fs::read("tokenizer.json")?)?;
    /// ```
    pub fn new<P: AsRef<[u8]>>(bert_model_bytes: P, tokenizer_bytes: P) -> Result<Self> {
        let bert = model::load_model(bert_model_bytes, true)?;
        let jtalk = jtalk::JTalk::new()?;
        let tokenizer = tokenizer::get_tokenizer(tokenizer_bytes)?;
        Ok(TTSModelHolder {
            bert,
            models: vec![],
            jtalk,
            tokenizer,
        })
    }

    /// Return a list of model names
    pub fn models(&self) -> Vec<String> {
        self.models.iter().map(|m| m.ident.to_string()).collect()
    }

    /// Load a .sbv2 file binary
    ///
    /// # Examples
    ///
    /// ```rs
    /// tts_holder.load_sbv2file("tsukuyomi", std::fs::read("tsukuyomi.sbv2")?)?;
    /// ```
    pub fn load_sbv2file<I: Into<TTSIdent>, P: AsRef<[u8]>>(
        &mut self,
        ident: I,
        sbv2_bytes: P,
    ) -> Result<()> {
        let (style_vectors, vits2) = crate::sbv2file::parse_sbv2file(sbv2_bytes)?;
        self.load(ident, style_vectors, vits2)?;
        Ok(())
    }

    /// Load a style vector and onnx model binary
    ///
    /// # Examples
    ///
    /// ```rs
    /// tts_holder.load("tsukuyomi", std::fs::read("style_vectors.json")?, std::fs::read("model.onnx")?)?;
    /// ```
    pub fn load<I: Into<TTSIdent>, P: AsRef<[u8]>>(
        &mut self,
        ident: I,
        style_vectors_bytes: P,
        vits2_bytes: P,
    ) -> Result<()> {
        let ident = ident.into();
        if self.find_model(ident.clone()).is_err() {
            self.models.push(TTSModel {
                vits2: model::load_model(vits2_bytes, false)?,
                style_vectors: style::load_style(style_vectors_bytes)?,
                ident,
            })
        }
        Ok(())
    }

    /// Unload a model
    pub fn unload<I: Into<TTSIdent>>(&mut self, ident: I) -> bool {
        let ident = ident.into();
        if let Some((i, _)) = self
            .models
            .iter()
            .enumerate()
            .find(|(_, m)| m.ident == ident)
        {
            self.models.remove(i);
            true
        } else {
            false
        }
    }

    /// Parse text and return the input for synthesize
    ///
    /// # Note
    /// This function is for low-level usage, use `easy_synthesize` for high-level usage.
    #[allow(clippy::type_complexity)]
    pub fn parse_text(
        &self,
        text: &str,
    ) -> Result<(Array2<f32>, Array1<i64>, Array1<i64>, Array1<i64>)> {
        crate::tts_util::parse_text_blocking(
            text,
            &self.jtalk,
            &self.tokenizer,
            |token_ids, attention_masks| {
                crate::bert::predict(&self.bert, token_ids, attention_masks)
            },
        )
    }

    fn find_model<I: Into<TTSIdent>>(&self, ident: I) -> Result<&TTSModel> {
        let ident = ident.into();
        self.models
            .iter()
            .find(|m| m.ident == ident)
            .ok_or(Error::ModelNotFoundError(ident.to_string()))
    }

    /// Get style vector by style id and weight
    ///
    /// # Note
    /// This function is for low-level usage, use `easy_synthesize` for high-level usage.
    pub fn get_style_vector<I: Into<TTSIdent>>(
        &self,
        ident: I,
        style_id: i32,
        weight: f32,
    ) -> Result<Array1<f32>> {
        style::get_style_vector(&self.find_model(ident)?.style_vectors, style_id, weight)
    }

    /// Synthesize text to audio
    ///
    /// # Examples
    ///
    /// ```rs
    /// let audio = tts_holder.easy_synthesize("tsukuyomi", "こんにちは", 0, SynthesizeOptions::default())?;
    /// ```
    pub fn easy_synthesize<I: Into<TTSIdent> + Copy>(
        &self,
        ident: I,
        text: &str,
        style_id: i32,
        options: SynthesizeOptions,
    ) -> Result<Vec<u8>> {
        let style_vector = self.get_style_vector(ident, style_id, options.style_weight)?;
        let audio_array = if options.split_sentences {
            let texts: Vec<&str> = text.split('\n').collect();
            let mut audios = vec![];
            for (i, t) in texts.iter().enumerate() {
                if t.is_empty() {
                    continue;
                }
                let (bert_ori, phones, tones, lang_ids) = self.parse_text(t)?;
                let audio = model::synthesize(
                    &self.find_model(ident)?.vits2,
                    bert_ori.to_owned(),
                    phones,
                    tones,
                    lang_ids,
                    style_vector.clone(),
                    options.sdp_ratio,
                    options.length_scale,
                )?;
                audios.push(audio.clone());
                if i != texts.len() - 1 {
                    audios.push(Array3::zeros((1, 1, 22050)));
                }
            }
            concatenate(
                Axis(2),
                &audios.iter().map(|x| x.view()).collect::<Vec<_>>(),
            )?
        } else {
            let (bert_ori, phones, tones, lang_ids) = self.parse_text(text)?;
            model::synthesize(
                &self.find_model(ident)?.vits2,
                bert_ori.to_owned(),
                phones,
                tones,
                lang_ids,
                style_vector,
                options.sdp_ratio,
                options.length_scale,
            )?
        };
        tts_util::array_to_vec(audio_array)
    }

    /// Synthesize text to audio
    ///
    /// # Note
    /// This function is for low-level usage, use `easy_synthesize` for high-level usage.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize<I: Into<TTSIdent>>(
        &self,
        ident: I,
        bert_ori: Array2<f32>,
        phones: Array1<i64>,
        tones: Array1<i64>,
        lang_ids: Array1<i64>,
        style_vector: Array1<f32>,
        sdp_ratio: f32,
        length_scale: f32,
    ) -> Result<Vec<u8>> {
        let audio_array = model::synthesize(
            &self.find_model(ident)?.vits2,
            bert_ori.to_owned(),
            phones,
            tones,
            lang_ids,
            style_vector,
            sdp_ratio,
            length_scale,
        )?;
        tts_util::array_to_vec(audio_array)
    }
}

/// Synthesize options
///
/// # Fields
/// - `sdp_ratio`: SDP ratio
/// - `length_scale`: Length scale
/// - `style_weight`: Style weight
/// - `split_sentences`: Split sentences
pub struct SynthesizeOptions {
    pub sdp_ratio: f32,
    pub length_scale: f32,
    pub style_weight: f32,
    pub split_sentences: bool,
}

impl Default for SynthesizeOptions {
    fn default() -> Self {
        SynthesizeOptions {
            sdp_ratio: 0.0,
            length_scale: 1.0,
            style_weight: 1.0,
            split_sentences: true,
        }
    }
}
