use crate::error::{Error, Result};
use crate::{bert, jtalk, model, nlp, norm, style, tokenizer, utils};
use hound::{SampleFormat, WavSpec, WavWriter};
use ndarray::{concatenate, s, Array, Array1, Array2, Array3, Axis};
use ort::Session;
use std::io::{Cursor, Read};
use tar::Archive;
use tokenizers::Tokenizer;
use zstd::decode_all;

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

pub struct TTSModelHolder {
    tokenizer: Tokenizer,
    bert: Session,
    models: Vec<TTSModel>,
    jtalk: jtalk::JTalk,
}

impl TTSModelHolder {
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

    pub fn models(&self) -> Vec<String> {
        self.models.iter().map(|m| m.ident.to_string()).collect()
    }

    pub fn load_sbv2file<I: Into<TTSIdent>, P: AsRef<[u8]>>(
        &mut self,
        ident: I,
        sbv2_bytes: P,
    ) -> Result<()> {
        let mut arc = Archive::new(Cursor::new(decode_all(Cursor::new(sbv2_bytes.as_ref()))?));
        let mut vits2 = None;
        let mut style_vectors = None;
        let mut et = arc.entries()?;
        while let Some(Ok(mut e)) = et.next() {
            let pth = String::from_utf8_lossy(&e.path_bytes()).to_string();
            let mut b = Vec::with_capacity(e.size() as usize);
            e.read_to_end(&mut b)?;
            match pth.as_str() {
                "model.onnx" => vits2 = Some(b),
                "style_vectors.json" => style_vectors = Some(b),
                _ => continue,
            }
        }
        if style_vectors.is_none() {
            return Err(Error::ModelNotFoundError("style_vectors".to_string()));
        }
        if vits2.is_none() {
            return Err(Error::ModelNotFoundError("vits2".to_string()));
        }
        self.load(ident, style_vectors.unwrap(), vits2.unwrap())?;
        Ok(())
    }

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

    #[allow(clippy::type_complexity)]
    pub fn parse_text(
        &self,
        text: &str,
    ) -> Result<(Array2<f32>, Array1<i64>, Array1<i64>, Array1<i64>)> {
        let normalized_text = norm::normalize_text(text);

        let process = self.jtalk.process_text(&normalized_text)?;
        let (phones, tones, mut word2ph) = process.g2p()?;
        let (phones, tones, lang_ids) = nlp::cleaned_text_to_sequence(phones, tones);

        let phones = utils::intersperse(&phones, 0);
        let tones = utils::intersperse(&tones, 0);
        let lang_ids = utils::intersperse(&lang_ids, 0);
        for item in &mut word2ph {
            *item *= 2;
        }
        word2ph[0] += 1;

        let text = {
            let (seq_text, _) = process.text_to_seq_kata()?;
            seq_text.join("")
        };
        let (token_ids, attention_masks) = tokenizer::tokenize(&text, &self.tokenizer)?;

        let bert_content = bert::predict(&self.bert, token_ids, attention_masks)?;

        assert!(
            word2ph.len() == text.chars().count() + 2,
            "{} {}",
            word2ph.len(),
            normalized_text.chars().count()
        );

        let mut phone_level_feature = vec![];
        for (i, reps) in word2ph.iter().enumerate() {
            let repeat_feature = {
                let (reps_rows, reps_cols) = (*reps, 1);
                let arr_len = bert_content.slice(s![i, ..]).len();

                let mut results: Array2<f32> =
                    Array::zeros((reps_rows as usize, arr_len * reps_cols));

                for j in 0..reps_rows {
                    for k in 0..reps_cols {
                        let mut view = results.slice_mut(s![j, k * arr_len..(k + 1) * arr_len]);
                        view.assign(&bert_content.slice(s![i, ..]));
                    }
                }
                results
            };
            phone_level_feature.push(repeat_feature);
        }
        let phone_level_feature = concatenate(
            Axis(0),
            &phone_level_feature
                .iter()
                .map(|x| x.view())
                .collect::<Vec<_>>(),
        )?;
        let bert_ori = phone_level_feature.t();
        Ok((
            bert_ori.to_owned(),
            phones.into(),
            tones.into(),
            lang_ids.into(),
        ))
    }

    fn find_model<I: Into<TTSIdent>>(&self, ident: I) -> Result<&TTSModel> {
        let ident = ident.into();
        self.models
            .iter()
            .find(|m| m.ident == ident)
            .ok_or(Error::ModelNotFoundError(ident.to_string()))
    }

    pub fn get_style_vector<I: Into<TTSIdent>>(
        &self,
        ident: I,
        style_id: i32,
        weight: f32,
    ) -> Result<Array1<f32>> {
        style::get_style_vector(&self.find_model(ident)?.style_vectors, style_id, weight)
    }

    pub fn easy_synthesize<I: Into<TTSIdent> + Copy>(
        &self,
        ident: I,
        text: &str,
        style_id: i32,
        options: SynthesizeOptions,
    ) -> Result<Vec<u8>> {
        let style_vector = self.get_style_vector(ident, style_id, options.style_weight)?;
        let audio_array = if options.split_sentences {
            let texts: Vec<&str> = text.split("\n").collect();
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
                audios.push(audio);
                if i != texts.len() - 1 {
                    audios.push(Array3::zeros((1, 22050, 1)));
                }
            }
            concatenate(
                Axis(0),
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
        Ok(Self::array_to_vec(audio_array)?)
    }

    fn array_to_vec(audio_array: Array3<f32>) -> Result<Vec<u8>> {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut cursor = Cursor::new(Vec::new());
        let mut writer = WavWriter::new(&mut cursor, spec)?;
        for i in 0..audio_array.shape()[0] {
            let output = audio_array.slice(s![i, 0, ..]).to_vec();
            for sample in output {
                writer.write_sample(sample)?;
            }
        }
        writer.finalize()?;
        Ok(cursor.into_inner())
    }

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
        Ok(Self::array_to_vec(audio_array)?)
    }
}

pub struct SynthesizeOptions {
    sdp_ratio: f32,
    length_scale: f32,
    style_weight: f32,
    split_sentences: bool,
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
