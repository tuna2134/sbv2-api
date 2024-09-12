use pyo3::prelude::*;
use sbv2_core::tts::TTSModelHolder;

use crate::style::StyleVector;

#[pyclass]
pub struct TTSModel {
    pub model: TTSModelHolder,
}

#[pymethods]
impl TTSModel {
    #[new]
    fn new(bert_model_bytes: Vec<u8>, tokenizer_bytes: Vec<u8>) -> anyhow::Result<Self> {
        Ok(Self {
            model: TTSModelHolder::new(bert_model_bytes, tokenizer_bytes)?,
        })
    }

    fn load_sbv2file(&mut self, ident: String, sbv2file_bytes: Vec<u8>) -> anyhow::Result<()> {
        self.model.load_sbv2file(ident, sbv2file_bytes)?;
        Ok(())
    }

    fn get_style_vector(
        &self,
        ident: String,
        style_id: i32,
        weight: f32,
    ) -> anyhow::Result<StyleVector> {
        Ok(StyleVector::new(
            self.model.get_style_vector(ident, style_id, weight)?,
        ))
    }

    fn synthesize(
        &self,
        text: String,
        ident: String,
        style_vector: StyleVector,
        sdp_ratio: f32,
        length_scale: f32,
    ) -> anyhow::Result<Vec<u8>> {
        let (bert_ori, phones, tones, lang_ids) = self.model.parse_text(&text)?;
        Ok(self.model.synthesize(
            ident,
            bert_ori,
            phones,
            tones,
            lang_ids,
            style_vector.get(),
            sdp_ratio,
            length_scale,
        )?)
    }
}
