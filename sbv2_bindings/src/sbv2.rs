use pyo3::prelude::*;
use pyo3::types::PyBytes;
use sbv2_core::tts::TTSModelHolder;

use crate::style::StyleVector;

use std::fs;

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

    #[staticmethod]
    fn from_path(bert_model_path: String, tokenizer_path: String) -> anyhow::Result<Self> {
        Ok(Self {
            model: TTSModelHolder::new(
                fs::read(bert_model_path)?,
                fs::read(tokenizer_path)?,
            )?,
        })
    }

    fn load_sbv2file(&mut self, ident: String, sbv2file_bytes: Vec<u8>) -> anyhow::Result<()> {
        self.model.load_sbv2file(ident, sbv2file_bytes)?;
        Ok(())
    }

    fn load_sbv2file_from_path(&mut self, ident: String, sbv2file_path: String) -> anyhow::Result<()> {
        self.model.load_sbv2file(ident, fs::read(sbv2file_path)?)?;
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

    fn synthesize<'p>(
        &'p self,
        py: Python<'p>,
        text: String,
        ident: String,
        style_vector: StyleVector,
        sdp_ratio: f32,
        length_scale: f32,
    ) -> anyhow::Result<Bound<PyBytes>> {
        let (bert_ori, phones, tones, lang_ids) = self.model.parse_text(&text)?;
        let data = self.model.synthesize(
            ident,
            bert_ori,
            phones,
            tones,
            lang_ids,
            style_vector.get(),
            sdp_ratio,
            length_scale,
        )?;
        Ok(PyBytes::new_bound(py, &data))
    }
}
