use pyo3::prelude::*;
use sbv2_core::tts::TTSModelHolder;

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

    fn load()
}