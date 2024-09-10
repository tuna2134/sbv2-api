use pyo3::prelude::*;
use sbv2_core::tts::TTSModel as BaseTTSModel;
use numpy::{convert::IntoPyArray

#[pyclass]
pub struct TTSModel {
    pub model: BaseTTSModel,
}

#[pymethods]
impl TTSModel {
    #[new]
    fn new(bert_model_path: &str, main_model_path: &str, style_vectors_path: &str) -> anyhow::Result<Self> {
        Ok(Self {
            model: BaseTTSModel::new(bert_model_path, main_model_path, style_vectors_path)?,
        })
    }

    fn get_style_vector
}