use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] tokenizers::Error),
    #[error("JPreprocess error: {0}")]
    JPreprocessError(#[from] jpreprocess::error::JPreprocessError),
    #[error("ONNX error: {0}")]
    OrtError(#[from] ort::Error),
    #[error("NDArray error: {0}")]
    NdArrayError(#[from] ndarray::ShapeError),
    #[error("Value error: {0}")]
    ValueError(String),
    #[error("Serde_json error: {0}")]
    SerdeJsonError(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("hound error: {0}")]
    HoundError(#[from] hound::Error),
    #[error("model not found error")]
    ModelNotFoundError(String),
}

pub type Result<T> = std::result::Result<T, Error>;
