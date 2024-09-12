use ndarray::Array1;
use pyo3::prelude::*;

/// StyleVector class
///
/// スタイルベクトルを表すクラス
#[pyclass]
#[derive(Clone)]
pub struct StyleVector(Array1<f32>);

impl StyleVector {
    pub fn new(data: Array1<f32>) -> Self {
        StyleVector(data)
    }

    pub fn get(&self) -> Array1<f32> {
        self.0.clone()
    }
}
