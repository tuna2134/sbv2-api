use crate::error::Result;
use ndarray::{s, Array1, Array2};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Data {
    pub shape: [usize; 2],
    pub data: Vec<Vec<f32>>,
}

pub fn load_style(path: &str) -> Result<Array2<f32>> {
    let data: Data = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    Ok(Array2::from_shape_vec(
        data.shape,
        data.data.iter().flatten().copied().collect(),
    )?)
}

pub fn get_style_vector(
    style_vectors: Array2<f32>,
    style_id: i32,
    weight: f32,
) -> Result<Array1<f32>> {
    let mean = style_vectors.slice(s![0, ..]).to_owned();
    let style_vector = style_vectors.slice(s![style_id as usize, ..]).to_owned();
    let diff = (style_vector - &mean) * weight;
    Ok(mean + &diff)
}
