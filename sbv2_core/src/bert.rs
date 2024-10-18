use crate::error::Result;
use ndarray::{Array2, Ix2};
use ort::Session;

pub fn predict(
    session: &Session,
    token_ids: Vec<i64>,
    attention_masks: Vec<i64>,
) -> Result<Array2<f32>> {
    let outputs = session.run(
        ort::inputs! {
            "input_ids" => Array2::from_shape_vec((1, token_ids.len()), token_ids).unwrap(),
            "attention_mask" => Array2::from_shape_vec((1, attention_masks.len()), attention_masks).unwrap(),
        }?
    )?;

    let output = outputs["output"]
        .try_extract_tensor::<f32>()?
        .into_dimensionality::<Ix2>()?
        .to_owned();

    Ok(output)
}
