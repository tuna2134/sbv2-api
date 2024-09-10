use crate::error::Result;
use ndarray::Array2;
use ort::{GraphOptimizationLevel, Session};

pub fn load_model(model_file: &str) -> Result<Session> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_file(model_file)?;
    Ok(session)
}

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

    let output = outputs.get("output").unwrap();

    let content = output.try_extract_tensor::<f32>()?.to_owned();
    println!("{:?}", content);

    Ok(Array2::from_shape_vec(
        (content.shape()[0], content.shape()[1]),
        content.into_raw_vec(),
    )
    .unwrap())
}
