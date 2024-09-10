use crate::error::Result;
use ndarray::{array, Array1, Array2, Axis};
use ort::Session;

pub fn synthesize(
    session: &Session,
    bert_ori: Array2<f32>,
    x_tst: Array1<i64>,
    tones: Array1<i64>,
    lang_ids: Array1<i64>,
    style_vector: Array1<f32>,
) -> Result<()> {
    let bert = bert_ori.insert_axis(Axis(0));
    let x_tst_lengths: Array1<i64> = array![x_tst.shape()[0] as i64];
    let x_tst = x_tst.insert_axis(Axis(0));
    let lang_ids = lang_ids.insert_axis(Axis(0));
    let tones = tones.insert_axis(Axis(0));
    let style_vector = style_vector.insert_axis(Axis(0));
    let outputs = session.run(ort::inputs! {
        "x_tst" => x_tst,
        "x_tst_lengths" => x_tst_lengths,
        "sid" => array![0 as i64],
        "tones" => tones,
        "language" => lang_ids,
        "bert" => bert,
        "ja_bert" => style_vector,
    }?)?;
    Ok(())
}
