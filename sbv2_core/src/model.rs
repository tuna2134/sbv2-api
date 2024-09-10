use crate::error::Result;
use hound::{SampleFormat, WavSpec, WavWriter};
use ndarray::{array, s, Array1, Array2, Axis};
use ort::{GraphOptimizationLevel, Session};
use std::io::Cursor;

pub fn load_model(model_file: &str) -> Result<Session> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_intra_threads(num_cpus::get_physical())?
        .with_parallel_execution(true)?
        .with_inter_threads(num_cpus::get_physical())?
        .commit_from_file(model_file)?;
    Ok(session)
}

pub fn synthesize(
    session: &Session,
    bert_ori: Array2<f32>,
    x_tst: Array1<i64>,
    tones: Array1<i64>,
    lang_ids: Array1<i64>,
    style_vector: Array1<f32>,
) -> Result<Vec<u8>> {
    let bert = bert_ori.insert_axis(Axis(0));
    let x_tst_lengths: Array1<i64> = array![x_tst.shape()[0] as i64];
    let x_tst = x_tst.insert_axis(Axis(0));
    let lang_ids = lang_ids.insert_axis(Axis(0));
    let tones = tones.insert_axis(Axis(0));
    let style_vector = style_vector.insert_axis(Axis(0));
    let outputs = session.run(ort::inputs! {
        "x_tst" => x_tst,
        "x_tst_lengths" => x_tst_lengths,
        "sid" => array![0_i64],
        "tones" => tones,
        "language" => lang_ids,
        "bert" => bert,
        "style_vector" => style_vector,
    }?)?;

    let audio_array = outputs
        .get("output")
        .unwrap()
        .try_extract_tensor::<f32>()?
        .to_owned();

    let buffer = {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut cursor = Cursor::new(Vec::new());
        let mut writer = WavWriter::new(&mut cursor, spec)?;
        for i in 0..audio_array.shape()[0] {
            let output = audio_array.slice(s![i, 0, ..]).to_vec();
            for sample in output {
                writer.write_sample(sample)?;
            }
        }
        writer.finalize()?;
        cursor.into_inner()
    };

    Ok(buffer)
}
