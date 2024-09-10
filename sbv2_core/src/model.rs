use crate::error::Result;
use hound::{SampleFormat, WavSpec, WavWriter};
use ndarray::{array, Array1, Array2, Axis};
use ort::Session;

fn write_wav(file_path: &str, audio: &[f32], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1, // モノラルの場合。ステレオなどの場合は2に変更
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(file_path, spec)?;
    for &sample in audio {
        let int_sample = (sample * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        writer.write_sample(int_sample)?;
    }
    writer.finalize()?;

    Ok(())
}

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
        "sid" => array![0_i64],
        "tones" => tones,
        "language" => lang_ids,
        "bert" => bert,
        "ja_bert" => style_vector,
    }?)?;

    let audio_array = outputs.get("output").unwrap().try_extract_tensor::<f32>()?;
    write_wav("output.wav", audio_array.as_slice().unwrap(), 44100)?;
    Ok(())
}
