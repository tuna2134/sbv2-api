use crate::error::Result;
use hound::{SampleFormat, WavSpec, WavWriter};
use ndarray::{array, s, Array1, Array2, Axis};
use ort::{GraphOptimizationLevel, Session};
use std::io::Cursor;

#[allow(clippy::vec_init_then_push, unused_variables)]
pub fn load_model<P: AsRef<[u8]>>(model_file: P, bert: bool) -> Result<Session> {
    let mut exp = Vec::new();
    #[cfg(feature = "tensorrt")]
    {
        if bert {
            exp.push(
                ort::TensorRTExecutionProvider::default()
                    .with_fp16(true)
                    .with_profile_min_shapes("input_ids:1x1,attention_mask:1x1")
                    .with_profile_max_shapes("input_ids:1x100,attention_mask:1x100")
                    .with_profile_opt_shapes("input_ids:1x25,attention_mask:1x25")
                    .build(),
            );
        }
    }
    #[cfg(feature = "cuda")]
    {
        #[allow(unused_mut)]
        let mut cuda = ort::CUDAExecutionProvider::default()
            .with_conv_algorithm_search(ort::CUDAExecutionProviderCuDNNConvAlgoSearch::Default);
        #[cfg(feature = "cuda_tf32")]
        {
            cuda = cuda.with_tf32(true);
        }
        exp.push(cuda.build());
    }
    #[cfg(feature = "directml")]
    {
        exp.push(ort::DirectMLExecutionProvider::default().build());
    }
    #[cfg(feature = "coreml")]
    {
        exp.push(ort::CoreMLExecutionProvider::default().build());
    }
    exp.push(ort::CPUExecutionProvider::default().build());
    Ok(Session::builder()?
        .with_execution_providers(exp)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(num_cpus::get_physical())?
        .with_parallel_execution(true)?
        .with_inter_threads(num_cpus::get_physical())?
        .commit_from_memory(model_file.as_ref())?)
}

#[allow(clippy::too_many_arguments)]
pub fn synthesize(
    session: &Session,
    bert_ori: Array2<f32>,
    x_tst: Array1<i64>,
    tones: Array1<i64>,
    lang_ids: Array1<i64>,
    style_vector: Array1<f32>,
    sdp_ratio: f32,
    length_scale: f32,
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
        "style_vec" => style_vector,
        "sdp_ratio" => array![sdp_ratio],
        "length_scale" => array![length_scale],
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
