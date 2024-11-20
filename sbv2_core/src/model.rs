use crate::error::Result;
use ndarray::{array, Array1, Array2, Array3, Axis, Ix3};
use ort::{GraphOptimizationLevel, Session};

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
    sid: Array1<i64>,
    tones: Array1<i64>,
    lang_ids: Array1<i64>,
    style_vector: Array1<f32>,
    sdp_ratio: f32,
    length_scale: f32,
    noise_scale: f32,
    noise_scale_w: f32,
) -> Result<Array3<f32>> {
    let bert = bert_ori.insert_axis(Axis(0));
    let x_tst_lengths: Array1<i64> = array![x_tst.shape()[0] as i64];
    let x_tst = x_tst.insert_axis(Axis(0));
    let lang_ids = lang_ids.insert_axis(Axis(0));
    let tones = tones.insert_axis(Axis(0));
    let style_vector = style_vector.insert_axis(Axis(0));
    let outputs = session.run(ort::inputs! {
        "x_tst" => x_tst,
        "x_tst_lengths" => x_tst_lengths,
        "sid" => sid,
        "tones" => tones,
        "language" => lang_ids,
        "bert" => bert,
        "style_vec" => style_vector,
        "sdp_ratio" => array![sdp_ratio],
        "length_scale" => array![length_scale],
        "noise_scale" => array![noise_scale],
        "noise_scale_w" => array![noise_scale_w]
    }?)?;

    let audio_array = outputs["output"]
        .try_extract_tensor::<f32>()?
        .into_dimensionality::<Ix3>()?
        .to_owned();

    Ok(audio_array)
}
