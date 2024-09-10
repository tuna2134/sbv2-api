use ndarray::{concatenate, s, Array, Array1, Array2, Axis};
use sbv2_core::{bert, error, jtalk, model, nlp, norm, style, utils};

fn main() -> error::Result<()> {
    let text = "こんにちは,世界!";

    let normalized_text = norm::normalize_text(text);
    println!("{}", normalized_text);

    let jtalk = jtalk::JTalk::new()?;
    let (phones, tones, mut word2ph) = jtalk.g2p(&normalized_text)?;
    let (phones, tones, lang_ids) = nlp::cleaned_text_to_sequence(phones, tones);

    // add black
    let phones = utils::intersperse(&phones, 0);
    let tones = utils::intersperse(&tones, 0);
    let lang_ids = utils::intersperse(&lang_ids, 0);
    for i in 0..word2ph.len() {
        word2ph[i] *= 2;
    }
    word2ph[0] += 1;

    let tokenizer = jtalk::get_tokenizer()?;
    let (token_ids, attention_masks) = jtalk::tokenize(&normalized_text, &tokenizer)?;

    let session = bert::load_model("models/debert.onnx")?;
    let bert_content = bert::predict(&session, token_ids, attention_masks)?;

    assert!(
        word2ph.len() == normalized_text.chars().count() + 2,
        "{} {}",
        word2ph.len(),
        normalized_text.chars().count()
    );

    let mut phone_level_feature = vec![];
    for i in 0..word2ph.len() {
        // repeat_feature = np.tile(bert_content[i], (word2ph[i], 1))
        let repeat_feature = {
            let (reps_rows, reps_cols) = (word2ph[i], 1);
            let arr_len = bert_content.slice(s![i, ..]).len();

            let mut results: Array2<f32> = Array::zeros((reps_rows as usize, arr_len * reps_cols));

            for j in 0..reps_rows {
                for k in 0..reps_cols {
                    let mut view = results.slice_mut(s![j, k * arr_len..(k + 1) * arr_len]);
                    view.assign(&bert_content.slice(s![i, ..]));
                }
            }
            results
        };
        phone_level_feature.push(repeat_feature);
    }
    // ph = np.concatenate(phone_level_feature, axis=0)
    // bert_ori = ph.T
    let phone_level_feature = concatenate(
        Axis(0),
        &phone_level_feature
            .iter()
            .map(|x| x.view())
            .collect::<Vec<_>>(),
    )?;
    let bert_ori = phone_level_feature.t();
    println!("{:?}", bert_ori.shape());
    // let data: Array2<f32> = Array2::from_shape_vec((bert_ori.shape()[0], bert_ori.shape()[1]), bert_ori.to_vec()).unwrap();
    // data

    let session = bert::load_model("models/model_opt.onnx")?;
    let style_vectors = style::load_style("models/style_vectors.json")?;
    let style_vector = style::get_style_vector(style_vectors, 0, 1.0)?;
    model::synthesize(
        &session,
        bert_ori.to_owned(),
        Array1::from_vec(phones.iter().map(|x| *x as i64).collect()),
        Array1::from_vec(tones.iter().map(|x| *x as i64).collect()),
        Array1::from_vec(lang_ids.iter().map(|x| *x as i64).collect()),
        style_vector,
    )?;

    Ok(())
}
