use once_cell::sync::Lazy;
use sbv2_core::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
mod array_helper;

static JTALK: Lazy<jtalk::JTalk> = Lazy::new(|| jtalk::JTalk::new().unwrap());

#[wasm_bindgen]
pub struct TokenizerWrap {
    tokenizer: tokenizer::Tokenizer,
}

#[wasm_bindgen]
pub fn load_tokenizer(s: js_sys::JsString) -> Result<TokenizerWrap, JsError> {
    if let Some(s) = s.as_string() {
        Ok(TokenizerWrap {
            tokenizer: tokenizer::Tokenizer::from_bytes(s.as_bytes())
                .map_err(|e| JsError::new(&e.to_string()))?,
        })
    } else {
        Err(JsError::new("invalid utf8"))
    }
}

#[wasm_bindgen]
pub struct StyleVectorWrap {
    style_vector: ndarray::Array2<f32>,
}

#[wasm_bindgen]
pub fn load_sbv2file(buf: js_sys::Uint8Array) -> Result<js_sys::Array, JsError> {
    let (style_vectors, vits2) = sbv2file::parse_sbv2file(array_helper::array8_to_vec8(buf))?;
    let buf = array_helper::vec8_to_array8(vits2);
    Ok(array_helper::vec_to_array(vec![
        StyleVectorWrap {
            style_vector: style::load_style(style_vectors)?,
        }
        .into(),
        buf.into(),
    ]))
}

#[allow(clippy::too_many_arguments)]
#[wasm_bindgen]
pub async fn synthesize(
    text: &str,
    tokenizer: &TokenizerWrap,
    bert_predict_fn: js_sys::Function,
    synthesize_fn: js_sys::Function,
    sdp_ratio: f32,
    length_scale: f32,
    style_id: i32,
    style_weight: f32,
    style_vectors: &StyleVectorWrap,
) -> Result<js_sys::Uint8Array, JsError> {
    let synthesize_wrap = |bert_ori: ndarray::Array2<f32>,
                           x_tst: ndarray::Array1<i64>,
                           tones: ndarray::Array1<i64>,
                           lang_ids: ndarray::Array1<i64>,
                           style_vector: ndarray::Array1<f32>,
                           sdp_ratio: f32,
                           length_scale: f32| async move {
        let arr = array_helper::vec_to_array(vec![
            array_helper::array2_f32_to_array(bert_ori).into(),
            array_helper::vec64_to_array64(x_tst.to_vec()).into(),
            array_helper::vec64_to_array64(tones.to_vec()).into(),
            array_helper::vec64_to_array64(lang_ids.to_vec()).into(),
            array_helper::vec_f32_to_array_f32(style_vector.to_vec()).into(),
            sdp_ratio.into(),
            length_scale.into(),
        ]);
        let res = synthesize_fn
            .apply(&js_sys::Object::new().into(), &arr)
            .map_err(|e| {
                error::Error::OtherError(e.as_string().unwrap_or("unknown".to_string()))
            })?;
        let res = JsFuture::from(Into::<js_sys::Promise>::into(res))
            .await
            .map_err(|e| {
                sbv2_core::error::Error::OtherError(e.as_string().unwrap_or("unknown".to_string()))
            })?;
        array_helper::array_to_array3_f32(res)
    };
    let (bert_ori, phones, tones, lang_ids) = tts_util::parse_text(
        text,
        &JTALK,
        &tokenizer.tokenizer,
        |token_ids: Vec<i64>, attention_masks: Vec<i64>| {
            Box::pin(async move {
                let arr = array_helper::vec_to_array(vec![
                    array_helper::vec64_to_array64(token_ids).into(),
                    array_helper::vec64_to_array64(attention_masks).into(),
                ]);
                let res = bert_predict_fn
                    .apply(&js_sys::Object::new().into(), &arr)
                    .map_err(|e| {
                        error::Error::OtherError(e.as_string().unwrap_or("unknown".to_string()))
                    })?;
                let res = JsFuture::from(Into::<js_sys::Promise>::into(res))
                    .await
                    .map_err(|e| {
                        sbv2_core::error::Error::OtherError(
                            e.as_string().unwrap_or("unknown".to_string()),
                        )
                    })?;
                array_helper::array_to_array2_f32(res)
            })
        },
    )
    .await?;
    let audio = synthesize_wrap(
        bert_ori.to_owned(),
        phones,
        tones,
        lang_ids,
        style::get_style_vector(&style_vectors.style_vector, style_id, style_weight)?,
        sdp_ratio,
        length_scale,
    )
    .await?;
    Ok(array_helper::vec8_to_array8(tts_util::array_to_vec(audio)?))
}
