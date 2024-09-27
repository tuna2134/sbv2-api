use once_cell::sync::Lazy;
use sbv2_core::*;
use wasm_bindgen::prelude::*;

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
    let mut body = vec![0; buf.length() as usize];
    buf.copy_to(&mut body[..]);
    let (style_vectors, vits2) = sbv2file::parse_sbv2file(body)?;
    let buf = js_sys::Uint8Array::new_with_length(vits2.len() as u32);
    buf.copy_from(&vits2);
    let arr = js_sys::Array::new_with_length(2);
    arr.set(
        0,
        StyleVectorWrap {
            style_vector: style::load_style(style_vectors)?,
        }
        .into(),
    );
    arr.set(1, buf.into());
    Ok(arr)
}

#[wasm_bindgen]
pub fn synthesize(
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
    fn synthesize_wrap(
        bert_ori: ndarray::Array2<f32>,
        x_tst: ndarray::Array1<i64>,
        tones: ndarray::Array1<i64>,
        lang_ids: ndarray::Array1<i64>,
        style_vector: ndarray::Array1<f32>,
        sdp_ratio: f32,
        length_scale: f32,
    ) -> error::Result<ndarray::Array3<f32>> {
        todo!()
    }
    let (bert_ori, phones, tones, lang_ids) = tts_util::parse_text(
        text,
        &JTALK,
        &tokenizer.tokenizer,
        |token_ids: Vec<i64>, attention_masks: Vec<i64>| {
            let token_ids_ = js_sys::BigInt64Array::new_with_length(token_ids.len() as u32);
            token_ids_.copy_from(&token_ids);
            let attention_masks_ =
                js_sys::BigInt64Array::new_with_length(attention_masks.len() as u32);
            attention_masks_.copy_from(&attention_masks);
            let arr = js_sys::Array::new_with_length(2);
            arr.set(0, token_ids_.into());
            arr.set(1, attention_masks_.into());
            let res = bert_predict_fn
                .apply(&js_sys::Object::new().into(), &arr)
                .map_err(|e| {
                    error::Error::OtherError(e.as_string().unwrap_or("unknown".to_string()))
                })?;
            let res: js_sys::Array = res.into();
            Ok(todo!())
        },
    )?;
    let audio = synthesize_wrap(
        bert_ori.to_owned(),
        phones,
        tones,
        lang_ids,
        style::get_style_vector(&style_vectors.style_vector, style_id, style_weight)?,
        sdp_ratio,
        length_scale,
    )?;
    let vec = tts_util::array_to_vec(audio)?;
    let buf = js_sys::Uint8Array::new_with_length(vec.len() as u32);
    buf.copy_from(&vec);
    Ok(buf)
}
