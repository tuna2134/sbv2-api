pub fn vec8_to_array8(v: Vec<u8>) -> js_sys::Uint8Array {
    let arr = js_sys::Uint8Array::new_with_length(v.len() as u32);
    arr.copy_from(&v);
    arr
}

pub fn vec_f32_to_array_f32(v: Vec<f32>) -> js_sys::Float32Array {
    let arr = js_sys::Float32Array::new_with_length(v.len() as u32);
    arr.copy_from(&v);
    arr
}

pub fn array8_to_vec8(buf: js_sys::Uint8Array) -> Vec<u8> {
    let mut body = vec![0; buf.length() as usize];
    buf.copy_to(&mut body[..]);
    body
}

pub fn vec64_to_array64(v: Vec<i64>) -> js_sys::BigInt64Array {
    let arr = js_sys::BigInt64Array::new_with_length(v.len() as u32);
    arr.copy_from(&v);
    arr
}

pub fn vec_to_array(v: Vec<wasm_bindgen::JsValue>) -> js_sys::Array {
    let arr = js_sys::Array::new_with_length(v.len() as u32);
    for (i, v) in v.into_iter().enumerate() {
        arr.set(i as u32, v);
    }
    arr
}

struct A {
    shape: Vec<u32>,
    data: Vec<f32>,
}

impl TryFrom<wasm_bindgen::JsValue> for A {
    type Error = sbv2_core::error::Error;

    fn try_from(value: wasm_bindgen::JsValue) -> Result<Self, Self::Error> {
        let value: js_sys::Array = value.into();
        let mut shape = vec![];
        let mut data = vec![];
        for (i, v) in value.iter().enumerate() {
            match i {
                0 => {
                    let v: js_sys::Uint32Array = v.into();
                    shape = vec![0; v.length() as usize];
                    v.copy_to(&mut shape);
                }
                1 => {
                    let v: js_sys::Float32Array = v.into();
                    data = vec![0.0; v.length() as usize];
                    v.copy_to(&mut data);
                }
                _ => {}
            };
        }
        Ok(A { shape, data })
    }
}

pub fn array_to_array2_f32(
    a: wasm_bindgen::JsValue,
) -> sbv2_core::error::Result<ndarray::Array2<f32>> {
    let a = A::try_from(a)?;
    if a.shape.len() != 2 {
        return Err(sbv2_core::error::Error::OtherError(
            "Length mismatch".to_string(),
        ));
    }
    let shape = [a.shape[0] as usize, a.shape[1] as usize];
    let arr = ndarray::Array2::from_shape_vec(shape, a.data.to_vec())
        .map_err(|e| sbv2_core::error::Error::OtherError(e.to_string()))?;
    Ok(arr)
}
pub fn array_to_array3_f32(
    a: wasm_bindgen::JsValue,
) -> sbv2_core::error::Result<ndarray::Array3<f32>> {
    let a = A::try_from(a)?;
    if a.shape.len() != 3 {
        return Err(sbv2_core::error::Error::OtherError(
            "Length mismatch".to_string(),
        ));
    }
    let shape = [
        a.shape[0] as usize,
        a.shape[1] as usize,
        a.shape[2] as usize,
    ];
    let arr = ndarray::Array3::from_shape_vec(shape, a.data.to_vec())
        .map_err(|e| sbv2_core::error::Error::OtherError(e.to_string()))?;
    Ok(arr)
}

pub fn array2_f32_to_array(a: ndarray::Array2<f32>) -> js_sys::Array {
    let shape: Vec<wasm_bindgen::JsValue> = a.shape().iter().map(|f| (*f as u32).into()).collect();
    let typed_array = js_sys::Float32Array::new_with_length(a.len() as u32);
    typed_array.copy_from(&a.into_flat().to_vec());
    vec_to_array(vec![vec_to_array(shape).into(), typed_array.into()])
}
