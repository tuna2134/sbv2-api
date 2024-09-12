use pyo3::prelude::*;
use pyo3::types::PyBytes;
use sbv2_core::tts::TTSModelHolder;

use crate::style::StyleVector;

use std::fs;

/// TTSModel class
///
/// 音声合成するために使うクラス
///
/// Parameters
/// ----------
/// bert_model_bytes : bytes
///     BERTモデルのバイナリデータ
/// tokenizer_bytes : bytes
///     トークナイザーのバイナリデータ
#[pyclass]
pub struct TTSModel {
    pub model: TTSModelHolder,
}

#[pymethods]
impl TTSModel {
    #[new]
    fn new(bert_model_bytes: Vec<u8>, tokenizer_bytes: Vec<u8>) -> anyhow::Result<Self> {
        Ok(Self {
            model: TTSModelHolder::new(bert_model_bytes, tokenizer_bytes)?,
        })
    }

    /// パスからTTSModelインスタンスを生成する
    ///
    /// Parameters
    /// ----------
    /// bert_model_path : str
    ///     BERTモデルのパス
    /// tokenizer_path : str
    ///     トークナイザーのパス
    #[staticmethod]
    fn from_path(bert_model_path: String, tokenizer_path: String) -> anyhow::Result<Self> {
        Ok(Self {
            model: TTSModelHolder::new(fs::read(bert_model_path)?, fs::read(tokenizer_path)?)?,
        })
    }

    /// SBV2ファイルを読み込む
    ///
    /// Parameters
    /// ----------
    /// ident : str
    ///     識別子
    /// sbv2file_bytes : bytes
    ///     SBV2ファイルのバイナリデータ
    fn load_sbv2file(&mut self, ident: String, sbv2file_bytes: Vec<u8>) -> anyhow::Result<()> {
        self.model.load_sbv2file(ident, sbv2file_bytes)?;
        Ok(())
    }

    /// パスからSBV2ファイルを読み込む
    ///
    /// Parameters
    /// ----------
    /// ident : str
    ///     識別子
    /// sbv2file_path : str
    ///     SBV2ファイルのパス
    fn load_sbv2file_from_path(
        &mut self,
        ident: String,
        sbv2file_path: String,
    ) -> anyhow::Result<()> {
        self.model.load_sbv2file(ident, fs::read(sbv2file_path)?)?;
        Ok(())
    }

    /// スタイルベクトルを取得する
    ///
    /// Parameters
    /// ----------
    /// ident : str
    ///     識別子
    /// style_id : int
    ///     スタイルID
    /// weight : float
    ///     重み
    ///
    /// Returns
    /// -------
    /// style_vector : StyleVector
    ///     スタイルベクトル
    fn get_style_vector(
        &self,
        ident: String,
        style_id: i32,
        weight: f32,
    ) -> anyhow::Result<StyleVector> {
        Ok(StyleVector::new(
            self.model.get_style_vector(ident, style_id, weight)?,
        ))
    }

    /// テキストから音声を合成する
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///     テキスト
    /// ident : str
    ///     識別子
    /// style_vector : StyleVector
    ///     スタイルベクトル
    /// sdp_ratio : float
    ///     SDP比率
    /// length_scale : float
    ///     音声の長さのスケール
    ///
    /// Returns
    /// -------
    /// voice_data : bytes
    ///     音声データ
    fn synthesize<'p>(
        &'p self,
        py: Python<'p>,
        text: String,
        ident: String,
        style_vector: StyleVector,
        sdp_ratio: f32,
        length_scale: f32,
    ) -> anyhow::Result<Bound<PyBytes>> {
        let (bert_ori, phones, tones, lang_ids) = self.model.parse_text(&text)?;
        let data = self.model.synthesize(
            ident,
            bert_ori,
            phones,
            tones,
            lang_ids,
            style_vector.get(),
            sdp_ratio,
            length_scale,
        )?;
        Ok(PyBytes::new_bound(py, &data))
    }
}
