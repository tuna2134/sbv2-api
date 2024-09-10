use crate::error::Result;
use crate::{bert, jtalk, model, nlp, norm, style, utils};
use ndarray::{concatenate, s, Array, Array1, Array2, Axis};
use ort::Session;

pub struct TTSModel {
    bert: Session,
    vits2: Session,
    style_vectors: Array2<f32>,
    jtalk: jtalk::JTalk,
}

impl TTSModel {
    pub fn new(
        bert_model_path: &str,
        main_model_path: &str,
        style_vector_path: &str,
    ) -> Result<Self> {
        let bert = model::load_model(bert_model_path)?;
        let vits2 = model::load_model(main_model_path)?;
        let style_vectors = style::load_style(style_vector_path)?;
        let jtalk = jtalk::JTalk::new()?;
        Ok(TTSModel {
            bert,
            vits2,
            style_vectors,
            jtalk,
        })
    }

    pub fn parse_text(
        &self,
        text: &str,
    ) -> Result<(Array2<f32>, Array1<i64>, Array1<i64>, Array1<i64>)> {
        let normalized_text = norm::normalize_text(text);

        let (phones, tones, mut word2ph) = self.jtalk.g2p(&normalized_text)?;
        let (phones, tones, lang_ids) = nlp::cleaned_text_to_sequence(phones, tones);

        let phones = utils::intersperse(&phones, 0);
        let tones = utils::intersperse(&tones, 0);
        let lang_ids = utils::intersperse(&lang_ids, 0);
        for i in 0..word2ph.len() {
            word2ph[i] *= 2;
        }
        word2ph[0] += 1;

        let tokenizer = jtalk::get_tokenizer()?;
        let (token_ids, attention_masks) = jtalk::tokenize(&normalized_text, &tokenizer)?;

        let bert_content = bert::predict(&self.bert, token_ids, attention_masks)?;

        assert!(
            word2ph.len() == normalized_text.chars().count() + 2,
            "{} {}",
            word2ph.len(),
            normalized_text.chars().count()
        );

        let mut phone_level_feature = vec![];
        for i in 0..word2ph.len() {
            let repeat_feature = {
                let (reps_rows, reps_cols) = (word2ph[i], 1);
                let arr_len = bert_content.slice(s![i, ..]).len();

                let mut results: Array2<f32> =
                    Array::zeros((reps_rows as usize, arr_len * reps_cols));

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
        let phone_level_feature = concatenate(
            Axis(0),
            &phone_level_feature
                .iter()
                .map(|x| x.view())
                .collect::<Vec<_>>(),
        )?;
        let bert_ori = phone_level_feature.t();
        Ok((
            bert_ori.to_owned(),
            phones.into(),
            tones.into(),
            lang_ids.into(),
        ))
    }

    pub fn get_style_vector(&self, style_id: i32, weight: f32) -> Result<Array1<f32>> {
        Ok(style::get_style_vector(
            self.style_vectors.clone(),
            style_id,
            weight,
        )?)
    }

    pub fn synthesize(
        &self,
        bert_ori: Array2<f32>,
        phones: Array1<i64>,
        tones: Array1<i64>,
        lang_ids: Array1<i64>,
        style_vector: Array1<f32>,
    ) -> Result<()> {
        model::synthesize(
            &self.vits2,
            bert_ori.to_owned(),
            phones,
            tones,
            lang_ids,
            style_vector,
        )?;
        Ok(())
    }
}
