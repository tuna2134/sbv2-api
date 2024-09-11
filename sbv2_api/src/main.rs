use axum::{
    extract::State,
    http::header::CONTENT_TYPE,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use sbv2_core::tts::TTSModelHolder;
use serde::Deserialize;
use std::env;
use std::sync::Arc;
use tokio::fs;
use tokio::sync::Mutex;

mod error;
use crate::error::AppResult;

#[derive(Deserialize)]
struct SynthesizeRequest {
    text: String,
    ident: String,
}

async fn synthesize(
    State(state): State<Arc<AppState>>,
    Json(SynthesizeRequest { text, ident }): Json<SynthesizeRequest>,
) -> AppResult<impl IntoResponse> {
    let buffer = {
        let mut tts_model = state.tts_model.lock().await;
        let tts_model = if let Some(tts_model) = &*tts_model {
            tts_model
        } else {
            let mut tts_holder = TTSModelHolder::new(
                &fs::read(env::var("BERT_MODEL_PATH")?).await?,
                &fs::read(env::var("TOKENIZER_PATH")?).await?,
            )?;
            tts_holder.load(
                "tsukuyomi",
                fs::read(env::var("STYLE_VECTORS_PATH")?).await?,
                fs::read(env::var("MODEL_PATH")?).await?,
            )?;
            *tts_model = Some(tts_holder);
            tts_model.as_ref().unwrap()
        };
        let (bert_ori, phones, tones, lang_ids) = tts_model.parse_text(&text)?;
        let style_vector = tts_model.get_style_vector(&ident, 0, 1.0)?;
        tts_model.synthesize(
            ident,
            bert_ori.to_owned(),
            phones,
            tones,
            lang_ids,
            style_vector,
        )?
    };
    Ok(([(CONTENT_TYPE, "audio/wav")], buffer))
}

struct AppState {
    tts_model: Arc<Mutex<Option<TTSModelHolder>>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let app = Router::new()
        .route("/", get(|| async { "Hello, World!" }))
        .route("/synthesize", post(synthesize))
        .with_state(Arc::new(AppState {
            tts_model: Arc::new(Mutex::new(None)),
        }));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;

    Ok(())
}
