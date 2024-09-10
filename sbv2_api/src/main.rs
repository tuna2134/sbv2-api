use axum::{
    extract::State,
    http::header::CONTENT_TYPE,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use sbv2_core::tts::TTSModel;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::Mutex;

mod error;
use crate::error::AppResult;

#[derive(Deserialize)]
struct SynthesizeRequest {
    text: String,
}

async fn synthesize(
    State(state): State<Arc<AppState>>,
    Json(SynthesizeRequest { text }): Json<SynthesizeRequest>,
) -> AppResult<impl IntoResponse> {
    let buffer = {
        let mut tts_model = state.tts_model.lock().await;
        let tts_model = if let Some(tts_model) = &*tts_model {
            tts_model
        } else {
            *tts_model = Some(TTSModel::new(
                "models/debert.onnx",
                "models/model_opt.onnx",
                "models/style_vectors.json",
            )?);
            &*tts_model.as_ref().unwrap()
        };
        let (bert_ori, phones, tones, lang_ids) = tts_model.parse_text(&text)?;
        let style_vector = tts_model.get_style_vector(0, 1.0)?;
        tts_model.synthesize(bert_ori.to_owned(), phones, tones, lang_ids, style_vector)?
    };
    Ok(([(CONTENT_TYPE, "audio/wav")], buffer))
}

struct AppState {
    tts_model: Arc<Mutex<Option<TTSModel>>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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
