use axum::{
    extract::State,
    http::header::CONTENT_TYPE,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use sbv2_core::tts::{SynthesizeOptions, TTSModelHolder};
use serde::Deserialize;
use std::env;
use std::sync::Arc;
use tokio::fs;
use tokio::sync::Mutex;
use utoipa::{OpenApi, ToSchema};
use utoipa_scalar::{Scalar, Servable};

mod error;
use crate::error::AppResult;

#[derive(OpenApi)]
#[openapi(paths(models, synthesize), components(schemas(SynthesizeRequest)))]
struct ApiDoc;

#[utoipa::path(
    get,
    path = "/models",
    responses(
        (status = 200, description = "Return model list", body = Vec<String>),
    )
)]
async fn models(State(state): State<AppState>) -> AppResult<impl IntoResponse> {
    Ok(Json(state.tts_model.lock().await.models()))
}

fn sdp_default() -> f32 {
    0.0
}

fn length_default() -> f32 {
    1.0
}

#[derive(Deserialize, ToSchema)]
struct SynthesizeRequest {
    text: String,
    ident: String,
    #[serde(default = "sdp_default")]
    sdp_ratio: f32,
    #[serde(default = "length_default")]
    length_scale: f32,
}

#[utoipa::path(
    post,
    path = "/synthesize",
    request_body = SynthesizeRequest,
    responses(
        (status = 200, description = "Return audio/wav", body = Vec<u8>, content_type = "audio/wav")
    )
)]
async fn synthesize(
    State(state): State<AppState>,
    Json(SynthesizeRequest {
        text,
        ident,
        sdp_ratio,
        length_scale,
    }): Json<SynthesizeRequest>,
) -> AppResult<impl IntoResponse> {
    log::debug!("processing request: text={text}, ident={ident}, sdp_ratio={sdp_ratio}, length_scale={length_scale}");
    let buffer = {
        let mut tts_model = state.tts_model.lock().await;
        tts_model.easy_synthesize(
            &ident,
            &text,
            0,
            SynthesizeOptions {
                sdp_ratio,
                length_scale,
                ..Default::default()
            },
        )?
    };
    Ok(([(CONTENT_TYPE, "audio/wav")], buffer))
}

#[derive(Clone)]
struct AppState {
    tts_model: Arc<Mutex<TTSModelHolder>>,
}

impl AppState {
    pub async fn new() -> anyhow::Result<Self> {
        let mut tts_model = TTSModelHolder::new(
            &fs::read(env::var("BERT_MODEL_PATH")?).await?,
            &fs::read(env::var("TOKENIZER_PATH")?).await?,
            env::var("HOLDER_MAX_LOADED_MODElS")
                .ok()
                .and_then(|x| x.parse().ok()),
        )?;
        let models = env::var("MODELS_PATH").unwrap_or("models".to_string());
        let mut f = fs::read_dir(&models).await?;
        let mut entries = vec![];
        while let Ok(Some(e)) = f.next_entry().await {
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".onnx") && name.starts_with("model_") {
                let name_len = name.len();
                let name = name.chars();
                entries.push(
                    name.collect::<Vec<_>>()[6..name_len - 5]
                        .iter()
                        .collect::<String>(),
                );
            } else if name.ends_with(".sbv2") {
                let entry = &name[..name.len() - 5];
                log::info!("Try loading: {entry}");
                let sbv2_bytes = match fs::read(format!("{models}/{entry}.sbv2")).await {
                    Ok(b) => b,
                    Err(e) => {
                        log::warn!("Error loading sbv2_bytes from file {entry}: {e}");
                        continue;
                    }
                };
                if let Err(e) = tts_model.load_sbv2file(entry, sbv2_bytes) {
                    log::warn!("Error loading {entry}: {e}");
                };
                log::info!("Loaded: {entry}");
            }
        }
        for entry in entries {
            log::info!("Try loading: {entry}");
            let style_vectors_bytes =
                match fs::read(format!("{models}/style_vectors_{entry}.json")).await {
                    Ok(b) => b,
                    Err(e) => {
                        log::warn!("Error loading style_vectors_bytes from file {entry}: {e}");
                        continue;
                    }
                };
            let vits2_bytes = match fs::read(format!("{models}/model_{entry}.onnx")).await {
                Ok(b) => b,
                Err(e) => {
                    log::warn!("Error loading vits2_bytes from file {entry}: {e}");
                    continue;
                }
            };
            if let Err(e) = tts_model.load(&entry, style_vectors_bytes, vits2_bytes) {
                log::warn!("Error loading {entry}: {e}");
            };
            log::info!("Loaded: {entry}");
        }
        Ok(Self {
            tts_model: Arc::new(Mutex::new(tts_model)),
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv_override().ok();
    env_logger::init();
    let app = Router::new()
        .route("/", get(|| async { "Hello, World!" }))
        .route("/synthesize", post(synthesize))
        .route("/models", get(models))
        .with_state(AppState::new().await?)
        .merge(Scalar::with_url("/docs", ApiDoc::openapi()));
    let addr = env::var("ADDR").unwrap_or("0.0.0.0:3000".to_string());
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    log::info!("Listening on {addr}");
    axum::serve(listener, app).await?;

    Ok(())
}
