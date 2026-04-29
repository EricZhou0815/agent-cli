use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod agents;
use agents::{Agent, AgentChatRequest, AgentChatResponse};

#[derive(Serialize, Deserialize)]
struct Message {
    message: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

async fn echo(Json(payload): Json<Message>) -> Json<Message> {
    Json(payload)
}

#[derive(Clone)]
struct AppState {
    agent: Option<Arc<Agent>>,
}

async fn agent_chat(
    State(state): State<AppState>,
    Json(payload): Json<AgentChatRequest>,
) -> Result<Json<AgentChatResponse>, (StatusCode, String)> {
    let agent = state
        .agent
        .as_ref()
        .ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "agent unavailable: missing OPENAI_API_KEY".to_string(),
        ))?;

    let response = agent
        .run(payload)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;

    Ok(Json(response))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rust_api=debug,tower_http=debug,axum=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let agent = match Agent::from_env() {
        Ok(agent) => Some(Arc::new(agent)),
        Err(e) => {
            tracing::warn!("agent disabled: {}", e);
            None
        }
    };
    let state = AppState { agent };

    let app = Router::new()
        .route("/health", get(health))
        .route("/echo", post(echo))
        .route("/agent/chat", post(agent_chat))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .unwrap();

    tracing::info!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}
