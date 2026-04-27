use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentChatRequest {
    pub messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentChatResponse {
    pub content: String,
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("missing OPENAI_API_KEY")]
    MissingApiKey,
    #[error("request to OpenAI-compatible API failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("invalid response from OpenAI-compatible API")]
    InvalidResponse,
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleClient {
    http: Client,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl OpenAiCompatibleClient {
    pub fn from_env() -> Result<Self, AgentError> {
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| AgentError::MissingApiKey)?;
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

        Ok(Self {
            http: Client::new(),
            base_url,
            api_key,
            model,
        })
    }

    pub async fn chat(&self, messages: Vec<ChatMessage>) -> Result<AgentChatResponse, AgentError> {
        let endpoint = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let payload = OpenAiChatRequest {
            model: self.model.clone(),
            messages,
            temperature: 0.2,
        };

        let resp = self
            .http
            .post(endpoint)
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()
            .await?
            .error_for_status()?;

        let data: OpenAiChatResponse = resp.json().await?;
        let content = data
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .filter(|s| !s.trim().is_empty())
            .ok_or(AgentError::InvalidResponse)?;

        Ok(AgentChatResponse { content })
    }
}

#[derive(Debug, Clone)]
pub struct Agent {
    client: OpenAiCompatibleClient,
}

impl Agent {
    pub fn from_env() -> Result<Self, AgentError> {
        let client = OpenAiCompatibleClient::from_env()?;
        Ok(Self { client })
    }

    pub async fn run(&self, request: AgentChatRequest) -> Result<AgentChatResponse, AgentError> {
        self.client.chat(request.messages).await
    }
}

#[derive(Serialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
}

#[derive(Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiAssistantMessage,
}

#[derive(Deserialize)]
struct OpenAiAssistantMessage {
    content: String,
}

