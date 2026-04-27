use reqwest::{header, Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::{env, fmt};

#[derive(Debug)]
pub enum AgentError {
    MissingApiKey,
    Http(reqwest::Error),
    Upstream(StatusCode, String),
    EmptyResponse,
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::MissingApiKey => write!(f, "missing OPENAI_API_KEY"),
            AgentError::Http(err) => write!(f, "http request failed: {}", err),
            AgentError::Upstream(status, body) => {
                write!(f, "upstream request failed with {}: {}", status, body)
            }
            AgentError::EmptyResponse => write!(f, "upstream response had no assistant message"),
        }
    }
}

impl std::error::Error for AgentError {}

#[derive(Clone)]
pub struct OpenAiCompatibleClient {
    http: Client,
    base_url: String,
    api_key: String,
}

impl OpenAiCompatibleClient {
    pub fn from_env() -> Result<Self, AgentError> {
        let api_key = env::var("OPENAI_API_KEY").map_err(|_| AgentError::MissingApiKey)?;
        let base_url = env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

        Ok(Self {
            http: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
        })
    }

    pub async fn chat_completion(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
    ) -> Result<String, AgentError> {
        let url = format!("{}/chat/completions", self.base_url);
        let payload = ChatCompletionRequest {
            model: model.to_string(),
            messages,
        };

        let response = self
            .http
            .post(url)
            .header(header::AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(header::CONTENT_TYPE, "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(AgentError::Http)?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_else(|_| "<failed to read body>".to_string());
            return Err(AgentError::Upstream(status, body));
        }

        let parsed: ChatCompletionResponse = response.json().await.map_err(AgentError::Http)?;
        parsed
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message.content)
            .filter(|content| !content.trim().is_empty())
            .ok_or(AgentError::EmptyResponse)
    }
}

#[derive(Clone)]
pub struct Agent {
    client: OpenAiCompatibleClient,
    model: String,
    system_prompt: Option<String>,
}

impl Agent {
    pub fn from_env() -> Result<Self, AgentError> {
        let client = OpenAiCompatibleClient::from_env()?;
        let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".to_string());
        let system_prompt = env::var("AGENT_SYSTEM_PROMPT").ok();

        Ok(Self {
            client,
            model,
            system_prompt,
        })
    }

    pub async fn respond(&self, user_prompt: &str) -> Result<String, AgentError> {
        let mut messages = Vec::new();
        if let Some(prompt) = &self.system_prompt {
            messages.push(ChatMessage::system(prompt));
        }
        messages.push(ChatMessage::user(user_prompt));
        self.client.chat_completion(&self.model, messages).await
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}
