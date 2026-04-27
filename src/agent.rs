use async_openai::{
    config::OpenAIConfig,
    error::OpenAIError,
    types::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
        CreateChatCompletionRequest, CreateChatCompletionResponse,
    },
    Client,
};
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
    OpenAi(#[from] OpenAIError),
    #[error("invalid response from OpenAI-compatible API")]
    InvalidResponse,
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleClient {
    client: Client<OpenAIConfig>,
    pub model: String,
}

impl OpenAiCompatibleClient {
    pub fn from_env() -> Result<Self, AgentError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| AgentError::MissingApiKey)?;
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
        let base_url = std::env::var("OPENAI_BASE_URL").ok();

        let mut config = OpenAIConfig::new().with_api_key(api_key);
        if let Some(url) = base_url {
            config = config.with_api_base(url);
        }

        Ok(Self {
            client: Client::with_config(config),
            model,
        })
    }

    pub async fn chat(&self, messages: Vec<ChatMessage>) -> Result<AgentChatResponse, AgentError> {
        let payload = CreateChatCompletionRequest {
            model: self.model.clone(),
            messages: to_openai_messages(messages),
            temperature: Some(0.2),
            ..Default::default()
        };

        let data = self.client.chat().create(payload).await?;
        let content = extract_content(data)?;

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

fn to_openai_messages(messages: Vec<ChatMessage>) -> Vec<ChatCompletionRequestMessage> {
    messages
        .into_iter()
        .map(|m| match m.role {
            ChatRole::System => {
                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: m.content.into(),
                    ..Default::default()
                })
            }
            ChatRole::User => ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: m.content.into(),
                ..Default::default()
            }),
            ChatRole::Assistant => {
                ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                    content: Some(m.content),
                    ..Default::default()
                })
            }
        })
        .collect()
}

fn extract_content(response: CreateChatCompletionResponse) -> Result<String, AgentError> {
    response
        .choices
        .into_iter()
        .filter_map(|choice| choice.message.content)
        .find(|content| !content.trim().is_empty())
        .ok_or(AgentError::InvalidResponse)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_env_var<F>(key: &str, value: Option<&str>, f: F)
    where
        F: FnOnce(),
    {
        let original = std::env::var(key).ok();
        match value {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }

        f();

        match original {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    #[test]
    fn from_env_requires_api_key() {
        with_env_var("OPENAI_API_KEY", None, || {
            let result = OpenAiCompatibleClient::from_env();
            assert!(matches!(result, Err(AgentError::MissingApiKey)));
        });
    }

    #[test]
    fn maps_roles_to_openai_messages() {
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are concise".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Hello".to_string(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "Hi".to_string(),
            },
        ];

        let mapped = to_openai_messages(messages);
        assert_eq!(mapped.len(), 3);
        assert!(matches!(mapped[0], ChatCompletionRequestMessage::System(_)));
        assert!(matches!(mapped[1], ChatCompletionRequestMessage::User(_)));
        assert!(matches!(mapped[2], ChatCompletionRequestMessage::Assistant(_)));
    }

    #[test]
    fn extracts_first_non_empty_content() {
        let response: CreateChatCompletionResponse = serde_json::from_str(
            r#"{
                "id":"chatcmpl-test",
                "object":"chat.completion",
                "created":1710000000,
                "model":"gpt-4o-mini",
                "choices":[
                    {
                        "index":0,
                        "message":{"role":"assistant","content":""},
                        "finish_reason":"stop"
                    },
                    {
                        "index":1,
                        "message":{"role":"assistant","content":"answer"},
                        "finish_reason":"stop"
                    }
                ],
                "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
            }"#,
        )
        .expect("response should deserialize");

        let content = extract_content(response).expect("content should parse");
        assert_eq!(content, "answer");
    }

    #[test]
    fn returns_invalid_response_when_no_content() {
        let response: CreateChatCompletionResponse = serde_json::from_str(
            r#"{
                "id":"chatcmpl-test",
                "object":"chat.completion",
                "created":1710000000,
                "model":"gpt-4o-mini",
                "choices":[],
                "usage":{"prompt_tokens":1,"completion_tokens":0,"total_tokens":1}
            }"#,
        )
        .expect("response should deserialize");

        let err = extract_content(response).expect_err("should fail");
        assert!(matches!(err, AgentError::InvalidResponse));
    }
}
