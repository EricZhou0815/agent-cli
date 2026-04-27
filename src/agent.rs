use async_openai::{
    config::OpenAIConfig,
    error::OpenAIError,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
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
    #[error("failed to build OpenAI request: {0}")]
    RequestBuild(String),
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleClient {
    client: Client<OpenAIConfig>,
    pub model: String,
}

impl OpenAiCompatibleClient {
    pub fn from_env() -> Result<Self, AgentError> {
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| AgentError::MissingApiKey)?;
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

        let config = OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key(api_key);

        Ok(Self {
            client: Client::with_config(config),
            model,
        })
    }

    pub async fn chat(&self, messages: Vec<ChatMessage>) -> Result<AgentChatResponse, AgentError> {
        let mapped_messages = messages
            .into_iter()
            .map(map_chat_message)
            .collect::<Result<Vec<_>, _>>()?;

        let request = CreateChatCompletionRequestArgs::default()
            .model(self.model.clone())
            .messages(mapped_messages)
            .temperature(0.2)
            .build()
            .map_err(|e| AgentError::RequestBuild(e.to_string()))?;

        let response = self.client.chat().create(request).await?;

        let content = response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .filter(|s| !s.trim().is_empty())
            .ok_or(AgentError::InvalidResponse)?;

        Ok(AgentChatResponse { content })
    }
}

fn map_chat_message(message: ChatMessage) -> Result<ChatCompletionRequestMessage, AgentError> {
    match message.role {
        ChatRole::System => ChatCompletionRequestSystemMessageArgs::default()
            .content(message.content)
            .build()
            .map(ChatCompletionRequestMessage::System)
            .map_err(|e| AgentError::RequestBuild(e.to_string())),
        ChatRole::User => ChatCompletionRequestUserMessageArgs::default()
            .content(message.content)
            .build()
            .map(ChatCompletionRequestMessage::User)
            .map_err(|e| AgentError::RequestBuild(e.to_string())),
        ChatRole::Assistant => ChatCompletionRequestAssistantMessageArgs::default()
            .content(message.content)
            .build()
            .map(ChatCompletionRequestMessage::Assistant)
            .map_err(|e| AgentError::RequestBuild(e.to_string())),
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

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::Method::POST;
    use httpmock::MockServer;
    use serial_test::serial;

    struct EnvGuard {
        original: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        fn new(keys: &[&'static str]) -> Self {
            let original = keys
                .iter()
                .map(|k| (*k, std::env::var(k).ok()))
                .collect::<Vec<_>>();

            Self { original }
        }

        fn set(&self, key: &'static str, value: &str) {
            std::env::set_var(key, value);
        }

        fn unset(&self, key: &'static str) {
            std::env::remove_var(key);
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in &self.original {
                match value {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
        }
    }

    #[test]
    #[serial]
    fn from_env_requires_api_key() {
        let env = EnvGuard::new(&["OPENAI_API_KEY"]);
        env.unset("OPENAI_API_KEY");

        let result = OpenAiCompatibleClient::from_env();
        assert!(matches!(result, Err(AgentError::MissingApiKey)));
    }

    #[tokio::test]
    #[serial]
    async fn chat_returns_first_choice_content() {
        let server = MockServer::start();

        let mock = server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    r#"{
                        "id":"chatcmpl-test",
                        "object":"chat.completion",
                        "created":1710000000,
                        "model":"gpt-4o-mini",
                        "choices":[
                            {
                                "index":0,
                                "message":{"role":"assistant","content":"hello from mock"},
                                "finish_reason":"stop"
                            }
                        ],
                        "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
                    }"#,
                );
        });

        let env = EnvGuard::new(&["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]);
        env.set("OPENAI_API_KEY", "test-key");
        env.set("OPENAI_BASE_URL", &server.base_url());
        env.set("OPENAI_MODEL", "gpt-4o-mini");

        let client = OpenAiCompatibleClient::from_env().expect("client setup");
        let response = client
            .chat(vec![ChatMessage {
                role: ChatRole::User,
                content: "Say hi".to_string(),
            }])
            .await
            .expect("chat response");

        assert_eq!(response.content, "hello from mock");
        mock.assert();
    }

    #[tokio::test]
    #[serial]
    async fn chat_returns_invalid_response_on_empty_choices() {
        let server = MockServer::start();

        let _mock = server.mock(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    r#"{
                        "id":"chatcmpl-test",
                        "object":"chat.completion",
                        "created":1710000000,
                        "model":"gpt-4o-mini",
                        "choices":[],
                        "usage":{"prompt_tokens":1,"completion_tokens":0,"total_tokens":1}
                    }"#,
                );
        });

        let env = EnvGuard::new(&["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]);
        env.set("OPENAI_API_KEY", "test-key");
        env.set("OPENAI_BASE_URL", &server.base_url());
        env.set("OPENAI_MODEL", "gpt-4o-mini");

        let client = OpenAiCompatibleClient::from_env().expect("client setup");
        let result = client
            .chat(vec![ChatMessage {
                role: ChatRole::User,
                content: "Say hi".to_string(),
            }])
            .await;

        assert!(matches!(result, Err(AgentError::InvalidResponse)));
    }
}
