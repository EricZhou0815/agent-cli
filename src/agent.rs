use async_openai::{
    config::OpenAIConfig,
    error::OpenAIError,
    types::{
        ChatCompletionAssistantMessage, ChatCompletionAssistantMessageContent,
        ChatCompletionRequestMessage, ChatCompletionSystemMessage,
        ChatCompletionSystemMessageContent, ChatCompletionUserMessage,
        ChatCompletionUserMessageContent, CreateChatCompletionRequest,
        CreateChatCompletionResponse,
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
    #[error("request to OpenAI API failed: {0}")]
    OpenAi(#[from] OpenAIError),
    #[error("invalid response from OpenAI API")]
    InvalidResponse,
    #[error("failed to build OpenAI request: {0}")]
    RequestBuild(String),
}

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    client: Client<OpenAIConfig>,
    pub model: String,
}

impl OpenAiClient {
    pub fn from_env() -> Result<Self, AgentError> {
        let base_url = std::env::var("OPENAI_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| AgentError::MissingApiKey)?;
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

        Self::from_parts(base_url, api_key, model)
    }

    fn from_parts(base_url: String, api_key: String, model: String) -> Result<Self, AgentError> {
        if api_key.trim().is_empty() {
            return Err(AgentError::MissingApiKey);
        }

        let config = OpenAIConfig::new().with_api_key(api_key).with_api_base(base_url);
        Ok(Self {
            client: Client::with_config(config),
            model,
        })
    }

    pub async fn chat(&self, messages: Vec<ChatMessage>) -> Result<AgentChatResponse, AgentError> {
        let request_messages = messages
            .into_iter()
            .map(to_openai_chat_message)
            .collect::<Result<Vec<_>, _>>()?;

        let request = CreateChatCompletionRequest {
            model: self.model.clone(),
            messages: request_messages,
            temperature: Some(0.2),
            ..Default::default()
        };

        let data = self.client.chat().create(request).await?;
        extract_response_content(data)
    }
}

#[derive(Debug, Clone)]
pub struct Agent {
    client: OpenAiClient,
}

impl Agent {
    pub fn from_env() -> Result<Self, AgentError> {
        let client = OpenAiClient::from_env()?;
        Ok(Self { client })
    }

    pub async fn run(&self, request: AgentChatRequest) -> Result<AgentChatResponse, AgentError> {
        self.client.chat(request.messages).await
    }
}

fn to_openai_chat_message(message: ChatMessage) -> Result<ChatCompletionRequestMessage, AgentError> {
    match message.role {
        ChatRole::System => Ok(ChatCompletionRequestMessage::System(
            ChatCompletionSystemMessage {
                content: ChatCompletionSystemMessageContent::Text(message.content),
                ..Default::default()
            },
        )),
        ChatRole::User => Ok(ChatCompletionRequestMessage::User(ChatCompletionUserMessage {
            content: ChatCompletionUserMessageContent::Text(message.content),
            ..Default::default()
        })),
        ChatRole::Assistant => Ok(ChatCompletionRequestMessage::Assistant(
            ChatCompletionAssistantMessage {
                content: Some(ChatCompletionAssistantMessageContent::Text(message.content)),
                ..Default::default()
            },
        )),
    }
}

fn extract_response_content(data: CreateChatCompletionResponse) -> Result<AgentChatResponse, AgentError> {
    extract_first_non_empty_content(
        data.choices
            .into_iter()
            .map(|choice| choice.message.content),
    )
}

fn extract_first_non_empty_content<I>(contents: I) -> Result<AgentChatResponse, AgentError>
where
    I: IntoIterator<Item = Option<ChatCompletionAssistantMessageContent>>,
{
    let content = contents
        .into_iter()
        .next()
        .flatten()
        .and_then(|c| match c {
            ChatCompletionAssistantMessageContent::Text(s) if !s.trim().is_empty() => Some(s),
            _ => None,
        })
        .ok_or(AgentError::InvalidResponse)?;

    Ok(AgentChatResponse { content })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_parts_requires_api_key() {
        let result = OpenAiClient::from_parts(
            "https://api.openai.com/v1".to_string(),
            " ".to_string(),
            "gpt-4o-mini".to_string(),
        );
        assert!(matches!(result, Err(AgentError::MissingApiKey)));
    }

    #[test]
    fn maps_user_message_to_openai_format() {
        let msg = ChatMessage {
            role: ChatRole::User,
            content: "hello".to_string(),
        };
        let mapped = to_openai_chat_message(msg).expect("should map");
        match mapped {
            ChatCompletionRequestMessage::User(_) => {}
            _ => panic!("expected user message"),
        }
    }

    #[test]
    fn extracts_first_choice_content() {
        let content = ChatCompletionAssistantMessageContent::Text("answer".to_string());
        let out = extract_first_non_empty_content(vec![Some(content)]).expect("content expected");
        assert_eq!(out.content, "answer");
    }

    #[test]
    fn fails_when_response_has_no_text() {
        let out = extract_first_non_empty_content(vec![None]);
        assert!(matches!(out, Err(AgentError::InvalidResponse)));
    }

    #[test]
    fn fails_when_response_has_blank_text() {
        let content = ChatCompletionAssistantMessageContent::Text("   ".to_string());
        let out = extract_first_non_empty_content(vec![Some(content)]);
        assert!(matches!(out, Err(AgentError::InvalidResponse)));
    }
}
