use super::prompt_config::PromptConfig;
use uuid;

#[derive(Debug)]
pub struct Prompt {
    pub id: String,
    pub content: String,
    pub config: PromptConfig,
}

impl Prompt {
    pub fn gen_id() -> String {
        uuid::Uuid::new_v4().to_string()
    }
}

impl From<String> for Prompt {
    fn from(content: String) -> Self {
        Self {
            id: Prompt::gen_id(),
            content,
            config: PromptConfig::default(),
        }
    }
}

impl From<&'static str> for Prompt {
    fn from(content: &str) -> Self {
        Self {
            id: Prompt::gen_id(),
            content: content.to_owned(),
            config: PromptConfig::default(),
        }
    }
}
