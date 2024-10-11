use super::ChatMessage;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub(crate) struct ChatTemplateInputs<'a> {
    pub messages: Vec<ChatMessage>,
    pub bos_token: Option<&'a str>,
    pub eos_token: Option<&'a str>,
    pub add_generation_prompt: bool,
}
