extern crate tokio; // Should decople from tokio in future.

use super::{GenerationLogitsProcessor, GenerationReply};
use crate::{Prompt, PromptConfig};

pub type GenerationReplySender = tokio::sync::mpsc::Sender<GenerationReply>;

#[derive(Debug)]
pub struct GenerationRequest {
    pub id: String,
    pub content: String,
    pub generated: String,
    pub config: PromptConfig,
    pub reply_sender: GenerationReplySender,
    pub logit: GenerationLogitsProcessor,
}

impl GenerationRequest {
    pub fn from_prompt(prompt: Prompt, reply_sender: GenerationReplySender) -> Self {
        let Prompt {
            id,
            content,
            config,
        } = prompt;
        let logit = GenerationLogitsProcessor::from_prompt_config(&config);
        Self {
            id,
            content,
            config,
            reply_sender,
            logit,
            generated: String::new()
        }
    }
}
