extern crate tokio; // Should decople from tokio in future.

use super::{GenerationLogitsProcessor, GenerationResult};
use crate::{Prompt, PromptConfig};

pub type GenerationResultSender = tokio::sync::mpsc::Sender<GenerationResult>;

#[derive(Debug)]
pub struct GenerationRequest {
    pub id: String,
    pub content: String,
    pub generated: String,
    pub config: PromptConfig,
    pub reply_sender: GenerationResultSender,
    // logit: GenerationLogitsProcessor,
}

impl GenerationRequest {
    pub fn from_prompt(prompt: Prompt, reply_sender: GenerationResultSender) -> Self {
        let Prompt {
            id,
            content,
            config,
        } = prompt;
        // let logit = GenerationLogitsProcessor::from_prompt_config(&config);
        Self {
            id,
            content,
            config,
            reply_sender,
            // logit,
            generated: String::new(),
        }
    }
}

impl GenerationRequest {
    pub fn sender(&self) -> &GenerationResultSender {
        &self.reply_sender
    }

    pub fn logit_processor(&self) -> GenerationLogitsProcessor {
        GenerationLogitsProcessor::from_prompt_config(&self.config)
    }
}
