use crate::utils;
use crate::v1::prompt;
use crate::v1::prompt::prompt_server;
use crate::EndpointResult;
use tonic::{Request, Response, Status};

#[derive(Debug)]
pub struct PromptServer {
    model_id: String,
    tokenizer: llm::Tokenizer,
    template: Option<llm::ChatTemplate>,
}

impl PromptServer {
    pub fn new(model_type: llm::ModelType) -> crate::Result<Self> {
        let tokenizer = llm::Tokenizer::load(model_type)?;
        if tokenizer.template.is_none() {
            tracing::debug!(
                "Prompt service: no chat template found for model: {:?}",
                &model_type
            );
        } else {
            tracing::debug!(
                "Prompt service: chat template found for model: {:?}",
                &model_type
            );
        }
        Ok(Self {
            model_id: model_type.id(),
            template: tokenizer.template.clone(),
            tokenizer,
        })
    }
}

#[tonic::async_trait]
impl prompt_server::Prompt for PromptServer {
    async fn apply_template(
        &self,
        req: Request<prompt::ApplyTemplateRequest>,
    ) -> EndpointResult<prompt::ApplyTemplateReply> {
        let prompt::ApplyTemplateRequest {
            id,
            messages,
            custom_template,
        } = req.into_inner();
        let template: Option<llm::ChatTemplate> =
            if let Some(custom_string) = utils::default_to_optional(custom_template) {
                tracing::debug!(
                    "Prompt apply template using custom template: {:?}",
                    &custom_string
                );
                Some(self.tokenizer.new_chat_template(custom_string))
            } else {
                self.template.clone()
            };
        if let Some(template) = template {
            let raw_template = template.get_template();
            let messages = messages.into_iter().map(|message| message.into()).collect();
            let content = template
                .apply(messages)
                .map_err(|error| Status::internal(error.to_string()))?;
            let response = prompt::ApplyTemplateReply {
                id,
                content,
                template: raw_template,
            };
            Ok(Response::new(response))
        } else {
            Err(Status::not_found(format!(
                "Could not load a template for model: {} and no custom template was given.",
                self.model_id.clone()
            )))
        }
    }

    async fn get_template(
        &self,
        _req: Request<prompt::GetTemplateRequest>,
    ) -> EndpointResult<prompt::GetTemplateReply> {
        if let Some(template) = self.template.as_ref() {
            let response = prompt::GetTemplateReply {
                model_id: self.model_id.clone(),
                template: template.get_template(),
            };
            Ok(Response::new(response))
        } else {
            Err(Status::not_found(format!(
                "Could not load a template for model: {}",
                self.model_id.clone()
            )))
        }
    }
}

impl From<prompt::Message> for llm::ChatMessage {
    fn from(value: prompt::Message) -> Self {
        let prompt::Message { role, content, .. } = value;
        Self { role, content }
    }
}

pub fn service(model_type: llm::ModelType) -> prompt_server::PromptServer<PromptServer> {
    tracing::info!("Adding prompt service");
    let server = PromptServer::new(model_type).expect("Error loading prompt service");
    prompt_server::PromptServer::new(server)
}
