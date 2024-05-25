use crate::Result;
use clap::{self, builder::Str};
use hf_hub::{api, Repo, RepoType};
// use tokenizers::parallelism::MaybeParallelRefIterator;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ModelType {
    #[value(name = "7b-v0.1")]
    Mistral7bV01026,
    #[value(name = "7b-v0.2")]
    Mistral7bV02,
    #[value(name = "7b-instruct-v0.1")]
    Mistral7bInstructV01,
    #[value(name = "7b-instruct-v0.2")]
    Mistral7bInstructV02,
}

impl ModelType {
    pub fn path(&self) -> String {
        match self {
            ModelType::Mistral7bV01 => "mistralai/Mistral-7B-v0.1".to_string(),
            ModelType::Mistral7bV02 => "mistralai/Mistral-7B-v0.2".to_string(),
            ModelType::Mistral7bInstructV01 => "mistralai/Mistral-7B-Instruct-v0.1".to_string(),
            ModelType::Mistral7bInstructV02 => "mistralai/Mistral-7B-Instruct-v0.2".to_string(),
        }
    }
}

pub struct Model {
    model_type: ModelType,
}

impl Model {
    pub fn new(model_type: ModelType) -> Self {
        Self { model_type }
    }

    pub fn load(&self) -> Result<()> {
        let api = api::sync::ApiBuilder::new()
            .with_cache_dir("./.cache/huggingface".into())
            .with_token(Some("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".to_owned()))
            .build()?;
        let model_id = self.model_type.path();
        tracing::debug!("loading model_id: {model_id}");
        let repo = api.repo(Repo::with_revision(
            model_id,
            RepoType::Model,
            "main".to_owned(),
        ));
        let repo_info = repo.info()?;
        tracing::debug!("loaded repo: {:?}", repo_info);
        let files: Result<Vec<String>> = repo_info
            .siblings
            .into_iter()
            .map(|sibling| {
                tracing::debug!("downloading: {}", &sibling.rfilename);
                repo.download(&sibling.rfilename)?;
                tracing::debug!("downloaded: {}", &sibling.rfilename);
                Ok(sibling.rfilename.to_owned())
            })
            .collect();
        tracing::info!("Model files {:?}", files);
        Ok(())
    }
}
