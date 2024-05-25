use crate::model::ModelType;
use crate::Result;
use hf_hub::{api, api::sync::ApiRepo, Repo, RepoType};
use std::path::PathBuf;
use tokenizers;

#[derive(Debug, Clone)]
pub struct TokenizerFiles {
    model: PathBuf,
    config: Option<PathBuf>,
}

impl TokenizerFiles {
    pub fn from_repo(repo: &ApiRepo) -> Result<Self> {
        let model_path = repo.get("tokenizer.json")?;
        let config_json: Option<PathBuf> = repo.get("tokenizer_config.json").ok();
        Ok(Self {
            model: model_path,
            config: config_json,
        })
    }
}

#[derive(Debug)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn load(model_type: ModelType) -> Result<Self> {
        let api = api::sync::ApiBuilder::new()
            .with_cache_dir("./.cache/huggingface".into())
            .with_token(Some("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".to_owned()))
            .build()?;
        let model_id = model_type.path();
        tracing::debug!("loading model_id: {model_id}");
        let repo = api.repo(Repo::with_revision(
            model_id,
            RepoType::Model,
            "main".to_owned(),
        ));
        Self::from_repo(&repo)
    }

    pub fn from_files(files: TokenizerFiles) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(files.model)?;
        Ok(Self { inner: tokenizer })
    }

    pub fn from_repo(repo: &ApiRepo) -> Result<Self> {
        let tokenizer_files = TokenizerFiles::from_repo(repo)?;
        Self::from_files(tokenizer_files)
    }
}
