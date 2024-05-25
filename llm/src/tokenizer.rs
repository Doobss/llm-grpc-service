use crate::batch_encoding::BatchEncoding;
use crate::model::ModelType;
use crate::{Error, Result};
use hf_hub::{api, api::sync::ApiRepo, Repo, RepoType};
use serde_json;
use std::path::PathBuf;
use tokenizers;

#[derive(Debug, Clone)]
pub struct TokenizerFiles {
    model: PathBuf,
    config: PathBuf,
    special_tokens: Option<PathBuf>,
}

impl TokenizerFiles {
    pub fn from_repo(repo: &ApiRepo) -> Result<Self> {
        let model = repo.get("tokenizer.json")?;
        let config = repo.get("tokenizer_config.json")?;
        let special_tokens = repo.get("special_tokens_map.json").ok();
        Ok(Self {
            model,
            config,
            special_tokens,
        })
    }
}

#[derive(Debug)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    padding: tokenizers::PaddingParams,
}

impl Tokenizer {
    pub fn encode_batch(&self, inputs: Vec<String>) -> Result<BatchEncoding> {
        let encodings = match self.inner.encode_batch(inputs, true) {
            Ok(batch) => Ok(batch),
            Err(error) => Err(Error::TokenizerError(error.to_string())),
        };
        let mut encodings = encodings?;
        tokenizers::pad_encodings(&mut encodings, &self.padding)?;
        BatchEncoding::from_encodings(encodings, &self.padding)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.inner.get_vocab(true).get(token_s).copied()
    }

    pub fn from_files(files: TokenizerFiles) -> Result<Self> {
        use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy};
        tracing::debug!("loading tokenizer config: {:?}", &files.config);
        let config: serde_json::Value = match files.special_tokens {
            Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
            None => serde_json::from_slice(&std::fs::read(files.config)?)?,
        };

        let pad_token: String = match config.get("pad_token") {
            Some(value) => value
                .as_str()
                .map(|slice| slice.to_owned())
                .expect("pad_token is not a string in the tokenizer config.json"),
            None => match config.get("bos_token") {
                Some(value) => value
                    .as_str()
                    .map(|slice| slice.to_owned())
                    .expect("bos_token is not a string in the tokenizer config.json"),
                None => "[PAD]".to_owned(),
            },
        };

        tracing::debug!("tokenizer config: {:?}", config);
        let tokenizer = tokenizers::Tokenizer::from_file(files.model)?;
        let pad_id = tokenizer.encode(pad_token.clone(), false)?.get_ids()[0];
        Ok(Self {
            inner: tokenizer,
            padding: PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                direction: PaddingDirection::Left,
                pad_to_multiple_of: None,
                pad_id,
                pad_type_id: pad_id,
                pad_token,
            },
        })
    }

    pub fn from_repo(repo: &ApiRepo) -> Result<Self> {
        let tokenizer_files = TokenizerFiles::from_repo(repo)?;
        Self::from_files(tokenizer_files)
    }

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
}
