use crate::{utils, Result};
use hf_hub::api::sync::ApiRepo;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct TokenizerFiles {
    pub model: PathBuf,
    pub config: PathBuf,
    pub special_tokens: Option<PathBuf>,
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

    pub fn load_file<T>(file_path: PathBuf) -> Result<T>
    where
        T: for<'a> serde::Deserialize<'a>,
    {
        utils::load_file::<T>(file_path)
    }

    pub fn load_config<T>(&self) -> Result<T>
    where
        T: for<'a> serde::Deserialize<'a>,
    {
        TokenizerFiles::load_file(self.config.clone())
    }

    pub fn load_model<T>(&self) -> Result<T>
    where
        T: for<'a> serde::Deserialize<'a>,
    {
        TokenizerFiles::load_file(self.model.clone())
    }

    pub fn load_special_tokens<T>(&self) -> Result<Option<T>>
    where
        T: for<'a> serde::Deserialize<'a>,
    {
        if let Some(config_path) = self.special_tokens.clone() {
            Ok(Some(TokenizerFiles::load_file(config_path)?))
        } else {
            Ok(None)
        }
    }
}
