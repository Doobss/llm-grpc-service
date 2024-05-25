use crate::{Error, Result};
use candle_examples::hub_load_safetensors;
use candle_nn::VarBuilder;
use candle_transformers::models;
use clap;
use hf_hub::{api, api::sync::ApiRepo, Repo, RepoType};
use std::path::PathBuf;
// use tokenizers::parallelism::MaybeParallelRefIterator;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ModelType {
    #[value(name = "7b-v0.1")]
    Mistral7bV01,
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

#[derive(Debug)]
pub struct Model {
    inner: models::mistral::Model,
    device: candle_core::Device,
    dtype: candle_core::DType,
}

impl Model {
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

    pub fn from_files(files: ModelFiles) -> Result<Self> {
        let config = serde_json::from_slice(&std::fs::read(files.config)?)?;
        let device = Model::init_device()?;
        let dtype = Model::init_dtype()?;
        let vars = unsafe { VarBuilder::from_mmaped_safetensors(&files.weights, dtype, &device)? };
        let model = models::mistral::Model::new(&config, vars)?;
        Ok(Self {
            inner: model,
            device,
            dtype,
        })
    }

    pub fn from_repo(repo: &ApiRepo) -> Result<Self> {
        let tokenizer_files = ModelFiles::from_repo(repo)?;
        Self::from_files(tokenizer_files)
    }

    pub fn init_device() -> Result<candle_core::Device> {
        candle_examples::device(false).map_err(|error| Error::CandleError(error.to_string()))
    }

    pub fn init_dtype() -> Result<candle_core::DType> {
        let device = Model::init_device()?;
        if device.is_cuda() {
            Ok(candle_core::DType::BF16)
        } else {
            Ok(candle_core::DType::F32)
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelFiles {
    config: PathBuf,
    weights: Vec<PathBuf>,
    quantized_weights: Vec<PathBuf>,
}

impl ModelFiles {
    pub fn from_repo(repo: &ApiRepo) -> Result<Self> {
        let config = repo.get("config.json")?;
        let weights = hub_load_safetensors(repo, "model.safetensors.index.json")?;
        let repo_info = repo.info()?;
        let quantized_weights: Vec<PathBuf> = repo_info
            .siblings
            .into_iter()
            .filter_map(|sibling| {
                let filename = sibling.rfilename.clone();
                if filename.ends_with(".gguf") {
                    return repo.get(&filename).ok();
                }
                None
            })
            .collect();
        Ok(Self {
            config,
            weights,
            quantized_weights,
        })
    }
}
