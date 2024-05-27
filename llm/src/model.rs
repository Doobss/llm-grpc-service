use crate::{models, Error, Result};
use candle_core::Tensor;
use candle_examples::hub_load_safetensors;
use candle_nn::VarBuilder;
use candle_transformers;
use clap;
use hf_hub::{api, api::sync::ApiRepo, Repo, RepoType};
use std::path::PathBuf;

// use tokenizers::parallelism::MaybeParallelRefIterator;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ModelType {
    #[value(name = "7b-v0.2")]
    Mistral7bV02,
    #[value(name = "7b-instruct-v0.2")]
    Mistral7bInstructV02,
    #[value(name = "7B-v0.1")]
    Mistral7bV01,
    #[value(name = "7B-v0.1-quant")]
    QuantizedMistral7bV01,
}

impl ModelType {
    pub fn path(&self) -> String {
        match self {
            ModelType::Mistral7bV01 => "mistralai/Mistral-7B-v0.1".to_string(),
            ModelType::QuantizedMistral7bV01 => "mistralai/Mistral-7B-v0.1".to_string(),
            ModelType::Mistral7bV02 => "mistralai/Mistral-7B-v0.2".to_string(),
            // ModelType::Mistral7bInstructV01 => "mistralai/Mistral-7B-Instruct-v0.1".to_string(),
            ModelType::Mistral7bInstructV02 => "mistralai/Mistral-7B-Instruct-v0.2".to_string(),
        }
    }

    pub fn is_quantized(self) -> bool {
        self == ModelType::QuantizedMistral7bV01
    }
}

#[derive(Debug)]
enum InnerModel {
    Mistral(models::mistral::Model),
    QuantizedMistral(models::quantized_mistral::Model),
}

#[derive(Debug)]
pub struct Model {
    inner: InnerModel,
    device: candle_core::Device,
    dtype: candle_core::DType,
}

impl Model {
    pub fn forward(&mut self, input_tokens: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        match &mut self.inner {
            InnerModel::Mistral(model) => Ok(model
                .forward_with_attention(input_tokens, attention_mask, 0)
                .expect("Error in model inner forward.")),
            InnerModel::QuantizedMistral(model) => Ok(model
                .forward_with_attention(input_tokens, attention_mask, 0)
                .expect("Error in model inner forward.")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelFiles {
    model_type: ModelType,
    config: PathBuf,
    weights: Vec<PathBuf>,
    quantized_weights: Vec<PathBuf>,
}

impl ModelFiles {
    pub fn from_repo(model_type: ModelType, repo: &ApiRepo) -> Result<Self> {
        let config = repo.get("config.json")?;
        let mut weights = Vec::new();
        let mut quantized_weights = Vec::new();
        let repo_info = repo.info()?;
        if model_type.is_quantized() {
            quantized_weights = repo_info
                .siblings
                .into_iter()
                .filter_map(|sibling| {
                    let filename = sibling.rfilename.clone();
                    tracing::info!("sibling filename: {:?}", &filename);
                    if filename.ends_with(".gguf") {
                        return repo.get(&filename).ok();
                    }
                    None
                })
                .collect();
            if quantized_weights.is_empty() {
                tracing::info!("attempting to get model-q4k.gguf");
                quantized_weights = vec![repo.get("model-q4k.gguf")?]
            }
        } else {
            weights = hub_load_safetensors(repo, "model.safetensors.index.json")?;
        }
        Ok(Self {
            model_type,
            config,
            weights,
            quantized_weights,
        })
    }
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
        Self::from_repo(model_type, &repo)
    }

    pub fn from_files(files: ModelFiles) -> Result<Self> {
        let device = Model::init_device()?;
        let dtype = Model::init_dtype()?;
        let inner = match files.model_type {
            ModelType::QuantizedMistral7bV01 => {
                let config = serde_json::from_slice(&std::fs::read(files.config)?)?;
                let gguf_file = files
                    .quantized_weights
                    .first()
                    .expect("Cannot load quantized weights. None were found.");
                let vars = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                    gguf_file, &device,
                )?;
                let model = models::quantized_mistral::Model::new(&config, vars)?;
                InnerModel::QuantizedMistral(model)
            }
            _ => {
                let config = serde_json::from_slice(&std::fs::read(files.config)?)?;
                let vars =
                    unsafe { VarBuilder::from_mmaped_safetensors(&files.weights, dtype, &device)? };
                let model = models::mistral::Model::new(&config, vars)?;
                InnerModel::Mistral(model)
            }
        };
        Ok(Self {
            inner,
            device,
            dtype,
        })
    }

    pub fn from_repo(model_type: ModelType, repo: &ApiRepo) -> Result<Self> {
        let tokenizer_files = ModelFiles::from_repo(model_type, repo)?;
        Self::from_files(tokenizer_files)
    }

    pub fn init_device() -> Result<candle_core::Device> {
        candle_examples::device(false).map_err(Error::CandleError)
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
