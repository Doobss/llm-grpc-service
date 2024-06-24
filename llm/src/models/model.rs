use super::ModelFiles;
use crate::{ModelResult, ModelType, TokenizedBatch};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use hf_hub::{api, api::sync::ApiRepo, Repo, RepoType};

#[derive(Debug)]
enum InnerModel {
    Mistral(super::models::mistral::Model),
    QuantizedMistral(super::models::quantized_mistral::Model),
}

#[derive(Debug)]
pub struct Model {
    inner: InnerModel,
    device: candle_core::Device,
    dtype: candle_core::DType,
}

impl Model {
    pub fn forward(&mut self, batch: &TokenizedBatch) -> ModelResult<Tensor> {
        match &mut self.inner {
            InnerModel::Mistral(model) => Ok(model
                .forward_with_attention(&batch.input_ids, &batch.attention_mask, 0)
                .expect("Error in model inner forward.")),
            InnerModel::QuantizedMistral(model) => Ok(model
                .forward_with_attention(&batch.input_ids, &batch.attention_mask, 0)
                .expect("Error in model inner forward.")),
        }
    }
}

impl Model {
    pub fn load(model_type: ModelType) -> ModelResult<Self> {
        let api = api::sync::ApiBuilder::new()
            .with_cache_dir("./.cache/huggingface".into())
            .with_token(Some(std::env!("HUGGING_FACE_TOKEN").to_owned()))
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

    pub fn from_files(files: ModelFiles) -> ModelResult<Self> {
        let device = Model::init_device()?;
        let dtype = Model::init_dtype()?;
        let inner = match files.model_type {
            ModelType::QuantizedMistral7bV01 => {
                let config = files.load_config()?;
                // let generation_config = files.load_generation_config();
                tracing::debug!("Model config: {:?}", &config);
                let gguf_file = files
                    .quantized_weights
                    .first()
                    .expect("Cannot load quantized weights. None were found.");
                let vars = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                    gguf_file, &device,
                )?;
                let model = super::models::quantized_mistral::Model::new(&config, vars)?;
                InnerModel::QuantizedMistral(model)
            }
            _ => {
                let config = files.load_config()?;
                tracing::debug!("Model config: {:?}", &config);
                let vars =
                    unsafe { VarBuilder::from_mmaped_safetensors(&files.weights, dtype, &device)? };
                let model = super::models::mistral::Model::new(&config, vars)?;
                InnerModel::Mistral(model)
            }
        };
        Ok(Self {
            inner,
            device,
            dtype,
        })
    }

    pub fn from_repo(model_type: ModelType, repo: &ApiRepo) -> ModelResult<Self> {
        let tokenizer_files = ModelFiles::from_repo(model_type, repo)?;
        Self::from_files(tokenizer_files)
    }

    pub fn init_device() -> ModelResult<candle_core::Device> {
        Ok(candle_examples::device(false)?)
    }

    pub fn init_dtype() -> ModelResult<candle_core::DType> {
        let device = Model::init_device()?;
        if device.is_cuda() {
            Ok(candle_core::DType::BF16)
        } else {
            Ok(candle_core::DType::F32)
        }
    }
}
