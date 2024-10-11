use crate::{ModelConfig, ModelFiles, ModelResult, TokenizedBatch};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use hf_hub::api::sync::ApiRepo;

#[derive(Debug)]
enum InnerModel {
    Mistral(super::models::mistral::Model),
    QuantizedMistral(super::models::quantized_mistral::Model),
}

#[derive(Debug)]
pub struct Model {
    config: ModelConfig,
    inner: InnerModel,
    device: candle_core::Device,
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
    pub fn load(config: ModelConfig) -> ModelResult<Self> {
        tracing::debug!("loading model_id: {:?}", config.model_id);
        let repo = config.api_repo()?;
        Self::from_repo(config, &repo)
    }

    pub fn from_files(config: ModelConfig, files: ModelFiles) -> ModelResult<Self> {
        let device = Model::init_device()?;
        let dtype = Model::init_dtype()?;
        let inner = match config.quantize {
            true => {
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
            config,
        })
    }

    pub fn from_repo(config: ModelConfig, repo: &ApiRepo) -> ModelResult<Self> {
        let tokenizer_files = ModelFiles::from_repo(config.model_id, repo)?;
        Self::from_files(config, tokenizer_files)
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
