#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

extern crate approx;
extern crate candle_core;
extern crate candle_examples;
extern crate candle_nn;
extern crate candle_transformers;
extern crate clap;
extern crate hf_hub;
extern crate rand;
extern crate serde;
extern crate serde_json;
extern crate tokenizers;
extern crate tracing;
extern crate tracing_subscriber;
extern crate uuid;

mod batch_encoding;
mod config;
mod error;
mod logits_preprocessor;
mod logits_processor;
mod model;
mod model_files;
mod models;
mod prompt;
mod prompt_config;
mod text_generation;
mod tokenizer;
mod tokenizer_files;
mod utils;

pub use batch_encoding::BatchEncoding;
pub use candle_transformers::generation::Sampling;
pub use config::Config;
pub use error::{Error, Result};
pub use logits_processor::LogitsProcessor;
pub use model::{Model, ModelType};
pub use model_files::ModelFiles;
pub use prompt::Prompt;
pub use prompt_config::PromptConfig;
pub use text_generation::TextGeneration;
pub use tokenizer::Tokenizer;
pub use tokenizer_files::TokenizerFiles;
pub use utils::*;
