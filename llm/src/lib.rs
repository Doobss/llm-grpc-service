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
extern crate tokenizers as huggingface_tokenizers;
extern crate tracing;
extern crate tracing_subscriber;
extern crate uuid;

mod batch_encoding;
mod config;
mod error;
mod generation;
mod logits;
mod models;
mod prompts;
mod tokenizers;
mod utils;

pub use batch_encoding::BatchEncoding;
pub use candle_transformers::generation::Sampling;
pub use config::Config;
pub use error::{Error, Result};

pub use generation::*;
pub use logits::*;
pub use models::*;
pub use prompts::*;
pub use tokenizers::*;
pub use utils::*;
