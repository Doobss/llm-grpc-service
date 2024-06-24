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
extern crate cudarc;
extern crate hf_hub;
extern crate minijinja;
extern crate minijinja_contrib;
extern crate rand;
extern crate serde;
extern crate serde_json;
extern crate tokenizers as huggingface_tokenizers;
extern crate tracing;
extern crate tracing_subscriber;
extern crate uuid;

mod device;
mod error;
mod generation;
mod logits;
mod models;
mod prompts;
mod tokenizers;
mod utils;

pub use candle_transformers::generation::Sampling;
pub use error::{Error, Result};

pub use device::*;
pub use generation::*;
pub use logits::*;
pub use models::*;
pub use prompts::*;
pub use tokenizers::*;
pub use utils::*;
