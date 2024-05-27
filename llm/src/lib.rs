#[macro_use]
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

mod batch_encoding;
mod config;
mod error;
mod logits_processor;
mod model;
mod models;
mod prompt;
mod text_generation;
mod tokenizer;

pub use batch_encoding::BatchEncoding;
pub use config::Config;
pub use error::{Error, Result};
pub use logits_processor::LogitsProcessor;
pub use model::{Model, ModelType};
pub use prompt::Prompt;
pub use text_generation::TextGeneration;
pub use tokenizer::Tokenizer;
