use super::template::ChatTemplate;
use super::tokenizer_files::TokenizerFiles;
use super::{BatchEncoding, TokenizerError, TokenizerResult};
use crate::models::ModelType;
use candle_core::Tensor;
use candle_examples::device as get_device;
use clap::builder::Str;
use hf_hub::{api, api::sync::ApiRepo, Repo, RepoType};
use huggingface_tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy};

#[derive(Debug)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    padding: tokenizers::PaddingParams,
    pub template: Option<ChatTemplate>,
    pub pad_id: u32,
    pub bos_id: u32,
    pub eos_id: u32,

    pub pad_token: String,
    pub bos_token: String,
    pub eos_token: String,
}

impl Tokenizer {
    pub fn encode_batch(
        &self,
        prompts: Vec<&str>,
        add_special_tokens: bool,
    ) -> TokenizerResult<BatchEncoding> {
        let encodings = match self.inner.encode_batch(prompts, add_special_tokens) {
            Ok(mut batch) => {
                if self.inner.get_padding().is_none() {
                    tokenizers::pad_encodings(&mut batch, &self.padding)?;
                }
                Ok(batch)
            }
            Err(error) => Err(TokenizerError::TokenizerError(error)),
        }?;

        let mut ids: Vec<&[u32]> = Vec::new();
        let mut attentions: Vec<&[u32]> = Vec::new();
        for encoding in encodings.iter() {
            ids.push(encoding.get_ids());
            attentions.push(encoding.get_attention_mask());
        }
        let device = get_device(false)?;
        let ids = Tensor::new(ids, &device)?;
        let attention_mask = Tensor::new(attentions, &device)?.to_dtype(candle_core::DType::F32)?;
        let ignore_mask = Tensor::full(
            f32::NEG_INFINITY,
            attention_mask.shape(),
            attention_mask.device(),
        )?
        .to_dtype(candle_core::DType::F32)?;
        let padding_tokens = attention_mask.eq(0_f32)?;
        let attention_mask = attention_mask.broadcast_sub(&Tensor::ones(
            1,
            attention_mask.dtype(),
            attention_mask.device(),
        )?)?;
        let attention_mask = padding_tokens.where_cond(&ignore_mask, &attention_mask)?;
        Ok(BatchEncoding {
            ids,
            attention_mask,
        })
    }

    pub fn batch_decode(
        &self,
        token_ids: &Vec<Vec<u32>>,
        skip_special_tokens: bool,
    ) -> TokenizerResult<Vec<String>> {
        let mut token_refs: Vec<&[u32]> = Vec::with_capacity(token_ids.len());
        for token_ref in token_ids {
            token_refs.push(token_ref);
        }
        Ok(self.inner.decode_batch(&token_refs, skip_special_tokens)?)
    }

    pub fn decode_batch(&self, batch: &BatchEncoding) -> TokenizerResult<Vec<String>> {
        let next_tokens_vec: Vec<Vec<u32>> = batch.ids.to_vec2()?;
        self.batch_decode(&next_tokens_vec, true)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.inner.get_vocab(true).get(token_s).copied()
    }

    pub fn id_to_token(&self, token_id: u32) -> Option<String> {
        self.inner.id_to_token(token_id)
    }
}

impl Tokenizer {
    pub fn from_files(files: TokenizerFiles) -> TokenizerResult<Self> {
        tracing::debug!("loading tokenizer config: {:?}", &files.config);

        let config = files.load_config::<serde_json::Value>()?;
        let special_tokens = match files.load_special_tokens::<serde_json::Value>()? {
            Some(value) => value,
            None => config.clone(),
        };

        tracing::debug!("tokenizer config: {:?}", &config);
        tracing::debug!("tokenizer special_tokens: {:?}", &special_tokens);

        let pad_token: String = match special_tokens.get("pad_token") {
            Some(value) => value
                .as_str()
                .map(|slice| slice.to_owned())
                .expect("pad_token is not a string in the tokenizer special_tokens.json"),
            None => match special_tokens.get("eos_token") {
                Some(value) => value
                    .as_str()
                    .map(|slice| slice.to_owned())
                    .expect("eos_token is not a string in the tokenizer special_tokens.json"),
                None => "[PAD]".to_owned(),
            },
        };

        let bos_token = special_tokens
            .get("bos_token")
            .expect("missing bos_token")
            .as_str()
            .map(|slice| slice.to_owned())
            .expect("bos_token is not a string in the tokenizer config.json");
        let eos_token = special_tokens
            .get("eos_token")
            .expect("missing eos_token")
            .as_str()
            .map(|slice| slice.to_owned())
            .expect("eos_token is not a string in the tokenizer config.json");

        let mut tokenizer = tokenizers::Tokenizer::from_file(files.model)?;
        let pad_id = tokenizer.encode(pad_token.clone(), false)?.get_ids()[0];
        let bos_id = tokenizer.encode(bos_token.clone(), false)?.get_ids()[0];
        let eos_id = tokenizer.encode(eos_token.clone(), false)?.get_ids()[0];

        let chat_template: Option<ChatTemplate> = match config.get("chat_template") {
            Some(value) => value.as_str().map(|value| {
                ChatTemplate::new(
                    value.to_owned(),
                    Some(bos_token.clone()),
                    Some(eos_token.clone()),
                )
            }),
            None => None,
        };
        tracing::debug!("tokenizer chat_template: {:?}", &chat_template);

        if tokenizer.get_vocab(true).get(&pad_token).is_none() {
            tracing::debug!(
                "Pad token {:?} not found in vocab, adding token.",
                &pad_token
            );
            tokenizer.add_special_tokens(&[tokenizers::AddedToken {
                content: pad_token.clone(),
                single_word: true,
                lstrip: false,
                rstrip: false,
                normalized: false,
                special: true,
            }]);
        } else {
            tracing::debug!("Pad token {:?} found in vocab.", &pad_token);
        };
        let pad_type_id = 0;
        tracing::debug!("tokenizer bos_token: {:?}", &bos_token);
        tracing::debug!("tokenizer bos_id: {:?}", &bos_id);
        tracing::debug!("tokenizer eos_token: {:?}", &eos_token);
        tracing::debug!("tokenizer eos_id: {:?}", &eos_id);
        tracing::debug!("tokenizer pad_token: {:?}", &pad_token);
        tracing::debug!("tokenizer pad_id: {:?}", &pad_id);
        tracing::debug!("tokenizer pad_type_id: {:?}", &pad_type_id);

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Left,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id,
            pad_token: pad_token.clone(),
        }));
        Ok(Self {
            inner: tokenizer,
            padding: PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                direction: PaddingDirection::Left,
                pad_to_multiple_of: None,
                pad_id,
                pad_type_id,
                pad_token: pad_token.clone(),
            },
            pad_id,
            bos_id,
            eos_id,
            bos_token,
            eos_token,
            pad_token,
            template: chat_template,
        })
    }

    pub fn from_repo(repo: &ApiRepo) -> TokenizerResult<Self> {
        let tokenizer_files = TokenizerFiles::from_repo(repo)?;
        Self::from_files(tokenizer_files)
    }

    pub fn load(model_type: ModelType) -> TokenizerResult<Self> {
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

impl Tokenizer {
    pub fn new_chat_template(&self, template: String) -> ChatTemplate {
        ChatTemplate::new(
            template,
            Some(self.bos_token.clone()),
            Some(self.eos_token.clone()),
        )
    }
}

#[cfg(test)]
mod tests {

    // use super::*;
    // use approx;
    // use rand::{self, Rng, SeedableRng};

    // struct Setup {
    //     // pub model_type: ModelType,
    //     pub tokenizer: Tokenizer,
    // }

    // impl Setup {
    //     fn new() -> TokenizerResult<Self> {
    //         let model_type = ModelType::Mistral7bInstructV02;
    //         Ok(Self {
    //             // model_type,
    //             tokenizer: Tokenizer::load(model_type)?,
    //         })
    //     }
    // }

    // #[test]
    // fn encode_prompts() -> Result<()> {
    //     let setup = Setup::new()?;
    //     let tokenizer = &setup.tokenizer;
    //     let prompts: Vec<Prompt> = vec![
    //         "<s>[INST] Where can I find the best restaurants in NYC? [/INST]</s>"
    //             .to_owned()
    //             .into(),
    //         "<s>[INST] Hello, how are you? [/INST]</s>"
    //             .to_owned()
    //             .into(),
    //     ];
    //     let _batch = tokenizer.encode_batch(prompts, false)?;

    //     Ok(())
    // }
}
