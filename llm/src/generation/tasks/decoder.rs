use candle_core::{IndexOp, Tensor};
use indexmap::IndexMap;

use super::{Receiver, Sender, TaskResult};
use crate::{
    GenerationLogitsProcessor, GenerationResult, GenerationStep, TokenizedBatch, Tokenizer,
};

#[derive(Debug)]
pub struct Decoder {
    tokenizer: std::sync::Arc<Tokenizer>,
    generation_result_receiver: Receiver<GenerationStep>,
    tokenized_batch_sender: Sender<TokenizedBatch>,
}

impl Decoder {
    pub fn task(self) -> TaskResult<()> {
        let Decoder {
            tokenizer,
            tokenized_batch_sender,
            mut generation_result_receiver,
        } = self;
        tokio::task::spawn_blocking(move || {
            let tokenizer = tokenizer.as_ref();
            loop {
                tracing::debug!("decode_task: awaiting results");
                if let Some(generation_result) = generation_result_receiver.blocking_recv() {
                    let loop_start = tokio::time::Instant::now();

                    let GenerationStep { batch, logits } = generation_result;
                    let TokenizedBatch {
                        mut requests,
                        input_ids,
                        attention_mask,
                        past_key_values,
                        mut token_ids,
                    } = batch;
                    struct ProcessedToken {
                        pub token_id: u32,
                        pub is_end_of_sequence: bool,
                    }

                    let mut processed_tokens = Vec::with_capacity(requests.len());
                    let mut added_tokens = Vec::with_capacity(token_ids.len());
                    for (index, request) in requests.values_mut().enumerate() {
                        let logit_row = logits.i(index)?.squeeze(0)?;
                        let GenerationLogitsProcessor {
                            preprocess: _,
                            mut process,
                        } = request.logit_processor();
                        let token_id = process.sample(&logit_row).expect("Errror sampling");
                        let is_end_of_sequence = token_id == tokenizer.eos_id;
                        processed_tokens.push(ProcessedToken {
                            token_id,
                            is_end_of_sequence,
                        });
                        token_ids[index].push(token_id);
                        added_tokens.push(token_id);
                        request.number_tokens_generated += 1;
                    }
                    let process_time = loop_start.elapsed();

                    let decoded_text = tokenizer
                        .batch_decode(&token_ids, false)
                        .expect("Error batch decode");

                    let mut indicies_to_keep = Vec::new();
                    let mut kept_requests = IndexMap::new();
                    let iterator = requests
                        .into_values()
                        .zip(processed_tokens)
                        .zip(decoded_text)
                        .enumerate();

                    let decode_time = loop_start.elapsed();

                    for (index, ((request, processed), generated)) in iterator {
                        let ProcessedToken {
                            token_id,
                            mut is_end_of_sequence,
                        } = processed;
                        let reached_max_tokens =
                            request.number_tokens_generated >= request.config.max_new_tokens as u32;
                        tracing::info!("reached_max_tokens: {}", &reached_max_tokens);
                        if reached_max_tokens {
                            is_end_of_sequence = true
                        }
                        if request.sender().is_closed() {
                            is_end_of_sequence = true
                        } else {
                            request.sender().blocking_send(GenerationResult {
                                id: request.id.clone(),
                                content: generated.clone(),
                                generated: tokenizer.id_to_token(token_id).unwrap_or_default(),
                                is_end_of_sequence,
                                config: request.config.clone(),
                            })?;
                        }
                        if !is_end_of_sequence {
                            indicies_to_keep.push(index as u32);
                            kept_requests.insert(request.id.clone(), request);
                        }
                    }
                    let send_time = loop_start.elapsed();
                    let num_indicies = indicies_to_keep.len();

                    if num_indicies > 0 {
                        let device = input_ids.device();

                        let mut filtered_token_ids = Vec::with_capacity(num_indicies);
                        for (index, vec) in token_ids.into_iter().enumerate() {
                            if indicies_to_keep.contains(&(index as u32)) {
                                filtered_token_ids.push(vec)
                            }
                        }
                        let indicies_to_keep =
                            Tensor::from_vec(indicies_to_keep, num_indicies, device)?;
                        let input_ids_shape = input_ids.dims();

                        let added_tokens =
                            Tensor::new(added_tokens, input_ids.device())?.unsqueeze(1)?;

                        let input_ids = Tensor::cat(&[&input_ids, &added_tokens], 1)
                            .unwrap_or_else(|error| {
                                panic!(
                                    "Error creating new input ids {:?} error: {:?}",
                                    &input_ids_shape, error
                                )
                            });
                        let input_ids = input_ids
                            .index_select(&indicies_to_keep, 0)
                            .unwrap_or_else(|error| {
                                panic!(
                                    "Error selecting new input ids {:?} error: {:?}",
                                    &indicies_to_keep, error
                                )
                            });
                        let added_attention =
                            Tensor::zeros((input_ids_shape[0], 1), attention_mask.dtype(), device)
                                .unwrap_or_else(|error| {
                                    panic!("Error creating added attention: {:?}", error)
                                });

                        let attention_mask = Tensor::cat(&[&attention_mask, &added_attention], 1)
                            .unwrap_or_else(|error| {
                                panic!("Error cating added attention: {:?}", error)
                            });
                        let attention_mask = attention_mask.index_select(&indicies_to_keep, 0)?;
                        let past_key_values = if let Some(past_key_values) = past_key_values {
                            Some(past_key_values.index_select(&indicies_to_keep, 0)?)
                        } else {
                            None
                        };

                        let next_batch = TokenizedBatch {
                            requests: kept_requests,
                            token_ids: filtered_token_ids,
                            input_ids,
                            attention_mask,
                            past_key_values,
                        };
                        if !next_batch.is_empty() {
                            tracing::debug!(
                                "decode_task: sending non finished batch back to generation."
                            );
                            tokenized_batch_sender.blocking_send(next_batch)?;
                        }
                    }
                    let filter_time = loop_start.elapsed();

                    let loop_end = loop_start.elapsed().as_micros();
                    let process_time = process_time.as_micros();
                    let decode_time = decode_time.as_micros() - process_time;
                    let send_time = send_time.as_micros() - (decode_time + process_time);
                    let filter_time =
                        filter_time.as_micros() - (send_time + decode_time + process_time);
                    tracing::debug!("decoder task finished in: {:?} micro seconds | process: {:?} ms | decode: {:?} ms | send: {:?} ms | filter: {:?} ms", loop_end, process_time, decode_time, send_time, filter_time);
                }
            }
        })
    }
}

impl Decoder {
    pub fn new(
        tokenizer: std::sync::Arc<Tokenizer>,
        generation_result_receiver: Receiver<GenerationStep>,
        tokenized_batch_sender: Sender<TokenizedBatch>,
    ) -> Self {
        Self {
            tokenizer,
            generation_result_receiver,
            tokenized_batch_sender,
        }
    }
}
