import torch
from os import getenv
from typing import List
from logging import getLogger
from transformers import BatchEncoding
from transformers.generation import GenerateDecoderOnlyOutput

from .tokenizer import Tokenizer
from .modelConfig import ModelConfig
from .load import load_model_tokenizer_for_generate
from service.typed import device as _device, IntTensor, TransformerModel
from .tokenizedGenerationPromptBatch import TokenizedGenerationPromptBatch

logger = getLogger()


class Model:

    tokenizer: Tokenizer
    config: ModelConfig
    _model: TransformerModel
    _model_path: str
    _model_cache: str
    _is_cuda: bool
    _cached_past_key_values: dict[str, tuple[tuple[tuple[torch.FloatTensor]]]]

    def __init__(self):
        self._model_path = getenv('services/llm/llm_model_path', 'openchat/openchat_3.5')
        logger.info(f'Initializing model for {self._model_path} ')
        self._is_cuda = torch.cuda.is_available()
        logger.info(f'Model cuda availability: {str(self._is_cuda)} ')
        self.config = ModelConfig()
        if self._is_cuda:
            logger.info(f'Initializing model with quant config: {str(self.config.quantization.to_dict())}')
        else:
            logger.info(f'Initializing model without quant config running in cpu mode')
            self.config.quantization = None
        self._model, tokenizer = load_model_tokenizer_for_generate(
            pretrained_model_name_or_path=self._model_path,
            config=self.config
        )
        self.config.model = self._model.config
        self.tokenizer = Tokenizer(tokenizer=tokenizer)
        self.config.generation.pad_token_id = self.tokenizer.encoded_bos_token.item()
        self.config.generation.bos_token_id = self.tokenizer.encoded_bos_token.item()
        self.config.generation.eos_token_id = self.tokenizer.encoded_eos_token.item()
        self._model.generation_config.update(**self.config.generation.to_dict())
        self.config.generation = self._model.generation_config
        logger.info(f'Quantization config: {self.config.quantization}')
        logger.info(f'Generation config: {self.config.generation}')
        self._cached_past_key_values = {}

    def tokenize(self, strings: List[str]) -> BatchEncoding:
        return self.tokenizer.tokenize(strings)

    def generate(self, prompt_batch: TokenizedGenerationPromptBatch, chunk_size: int = None) -> IntTensor:
        generation_id = prompt_batch.gen_id
        if generation_id in self._cached_past_key_values:
            past_key_values = self._cached_past_key_values[generation_id]
        else:
            past_key_values = None
            self._cached_past_key_values = {}
        generated_kwargs = {
            "past_key_values": past_key_values
        } if past_key_values is not None else {}
        generated: GenerateDecoderOnlyOutput = self._model.generate(
            **prompt_batch.model_inputs,
            max_new_tokens=chunk_size or self.config.chunk_size,
            pad_token_id=self.tokenizer.encoded_bos_token.item(),
            generation_config=self.config.generation,
            return_dict_in_generate=True,
            **generated_kwargs
        )
        self._cached_past_key_values[generation_id] = generated.past_key_values
        return generated.sequences

    def log_status(self) -> None:
        device = self.device()
        if device.type == 'cpu':
            logger.info('Model running on CPU, skipping GPU status logging')
            return

        free_mem, total_memory = torch.cuda.mem_get_info(device)
        summary = torch.cuda.memory_summary(device, abbreviated=True)
        logger.info(f'''
    Model status: 
        memory: {self._model.get_memory_footprint(return_buffers=True)} bytes
    GPU usage:
        free memory: {free_mem} bytes
        total memory: {total_memory} bytes
        summary: 
            {summary}''')

    def device(self) -> torch.device:
        return _device

    def clean_up(self):
        logger.info(f'''Running model cleanup''')
        torch.cuda.empty_cache()
        logger.info(f'''Finished model cleanup''')
        self.log_status()
