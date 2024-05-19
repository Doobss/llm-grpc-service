import torch as T
from os import getenv
from logging import getLogger
from transformers import BatchEncoding
from typing import List

from service.typed import device as _device, IntTensor, TransformerModel
from service.llm.load import load_model_tokenizer_for_generate
from .tokenizedGenerationPrompt import TokenizedGenerationPrompt
from .tokenizedGenerationPromptBatch import TokenizedGenerationPromptBatch
from .tokenizer import Tokenizer
from .modelConfig import ModelConfig

logger = getLogger()


class Model:

    tokenizer: Tokenizer
    config: ModelConfig
    _model: TransformerModel
    _model_path: str
    _model_cache: str
    _is_cuda: bool

    def __init__(self):
        self._model_path = getenv('services/llm/llm_model_path', 'openchat/openchat_3.5')
        logger.info(f'Initializing model for {self._model_path} ')
        self._is_cuda = T.cuda.is_available()
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
        self._model.generation_config = self.config.generation
        logger.info(f'Quantization config: {self.config.quantization}')
        logger.info(f'Generation config: {self.config.generation}')

    def tokenize(self, strings: List[str]) -> BatchEncoding:
        return self.tokenizer.tokenize(strings)

    def generate(self, prompt_batch: TokenizedGenerationPromptBatch, chunk_size: int = None) -> IntTensor:
        return self._model.generate(
                    **prompt_batch.model_inputs,
                    max_new_tokens=chunk_size or self.config.chunk_size,
                    pad_token_id=self.tokenizer.encoded_bos_token.item(),
                    generation_config=self.config.generation
                )

    def log_status(self) -> None:
        device = self.device()
        if device.type == 'cpu':
            logger.info('Model running on CPU, skipping GPU status logging')
            return

        free_mem, total_memory = T.cuda.mem_get_info(device)
        summary = T.cuda.memory_summary(device, abbreviated=True)
        logger.info(f'''
    Model status: 
        memory: {self._model.get_memory_footprint(return_buffers=True)} bytes
    GPU usage:
        free memory: {free_mem} bytes
        total memory: {total_memory} bytes
        summary: 
            {summary}''')

    def device(self) -> T.device:
        return _device

    def clean_up(self):
        logger.info(f'''Running model cleanup''')
        T.cuda.empty_cache()
        logger.info(f'''Finished model cleanup''')
        self.log_status()