from transformers import LlamaModel, MistralModel, LlamaTokenizerFast
import torch as T
from typing import Union
from multiprocessing import set_start_method
from logging import getLogger

logger = getLogger()

try:
    set_start_method('spawn', force=True)
    logger.info(f"MP start method set to spawned")
except RuntimeError:
    pass

device = T.device('cpu')
if T.cuda.is_available():
    device = T.device('cuda:0')
# T.set_default_device(device)
logger.info(f'PYTORCH default device: {device}')

IntTensor = Union[T.IntTensor, T.cuda.IntTensor]
BoolTensor = Union[T.BoolTensor, T.cuda.BoolTensor]


TransformerModel = Union[LlamaModel, MistralModel]
TransformerTokenizer = Union[LlamaTokenizerFast]