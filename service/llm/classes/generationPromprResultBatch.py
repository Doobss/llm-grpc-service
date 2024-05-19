from typing import Union, Dict, List
from .generationBatch import GenerationBatch
from .generationPromptResult import GenerationPromptResult


class GenerationPromptResultBatch(GenerationBatch[GenerationPromptResult]):
    process_time: float = 0.0
