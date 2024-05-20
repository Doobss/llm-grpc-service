from typing import Union, Dict, List
from .generationBatch import GenerationBatch
from .generationPrompt import GenerationPrompt


class GenerationPromptBatch(GenerationBatch[GenerationPrompt]):
    ...
