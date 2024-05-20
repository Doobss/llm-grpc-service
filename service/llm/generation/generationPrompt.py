from dataclasses import dataclass


@dataclass
class GenerationPrompt:

    id: str
    prompt: str = ""
    generated: str = ""
    reached_eos: bool = False,