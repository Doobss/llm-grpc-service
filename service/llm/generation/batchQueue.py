from multiprocessing import get_context, queues
from .tokenizedGenerationPromptBatch import TokenizedGenerationPromptBatch


class BatchQueue(queues.Queue):

    def __init__(self, *args, **kwargs):
        ctx = get_context()
        super().__init__(*args, **kwargs, ctx=ctx)

    def get_batch(
            self,
            batch: TokenizedGenerationPromptBatch,
            max_batch_size: int = 5
    ) -> TokenizedGenerationPromptBatch:
        while max_batch_size > len(batch) and not self.empty():
            batch += self.get()
        if len(batch) == 0:
            batch += self.get()
        return batch