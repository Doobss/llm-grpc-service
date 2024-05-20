import sys
import torch
import signal
import asyncio
import traceback
from time import time
from threading import Thread
from logging import getLogger
from typing import AsyncIterator
from service.typed import device
from .batchQueue import BatchQueue
from .model import Model, ModelConfig
from multiprocessing import Process, Queue
from .generationPrompt import GenerationPrompt
from .generationPromptResult import GenerationPromptResult
from .generationPromprResultBatch import GenerationPromptResultBatch
from .tokenizedGenerationPromptBatch import TokenizedGenerationPromptBatch

log = getLogger(__name__)


def _target(input_queue: BatchQueue, output_queue: Queue) -> None:
    m_log = getLogger('ModelProcess')

    with torch.no_grad():
        m_log.info(f' initializing model')
        model = Model()
        output_queue.put(model.config)
        m_log.info(f' finished initializing model')
        batch = TokenizedGenerationPromptBatch(tokenizer=model.tokenizer, prompts=[])
        while True:
            try:
                batch = input_queue.get_batch(batch, max_batch_size=model.config.max_batch_size)
                if len(batch) and not batch.finished_generation:
                    batch_start = time()
                    m_log.info(f' processing batch {len(batch) = }')
                    batch = batch.process_chunk(
                        generation_output=model.generate(
                            prompt_batch=batch
                        )
                    )
                    m_log.info(f' finished processing batch {len(batch) = }')
                    output_queue.put(
                        batch.results(
                            process_time=time() - batch_start
                        )
                    )
            except ValueError as closed_error:
                m_log.error(f' input queue was closed, breaking generation loop: {str(closed_error) = }')
                break
            except Exception as error:
                trace = traceback.format_exc()
                m_log.error(f' error occurred: {str(error) = }\n {trace = }')
                break


class ModelProcess:

    _input_queue: BatchQueue
    _output_queue: Queue
    _model_process: Process
    _process_output_thread: Thread
    _result_queues: dict[str, Queue]
    config: ModelConfig

    def __init__(self) -> None:
        self._input_queue = BatchQueue()
        self._output_queue = Queue()
        self._result_queues = {}
        self._model_process = Process(
            target=_target,
            args=(self._input_queue, self._output_queue)
        )

        def process_output() -> None:
            while True:
                batch_results: GenerationPromptResultBatch = self._output_queue.get()
                for key, result in batch_results:
                    result_queue = self._result_queues.get(key, None)
                    if result_queue is not None:
                        result_queue.put_nowait(result)

        self._process_output_thread = Thread(target=process_output)

        def shutdown(sig, frame) -> None:
            log.info(f'Killing model process pid: {self._model_process.pid}')
            self._process_output_thread.join()
            self._model_process.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
        self._model_process.start()
        self.config = self._output_queue.get()
        self._process_output_thread.start()
        log.info(f'Model process running pid: {self._model_process.pid}')

    async def prompt(self, message: GenerationPrompt, result_key: str = None) -> AsyncIterator[GenerationPromptResult]:
        try:
            result_key = result_key or message.id
            result_queue = self._result_queues.get(result_key, Queue())
            self._result_queues[result_key] = result_queue
            self._input_queue.put(GenerationPrompt(
                prompt=message.prompt,
                id=result_key,
            ))
            while True:
                queue_result = await asyncio.to_thread(result_queue.get)
                yield queue_result
                if queue_result.reached_eos:
                    break
        except Exception as error:
            log.error(f'ModelProcess.prompt error: {str(error)}')
            raise error
        finally:
            if result_key in self._result_queues:
                del self._result_queues[result_key]

    @property
    def model_info(self) -> dict:
        is_cuda = torch.cuda.is_available()
        return {
            "type": self.config.model.model_type,
            "device": torch.cuda.get_device_name(device)
            if is_cuda else str(device),
            "number_of_devices": torch.cuda.device_count() if is_cuda else 1,
            "dtype": str(self.config.dtype),
            "config": self.config.to_dict(),
        }

    @property
    def get_chunk_size(self) -> int:
        return self.config.chunk_size


# Can you list the letters of the cyrilic alpahbet?
# What are the 10 best restaurants in NYC and what kind of food do they make?
