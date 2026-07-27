import os
import gc
import math
import torch
from typing import List, Optional
from multiprocess import get_context
from multiprocess.queues import Empty
from huggingface_hub import snapshot_download
from src.utils.utils import init_logger

logger = init_logger()


class InferenceTask:
    GENERATE = "generate"
    REWARD = "reward"
    EMBEDDING = "embedding"
    CLASSIFY = "classify"


class ParallelvLLMInference:
    def __init__(
        self,
        model_path: str,
        n_instances: Optional[int] = None,
        gpus_per_instance: int = 2,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 9000,
        max_num_seqs: int = 5,
        enforce_eager: bool = False,
        model_save_path: Optional[str] = None,
        use_lora: bool = False,
        load_and_unload: bool = True,
        max_number_of_instances: int = -1,
        inference_task: InferenceTask = InferenceTask.GENERATE,
        bits_and_bytes: bool = False,
        enable_sleep_mode: bool = True,
        from_0: bool = True,
        use_v0: bool = False,
        logging_enabled: bool = False,
        log_file_path: Optional[str] = None,
        gpu_ids: Optional[List[int]] = None,
        userlm_mode: bool = False,
        chat_template_kwargs: Optional[dict] = None,
    ):
        self.model_path = model_path
        self.model_save_path = model_save_path
        self.total_gpus = torch.cuda.device_count()
        # gpu_ids: pin this model to exactly these GPU indices as a single
        # resident instance (used to give each eval model its own GPU).
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        # userlm_mode: render prompts with the model's chat template + a
        # prepended BOS, then call .generate() (UserLM's template omits BOS).
        self.userlm_mode = userlm_mode
        # chat_template_kwargs: extra kwargs for llm.chat's template rendering
        # (e.g. {"reasoning_effort": "low"} for gpt-oss judges).
        self.chat_template_kwargs = chat_template_kwargs
        if self.gpu_ids is not None:
            gpus_per_instance = len(self.gpu_ids)
        self.gpus_per_instance = gpus_per_instance
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self.use_lora = use_lora
        self.load_and_unload = load_and_unload
        self.inference_task = inference_task
        self.bits_and_bytes = bits_and_bytes
        self.enable_sleep_mode = enable_sleep_mode
        self.use_v0 = use_v0
        self.logging_enabled = logging_enabled
        self.log_file = (
            os.path.join(log_file_path, "input_prompts.txt") if log_file_path else ""
        )

        if self.load_and_unload and not self.enable_sleep_mode:
            raise ValueError("Cannot use load_and_unload without enabling sleep mode")

        logger.info(
            f"Total GPUs: {self.total_gpus}, using {self.gpus_per_instance} per instance"
        )

        # Try downloading the model to cache
        try:
            snapshot_download(self.model_path)
        except Exception as e:
            logger.warning(f"Could not download model: {e}; will load from cache/local")

        # Determine number of instances
        if n_instances:
            required = n_instances * gpus_per_instance
            if required > self.total_gpus:
                raise ValueError(
                    f"Need {required} GPUs, only {self.total_gpus} available"
                )
            self.n_instances = n_instances
        else:
            self.n_instances = max(1, self.total_gpus // self.gpus_per_instance)

        # Build GPU groups [[0,1], [2,3], ...]
        self.gpu_groups = [
            list(range(i, i + self.gpus_per_instance))
            for i in range(
                0, self.n_instances * self.gpus_per_instance, self.gpus_per_instance
            )
        ]
        if not from_0:
            self.gpu_groups = [
                [(self.total_gpus - 1 - gpu) for gpu in group]
                for group in self.gpu_groups
            ]

        if max_number_of_instances > 0:
            self.n_instances = min(max_number_of_instances, self.n_instances)
            self.gpu_groups = self.gpu_groups[: self.n_instances]
            logger.info(f"Limiting number of instances to {self.n_instances}")

        # Explicit GPU pinning: one resident instance on exactly self.gpu_ids.
        if self.gpu_ids is not None:
            self.gpu_groups = [list(self.gpu_ids)]
            self.n_instances = 1
            logger.info(f"Pinned single instance to GPUs {self.gpu_ids}")

        # Track last checkpoint ID seen
        self._last_reload_ckpt = None

        # Start worker processes
        self._start_workers()

    def _get_latest_checkpoint_id(self) -> Optional[int]:
        """Scan model_save_path for 'checkpoint-<n>' folders and return max(n), or None."""
        if self.model_save_path and os.path.isdir(self.model_save_path):
            ids = []
            for d in os.listdir(self.model_save_path):
                if d.startswith("checkpoint-"):
                    parts = d.split("-")
                    if parts[-1].isdigit():
                        ids.append(int(parts[-1]))
            if ids:
                return max(ids)
        return None

    def _start_workers(self):
        """Spawn worker processes and wait until they're READY."""
        self.ctx = get_context("spawn")
        self.task_queues = [self.ctx.Queue() for _ in range(self.n_instances)]
        self.result_queues = [self.ctx.Queue() for _ in range(self.n_instances)]
        self.processes = []

        for idx, gpu_group in enumerate(self.gpu_groups):
            p = self.ctx.Process(
                target=self._worker_loop,
                args=(
                    gpu_group,
                    self.task_queues[idx],
                    self.result_queues[idx],
                    self.inference_task,
                ),
            )
            p.start()
            self.processes.append(p)

        # Wait for each worker to signal "READY" -- fail fast instead of
        # hanging forever if a worker dies before ever sending it (e.g. a
        # model-loading crash). Without this, a dead worker leaves the
        # server hung indefinitely on this call, silently burning a full
        # GPU allocation instead of exiting so the job can be resubmitted.
        for idx, q in enumerate(self.result_queues):
            p = self.processes[idx]
            while True:
                try:
                    q.get(timeout=1.0)
                    break
                except Empty:
                    if not p.is_alive():
                        raise RuntimeError(
                            f"Worker {idx} (GPUs {self.gpu_groups[idx]}) died "
                            f"during startup with exit code {p.exitcode} "
                            "before signaling READY -- see its stderr above "
                            "for the actual crash."
                        )

    def _reload_workers(self):
        latest = self._get_latest_checkpoint_id()
        if latest is not None:
            new_path = os.path.join(self.model_save_path, f"checkpoint-{latest}")
            logger.info(f"Reload: switching model_path to {new_path}")
            self.model_path = new_path
            self._last_reload_ckpt = latest

        self.cleanup()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        self._start_workers()
        torch.cuda.empty_cache()

    def run_batch(
        self, messages: List[str], sampling_params: dict, meta: Optional[dict] = None
    ):

        current = self._get_latest_checkpoint_id()
        if current is not None and current != self._last_reload_ckpt:
            logger.info(
                f"New checkpoint {current} detected (was {self._last_reload_ckpt}); reloading workers."
            )
            self._reload_workers()

        indexed = list(enumerate(messages))
        total = len(indexed)
        chunk_size = math.ceil(total / self.n_instances)
        chunks = [
            indexed[i * chunk_size : (i + 1) * chunk_size]
            for i in range(self.n_instances)
        ]

        for i, chunk in enumerate(chunks):
            self.task_queues[i].put((chunk, sampling_params, meta))

        results = []
        received = 0
        expected = total
        while received < expected:
            for i, q in enumerate(self.result_queues):
                try:
                    batch = q.get(block=True, timeout=0.1)
                    if batch not in ("READY", "SLEEP_DONE"):
                        results.extend(batch)
                        received += len(batch)
                except Empty:
                    # A dead worker will never produce the rest of its
                    # chunk -- raise instead of spinning on this Empty
                    # forever. This propagates as a 500 to the trainer's
                    # HTTP call, which crashes the trainer and (running in
                    # the foreground under `set -euo pipefail`) ends the
                    # whole job instead of it hanging until manually killed.
                    if not self.processes[i].is_alive():
                        raise RuntimeError(
                            f"Worker {i} (GPUs {self.gpu_groups[i]}) died "
                            f"with exit code {self.processes[i].exitcode} "
                            "mid-batch -- see its stderr above for the "
                            "actual crash."
                        )
                    continue

        return [out for _, out in sorted(results, key=lambda x: x[0])]

    def sleep(self):
        # Resident mode: keep weights on-GPU, never offload.
        if not self.load_and_unload:
            return
        for q in self.task_queues:
            q.put("SLEEP")
        for q in self.result_queues:
            resp = q.get()
            if resp != "SLEEP_DONE":
                logger.error("Unexpected response to SLEEP:", resp)
        logger.info("All workers are now asleep.")

    def cleanup(self):
        for q in self.task_queues:
            q.put(None)
        for p in self.processes:
            p.join()
        for q in (*self.task_queues, *self.result_queues):
            q.close()
        torch.cuda.empty_cache()
        logger.info("Cleaned up all resources")

    def _handle_reward_task(self, llm, prompts: List[str], tokenizer):
        prompts = [
            tokenizer.decode(tokenizer.encode(p)[: self.max_model_len - 1])
            for p in prompts
        ]
        return llm.encode(prompts)

    def _handle_embedding_task(self, llm, prompts: List[str]):
        return llm.embed(prompts)

    def _handle_classify_task(self, llm, prompts: List[str]):
        return llm.classify(prompts)

    def _handle_causallm_task(
        self,
        llm,
        prompts: List[str],
        sampling_params: dict,
        meta: Optional[dict],
        counter: int,
    ):
        # UserLM: its chat template omits the BOS its Llama-3.1 base expects, so
        # render the template ourselves, prepend BOS, and use .generate().
        if self.userlm_mode:
            from vllm import TokensPrompt

            tok = llm.get_tokenizer()
            token_prompts = []
            for msgs in prompts:
                ids = tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True, return_dict=False
                )
                if hasattr(ids, "input_ids"):  # BatchEncoding (transformers v5)
                    ids = ids["input_ids"]
                ids = list(ids)
                if tok.bos_token_id is not None and (
                    not ids or ids[0] != tok.bos_token_id
                ):
                    ids = [tok.bos_token_id] + ids
                token_prompts.append(TokensPrompt(prompt_token_ids=ids))
            return llm.generate(token_prompts, sampling_params=sampling_params)

        chat_kwargs = (
            {"chat_template_kwargs": self.chat_template_kwargs}
            if self.chat_template_kwargs
            else {}
        )

        # Empty meta ({}) means "no policy-weight reload" (the eval path); only the
        # RL trainer passes a non-empty meta with shared-memory weight handles.
        if meta:
            from ..utils.shared_memory import load_shared_state_dict

            state = load_shared_state_dict(meta).items()
            llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
                state
            )
            return llm.chat(prompts, sampling_params=sampling_params, **chat_kwargs)

        return llm.chat(prompts, sampling_params=sampling_params, **chat_kwargs)

    def _worker_loop(
        self, gpu_group: List[int], task_queue, result_queue, inference_task
    ):
        import os
        import gc

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_group))

        if self.use_v0:
            os.environ["VLLM_USE_V1"] = "0"
        else:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        from vllm import LLM

        print(f"Worker on GPUs {gpu_group} initializing for task '{inference_task}'...")

        # vllm 0.23 removed the `task=` kwarg (was on EngineArgs in 0.8.x).
        # Generation is the default runner; our eval path is generate-only
        # (reward_model="Answer" => no pooling model is ever loaded here).
        llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.gpus_per_instance,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            enable_lora=self.use_lora,
            enforce_eager=self.enforce_eager,
            enable_prefix_caching=False,
            enable_sleep_mode=self.enable_sleep_mode,
            quantization="bitsandbytes" if self.bits_and_bytes else None,
            load_format="bitsandbytes" if self.bits_and_bytes else "auto",
        )
        tokenizer = llm.get_tokenizer()

        if self.load_and_unload:
            llm.sleep()

        result_queue.put("READY")
        counter = 0

        while True:
            task = task_queue.get()
            if task is None:
                break
            if task == "SLEEP":
                try:
                    llm.sleep()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error during sleep: {e}")
                result_queue.put("SLEEP_DONE")
                continue

            if self.inference_task == InferenceTask.GENERATE and self.load_and_unload:
                llm.wake_up()

            chunk, sampling_params, meta = task
            prompts = [p for _, p in chunk]

            if inference_task == InferenceTask.REWARD:
                outs = self._handle_reward_task(llm, prompts, tokenizer)
            elif inference_task == InferenceTask.EMBEDDING:
                outs = self._handle_embedding_task(llm, prompts)
            elif inference_task == InferenceTask.CLASSIFY:
                outs = self._handle_classify_task(llm, prompts)
            else:
                outs = self._handle_causallm_task(
                    llm, prompts, sampling_params, meta, counter
                )
                counter += 1

            gc.collect()
            torch.cuda.empty_cache()
            if self.load_and_unload:
                llm.sleep()

            result_queue.put([(idx, out) for (idx, _), out in zip(chunk, outs)])

        # Final cleanup
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        import contextlib

        destroy_model_parallel()
        destroy_distributed_environment()
        # vllm 0.23 renamed/removed LLMEngine.model_executor; tolerate its absence.
        with contextlib.suppress(AttributeError):
            del llm.llm_engine.model_executor
        del llm
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
