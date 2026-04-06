from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from cs336_scaling.api_contract import TrainingConfig, estimate_non_embedding_parameters, estimate_train_tokens
from cs336_scaling.token_dataset import TokenizedDataset
from cs336_scaling.training_runner import TrainingRunner

logger = logging.getLogger("uvicorn.error")


class BackendUnavailableError(RuntimeError):
    pass


class TrainingOOMError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingResult:
    loss: float


class TrainingBackend(Protocol):
    max_concurrency: int

    def run(self, config: TrainingConfig) -> TrainingResult: ...

    def begin_shutdown(self) -> None: ...

    def wait_for_shutdown(self, timeout: float | None = None) -> bool: ...


class RunnerProtocol(Protocol):
    device: torch.device

    def run(self, config: TrainingConfig): ...


class PlaceholderTrainingBackend:
    """Stub backend for the assignment-style API.

    The contract layer, persistence, caching, and FLOPs accounting can be built
    before the real trainer is ready. This backend makes that state explicit so
    the API returns a clear 503 on cache misses instead of silently inventing a loss.
    """

    max_concurrency = 1

    def run(self, config: TrainingConfig) -> TrainingResult:
        estimated_tokens = estimate_train_tokens(
            train_flops=config.train_flops,
            d_model=config.d_model,
            num_layers=config.num_layers,
        )
        estimated_params = estimate_non_embedding_parameters(
            d_model=config.d_model,
            num_layers=config.num_layers,
        )
        raise BackendUnavailableError(
            "Training backend is not configured yet. "
            f"This query would train roughly {estimated_tokens:.0f} tokens "
            f"for a model with about {estimated_params} non-embedding parameters."
        )

    def begin_shutdown(self) -> None:
        return None

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        return True


@dataclass
class _WorkerJob:
    config: TrainingConfig
    event: threading.Event
    result: TrainingResult | None = None
    error: Exception | None = None


class TorchTrainingBackend:
    def __init__(
        self,
        *,
        train_data_meta_path: str | Path,
        vocab_size: int,
        context_length: int = 512,
        device: str = "cpu",
        devices: list[str] | None = None,
        mixed_precision: str = "off",
        activation_checkpointing: bool = False,
        preload_dataset: bool = True,
        max_steps_cap: int | None = None,
        runners: list[RunnerProtocol] | None = None,
    ) -> None:
        self.train_data_meta_path = Path(train_data_meta_path).expanduser()
        if runners is not None:
            self.runners = list(runners)
            self.device_specs = [str(runner.device) for runner in self.runners]
        else:
            self.device_specs = devices or _resolve_device_specs(device)
            shared_dataset = (
                TokenizedDataset.from_meta(self.train_data_meta_path)
                if preload_dataset
                else None
            )
            self.runners = [
                TrainingRunner(
                    train_data_meta_path=self.train_data_meta_path,
                    vocab_size=vocab_size,
                    context_length=context_length,
                    device=device_spec,
                    mixed_precision=mixed_precision,
                    activation_checkpointing=activation_checkpointing,
                    preload_dataset=preload_dataset,
                    dataset=shared_dataset,
                    max_steps_cap=max_steps_cap,
                )
                for device_spec in self.device_specs
            ]
        self.runner = self.runners[0]
        self.max_concurrency = len(self.runners)
        self._queue: deque[_WorkerJob] = deque()
        self._condition = threading.Condition()
        self._shutdown_requested = False
        self._running_jobs = 0
        self._workers = [
            threading.Thread(
                target=self._worker_loop,
                args=(index, runner),
                daemon=True,
                name=f"training-backend-worker-{index}",
            )
            for index, runner in enumerate(self.runners)
        ]
        for worker in self._workers:
            worker.start()

    def _worker_loop(self, worker_index: int, runner: RunnerProtocol) -> None:
        while True:
            with self._condition:
                while not self._queue:
                    if self._shutdown_requested:
                        return
                    self._condition.wait()
                job = self._queue.popleft()
                self._running_jobs += 1

            try:
                logger.info(
                    "backend worker %s on %s starting job for api_key=%s d_model=%s layers=%s heads=%s batch=%s lr=%s flops=%s",
                    worker_index,
                    runner.device,
                    job.config.api_key,
                    job.config.d_model,
                    job.config.num_layers,
                    job.config.num_heads,
                    job.config.batch_size,
                    job.config.learning_rate,
                    job.config.train_flops,
                )
                run_result = runner.run(job.config)
                job.result = TrainingResult(loss=run_result.loss)
                logger.info(
                    "backend worker %s on %s finished job for api_key=%s with loss=%s",
                    worker_index,
                    runner.device,
                    job.config.api_key,
                    job.result.loss,
                )
            except Exception as exc:  # noqa: BLE001 - propagate through the waiting request
                job.error = exc
                logger.exception(
                    "backend worker %s on %s failed job for api_key=%s",
                    worker_index,
                    runner.device,
                    job.config.api_key,
                )
            finally:
                job.event.set()
                with self._condition:
                    self._running_jobs -= 1
                    if self._shutdown_requested and self._running_jobs == 0 and not self._queue:
                        self._condition.notify_all()

    def run(self, config: TrainingConfig) -> TrainingResult:
        job = _WorkerJob(config=config, event=threading.Event())
        with self._condition:
            if self._shutdown_requested:
                raise BackendUnavailableError("Training backend is shutting down.")
            self._queue.append(job)
            self._condition.notify()

        job.event.wait()
        if job.error is not None:
            if isinstance(job.error, RuntimeError):
                message = str(job.error).lower()
                if "out of memory" in message:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise TrainingOOMError(
                        "Training run ran out of memory for this configuration."
                    ) from job.error
            raise job.error
        assert job.result is not None
        return job.result

    def begin_shutdown(self) -> None:
        with self._condition:
            if self._shutdown_requested:
                return
            self._shutdown_requested = True
            while self._queue:
                job = self._queue.popleft()
                job.error = BackendUnavailableError("Training backend is shutting down.")
                job.event.set()
            self._condition.notify_all()

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        with self._condition:
            if timeout is None:
                while self._running_jobs > 0 or self._queue:
                    self._condition.wait()
                return True

            deadline = time.monotonic() + timeout
            while self._running_jobs > 0 or self._queue:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True


def _parse_devices(raw_value: str | None) -> list[str]:
    if raw_value is None or raw_value.strip() == "":
        return []
    return [token.strip() for token in raw_value.split(",") if token.strip()]


def _resolve_device_specs(device: str) -> list[str]:
    normalized = device.strip()
    if normalized == "cuda":
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 0:
                return [f"cuda:{index}" for index in range(count)]
        return ["cpu"]
    return [normalized]


def get_training_backend() -> TrainingBackend:
    train_data_meta_path = os.environ.get("CS336_TRAIN_DATA_META_PATH")
    vocab_size = os.environ.get("CS336_VOCAB_SIZE")
    if not train_data_meta_path or not vocab_size:
        return PlaceholderTrainingBackend()

    context_length = int(os.environ.get("CS336_CONTEXT_LENGTH", "512"))
    device = os.environ.get("CS336_DEVICE", "cpu")
    devices = _parse_devices(os.environ.get("CS336_DEVICES"))
    default_mixed_precision = "bf16" if device.startswith("cuda") else "off"
    mixed_precision = os.environ.get("CS336_MIXED_PRECISION", default_mixed_precision)
    activation_checkpointing_raw = os.environ.get(
        "CS336_ACTIVATION_CHECKPOINTING",
        "1" if device.startswith("cuda") else "0",
    )
    activation_checkpointing = activation_checkpointing_raw == "1"
    preload_dataset_raw = os.environ.get("CS336_PRELOAD_DATASET", "1")
    preload_dataset = preload_dataset_raw == "1"
    max_steps_cap_raw = os.environ.get("CS336_MAX_STEPS_CAP")
    max_steps_cap = int(max_steps_cap_raw) if max_steps_cap_raw else None
    return TorchTrainingBackend(
        train_data_meta_path=train_data_meta_path,
        vocab_size=int(vocab_size),
        context_length=context_length,
        device=device,
        devices=devices or None,
        mixed_precision=mixed_precision,
        activation_checkpointing=activation_checkpointing,
        preload_dataset=preload_dataset,
        max_steps_cap=max_steps_cap,
    )
