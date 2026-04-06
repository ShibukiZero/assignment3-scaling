from __future__ import annotations

import json
import threading
from array import array
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from cs336_scaling.api_backend import TrainingResult, TorchTrainingBackend
from cs336_scaling.api_contract import build_training_config
from cs336_scaling.api_server import ApiRuntime, create_app


def write_backend_corpus(tmp_path: Path) -> Path:
    output_prefix = tmp_path / "backend_train"
    ids_path = output_prefix.with_suffix(".bin")
    idx_path = output_prefix.with_suffix(".idx")
    meta_path = output_prefix.with_suffix(".meta.json")

    tokens = [1, 2, 3, 4, 5, 6, 7, 8] * 32
    array("H", tokens).tofile(ids_path.open("wb"))
    array("Q", [0, len(tokens)]).tofile(idx_path.open("wb"))
    meta_path.write_text(
        json.dumps(
            {
                "dtype": "uint16",
                "num_documents": 1,
                "num_tokens": len(tokens),
                "ids_path": str(ids_path),
                "idx_path": str(idx_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return meta_path


def build_client(tmp_path: Path) -> TestClient:
    meta_path = write_backend_corpus(tmp_path)
    backend = TorchTrainingBackend(
        train_data_meta_path=meta_path,
        vocab_size=32,
        context_length=8,
        device="cpu",
        max_steps_cap=1,
    )
    runtime = ApiRuntime(
        db_path=tmp_path / "api.db",
        backend=backend,
        accept_all_keys=True,
    )
    return TestClient(create_app(runtime))


class FakeRunner:
    def __init__(self, device: str, *, loss: float = 2.5) -> None:
        self.device = torch.device(device)
        self.loss = loss
        self.calls = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def run(self, config) -> object:
        with self._condition:
            self.calls += 1
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            self.started.set()
            self._condition.notify_all()
        self.release.wait(timeout=5.0)
        with self._condition:
            self.active_calls -= 1
            self._condition.notify_all()
        return type("RunResult", (), {"loss": self.loss})()

    def wait_until_started(self, timeout: float = 5.0) -> bool:
        return self.started.wait(timeout=timeout)


def test_real_backend_cache_miss_trains_and_cache_hit_reuses_result(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    params = {
        "api_key": "real-backend-key",
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 2,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "train_flops": int(1e13),
    }

    first = client.get("/loss", params=params)
    second = client.get("/loss", params=params)
    history = client.get("/previous_runs", params={"api_key": "real-backend-key"})
    total = client.get("/total_flops_used", params={"api_key": "real-backend-key"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert isinstance(first.json()["loss"], float)
    assert first.json() == second.json()
    assert history.status_code == 200
    assert len(history.json()["previous_runs"]) == 1
    assert total.status_code == 200
    assert total.json() == float(int(1e13))


def test_get_training_backend_defaults_to_bf16_on_cuda(monkeypatch, tmp_path: Path) -> None:
    from cs336_scaling.api_backend import get_training_backend

    meta_path = write_backend_corpus(tmp_path)
    monkeypatch.setenv("CS336_TRAIN_DATA_META_PATH", str(meta_path))
    monkeypatch.setenv("CS336_VOCAB_SIZE", "32")
    monkeypatch.setenv("CS336_DEVICE", "cuda")
    monkeypatch.delenv("CS336_MIXED_PRECISION", raising=False)

    backend = get_training_backend()

    assert isinstance(backend, TorchTrainingBackend)
    assert backend.runner.mixed_precision == "bf16"


def test_get_training_backend_allows_explicit_mixed_precision_override(monkeypatch, tmp_path: Path) -> None:
    from cs336_scaling.api_backend import get_training_backend

    meta_path = write_backend_corpus(tmp_path)
    monkeypatch.setenv("CS336_TRAIN_DATA_META_PATH", str(meta_path))
    monkeypatch.setenv("CS336_VOCAB_SIZE", "32")
    monkeypatch.setenv("CS336_DEVICE", "cuda")
    monkeypatch.setenv("CS336_MIXED_PRECISION", "fp16")

    backend = get_training_backend()

    assert isinstance(backend, TorchTrainingBackend)
    assert backend.runner.mixed_precision == "fp16"


def test_get_training_backend_defaults_to_activation_checkpointing_on_cuda(monkeypatch, tmp_path: Path) -> None:
    from cs336_scaling.api_backend import get_training_backend

    meta_path = write_backend_corpus(tmp_path)
    monkeypatch.setenv("CS336_TRAIN_DATA_META_PATH", str(meta_path))
    monkeypatch.setenv("CS336_VOCAB_SIZE", "32")
    monkeypatch.setenv("CS336_DEVICE", "cuda")
    monkeypatch.delenv("CS336_ACTIVATION_CHECKPOINTING", raising=False)

    backend = get_training_backend()

    assert isinstance(backend, TorchTrainingBackend)
    assert backend.runner.activation_checkpointing is True


def test_get_training_backend_allows_disabling_activation_checkpointing(monkeypatch, tmp_path: Path) -> None:
    from cs336_scaling.api_backend import get_training_backend

    meta_path = write_backend_corpus(tmp_path)
    monkeypatch.setenv("CS336_TRAIN_DATA_META_PATH", str(meta_path))
    monkeypatch.setenv("CS336_VOCAB_SIZE", "32")
    monkeypatch.setenv("CS336_DEVICE", "cuda")
    monkeypatch.setenv("CS336_ACTIVATION_CHECKPOINTING", "0")

    backend = get_training_backend()

    assert isinstance(backend, TorchTrainingBackend)
    assert backend.runner.activation_checkpointing is False


def test_get_training_backend_defaults_to_preloading_dataset(monkeypatch, tmp_path: Path) -> None:
    from cs336_scaling.api_backend import get_training_backend

    meta_path = write_backend_corpus(tmp_path)
    monkeypatch.setenv("CS336_TRAIN_DATA_META_PATH", str(meta_path))
    monkeypatch.setenv("CS336_VOCAB_SIZE", "32")
    monkeypatch.delenv("CS336_PRELOAD_DATASET", raising=False)

    backend = get_training_backend()

    assert isinstance(backend, TorchTrainingBackend)
    assert backend.runner.preload_dataset is True
    assert backend.runner.dataset is not None


def test_get_training_backend_allows_disabling_dataset_preload(monkeypatch, tmp_path: Path) -> None:
    from cs336_scaling.api_backend import get_training_backend

    meta_path = write_backend_corpus(tmp_path)
    monkeypatch.setenv("CS336_TRAIN_DATA_META_PATH", str(meta_path))
    monkeypatch.setenv("CS336_VOCAB_SIZE", "32")
    monkeypatch.setenv("CS336_PRELOAD_DATASET", "0")

    backend = get_training_backend()

    assert isinstance(backend, TorchTrainingBackend)
    assert backend.runner.preload_dataset is False
    assert backend.runner.dataset is None


def test_get_training_backend_expands_cuda_to_all_visible_devices(monkeypatch, tmp_path: Path) -> None:
    from cs336_scaling.api_backend import get_training_backend

    meta_path = write_backend_corpus(tmp_path)
    monkeypatch.setenv("CS336_TRAIN_DATA_META_PATH", str(meta_path))
    monkeypatch.setenv("CS336_VOCAB_SIZE", "32")
    monkeypatch.setenv("CS336_DEVICE", "cuda")
    monkeypatch.delenv("CS336_DEVICES", raising=False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 3)

    backend = get_training_backend()

    assert isinstance(backend, TorchTrainingBackend)
    assert backend.device_specs == ["cuda:0", "cuda:1", "cuda:2"]
    assert backend.max_concurrency == 3


def test_torch_training_backend_runs_two_jobs_concurrently_across_two_workers(tmp_path: Path) -> None:
    meta_path = write_backend_corpus(tmp_path)
    runner_a = FakeRunner("cpu", loss=3.1)
    runner_b = FakeRunner("cpu", loss=4.2)
    backend = TorchTrainingBackend(
        train_data_meta_path=meta_path,
        vocab_size=32,
        context_length=8,
        runners=[runner_a, runner_b],
    )
    config_a = build_training_config(
        api_key="worker-a",
        d_model=64,
        num_layers=2,
        num_heads=2,
        batch_size=128,
        learning_rate=1e-3,
        train_flops=int(1e13),
    )
    config_b = build_training_config(
        api_key="worker-b",
        d_model=64,
        num_layers=2,
        num_heads=2,
        batch_size=128,
        learning_rate=9e-4,
        train_flops=int(3e13),
    )

    results: list[TrainingResult] = []

    def worker(config) -> None:
        results.append(backend.run(config))

    thread_a = threading.Thread(target=worker, args=(config_a,))
    thread_b = threading.Thread(target=worker, args=(config_b,))
    thread_a.start()
    thread_b.start()
    assert runner_a.wait_until_started() or runner_b.wait_until_started()
    # Wait until both jobs have been picked up by the two workers.
    for _ in range(50):
        if runner_a.calls == 1 and runner_b.calls == 1:
            break
        threading.Event().wait(0.05)
    runner_a.release.set()
    runner_b.release.set()
    thread_a.join(timeout=5.0)
    thread_b.join(timeout=5.0)

    assert runner_a.calls == 1
    assert runner_b.calls == 1
    assert len(results) == 2
