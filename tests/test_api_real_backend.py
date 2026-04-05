from __future__ import annotations

import json
from array import array
from pathlib import Path

from fastapi.testclient import TestClient

from cs336_scaling.api_backend import TorchTrainingBackend
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
