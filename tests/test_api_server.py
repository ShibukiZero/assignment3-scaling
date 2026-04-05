from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from cs336_scaling.api_backend import BackendUnavailableError, TrainingOOMError, TrainingResult
from cs336_scaling.api_contract import TrainingConfig
from cs336_scaling.api_server import ApiRuntime, create_app


class RecordingBackend:
    def __init__(self, loss: float = 1.2345) -> None:
        self.loss = loss
        self.calls: list[TrainingConfig] = []

    def run(self, config: TrainingConfig) -> TrainingResult:
        self.calls.append(config)
        return TrainingResult(loss=self.loss)


class UnavailableBackend:
    def run(self, config: TrainingConfig) -> TrainingResult:
        raise BackendUnavailableError("Training backend is not configured yet.")


class OOMBackend:
    def run(self, config: TrainingConfig) -> TrainingResult:
        raise TrainingOOMError("Training run ran out of memory for this configuration.")


def make_client(
    tmp_path: Path,
    *,
    accept_all_keys: bool = True,
    allowed_api_keys: set[str] | None = None,
    loss: float = 1.2345,
) -> tuple[TestClient, RecordingBackend]:
    backend = RecordingBackend(loss=loss)
    runtime = ApiRuntime(
        db_path=tmp_path / "api.db",
        backend=backend,
        allowed_api_keys=allowed_api_keys,
        accept_all_keys=accept_all_keys,
    )
    app = create_app(runtime)
    return TestClient(app), backend


def build_query(**overrides: object) -> dict[str, object]:
    query: dict[str, object] = {
        "api_key": "demo-key",
        "d_model": 512,
        "num_layers": 8,
        "num_heads": 8,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "train_flops": int(1e16),
    }
    query.update(overrides)
    return query


def test_loss_rejects_invalid_hyperparameter_with_assignment_style_message(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.get("/loss", params=build_query(d_model=9999))

    assert response.status_code == 404
    assert response.json() == {"detail": {"message": "d_model must be in range [64, 1024], got 9999"}}


def test_loss_rejects_num_layers_out_of_range(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.get("/loss", params=build_query(num_layers=1))

    assert response.status_code == 404
    assert response.json() == {"detail": {"message": "num_layers must be in range [2, 24], got 1"}}


def test_loss_rejects_num_heads_out_of_range(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.get("/loss", params=build_query(num_heads=32))

    assert response.status_code == 404
    assert response.json() == {"detail": {"message": "num_heads must be in range [2, 16], got 32"}}


def test_loss_rejects_invalid_batch_size(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.get("/loss", params=build_query(batch_size=64))

    assert response.status_code == 404
    assert response.json() == {"detail": {"message": "batch_size must be one of {128, 256}, got 64"}}


def test_loss_rejects_learning_rate_below_range(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.get("/loss", params=build_query(learning_rate=5e-5))

    assert response.status_code == 404
    assert response.json() == {
        "detail": {"message": "learning_rate must be in range [0.0001, 0.001], got 5e-05"}
    }


def test_loss_rejects_invalid_train_flops_value(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.get("/loss", params=build_query(train_flops=int(2e16)))

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "message": "train_flops must be one of [10000000000000, 30000000000000, 60000000000000, 100000000000000, 300000000000000, 600000000000000, 1000000000000000, 3000000000000000, 6000000000000000, 10000000000000000, 30000000000000000, 60000000000000000, 100000000000000000, 300000000000000000, 600000000000000000, 1000000000000000000], got 20000000000000000"
        }
    }


def test_loss_rejects_incompatible_d_model_and_num_heads(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.get("/loss", params=build_query(d_model=510, num_heads=8))

    assert response.status_code == 404
    assert response.json() == {
        "detail": {"message": "d_model must be divisible by num_heads, got d_model=510 and num_heads=8"}
    }


def test_loss_accepts_boundary_values(tmp_path: Path) -> None:
    client, backend = make_client(tmp_path, loss=3.21)

    response = client.get(
        "/loss",
        params=build_query(
            api_key="boundary-key",
            d_model=1024,
            num_layers=24,
            num_heads=16,
            batch_size=256,
            learning_rate=1e-4,
            train_flops=int(1e18),
        ),
    )

    assert response.status_code == 200
    assert response.json() == {"loss": 3.21, "total_flops_used": float(int(1e18))}
    assert len(backend.calls) == 1


def test_loss_response_fields_are_floats(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, loss=7.5)

    response = client.get("/loss", params=build_query(api_key="float-key"))

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["loss"], float)
    assert isinstance(payload["total_flops_used"], float)


def test_loss_uses_cache_and_does_not_double_count_flops(tmp_path: Path) -> None:
    client, backend = make_client(tmp_path, loss=6.789)
    params = build_query(api_key="cache-key", train_flops=int(3e16))

    first = client.get("/loss", params=params)
    second = client.get("/loss", params=params)
    history = client.get("/previous_runs", params={"api_key": "cache-key"})
    total = client.get("/total_flops_used", params={"api_key": "cache-key"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == {"loss": 6.789, "total_flops_used": float(int(3e16))}
    assert second.json() == {"loss": 6.789, "total_flops_used": float(int(3e16))}
    assert len(backend.calls) == 1

    assert history.status_code == 200
    assert history.json() == {
        "previous_runs": [
            {
                "d_model": 512,
                "num_layers": 8,
                "num_heads": 8,
                "batch_size": 128,
                "learning_rate": 1e-3,
                "train_flops": int(3e16),
                "loss": 6.789,
            }
        ]
    }
    assert total.status_code == 200
    assert total.json() == float(int(3e16))


def test_total_flops_used_requires_existing_history(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.get("/total_flops_used", params={"api_key": "fresh-key"})

    assert response.status_code == 422
    assert response.json() == {"detail": {"message": "API key has no queries yet: fresh-key"}}


def test_previous_runs_requires_existing_history(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.get("/previous_runs", params={"api_key": "fresh-key"})

    assert response.status_code == 422
    assert response.json() == {"detail": {"message": "API key has no queries yet: fresh-key"}}


def test_invalid_api_key_is_rejected_when_allowlist_mode_is_enabled(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, accept_all_keys=False, allowed_api_keys={"allowed-key"})

    response = client.get("/loss", params=build_query(api_key="forbidden-key"))

    assert response.status_code == 422
    assert response.json() == {
        "detail": {"message": "Invalid API key provided: forbidden-key"}
    }


def test_invalid_api_key_is_rejected_on_history_endpoints_in_allowlist_mode(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, accept_all_keys=False, allowed_api_keys={"allowed-key"})

    total_response = client.get("/total_flops_used", params={"api_key": "forbidden-key"})
    history_response = client.get("/previous_runs", params={"api_key": "forbidden-key"})

    assert total_response.status_code == 422
    assert total_response.json() == {
        "detail": {"message": "Invalid API key provided: forbidden-key"}
    }
    assert history_response.status_code == 422
    assert history_response.json() == {
        "detail": {"message": "Invalid API key provided: forbidden-key"}
    }


def test_same_config_under_different_api_keys_is_billed_separately(tmp_path: Path) -> None:
    client, backend = make_client(tmp_path, loss=2.5)
    params_a = build_query(api_key="key-a", train_flops=int(1e16))
    params_b = build_query(api_key="key-b", train_flops=int(1e16))

    response_a = client.get("/loss", params=params_a)
    response_b = client.get("/loss", params=params_b)
    total_a = client.get("/total_flops_used", params={"api_key": "key-a"})
    total_b = client.get("/total_flops_used", params={"api_key": "key-b"})

    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert len(backend.calls) == 2
    assert total_a.json() == float(int(1e16))
    assert total_b.json() == float(int(1e16))


def test_loss_returns_503_on_training_oom_without_recording_history(tmp_path: Path) -> None:
    runtime = ApiRuntime(
        db_path=tmp_path / "api.db",
        backend=OOMBackend(),
        accept_all_keys=True,
    )
    client = TestClient(create_app(runtime))

    response = client.get("/loss", params=build_query(api_key="oom-key"))
    history = client.get("/previous_runs", params={"api_key": "oom-key"})
    total = client.get("/total_flops_used", params={"api_key": "oom-key"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": {"message": "Training run ran out of memory for this configuration."}
    }
    assert history.status_code == 422
    assert total.status_code == 422


def test_service_remains_usable_after_training_oom(tmp_path: Path) -> None:
    class FlakyBackend:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, config: TrainingConfig) -> TrainingResult:
            self.calls += 1
            if self.calls == 1:
                raise TrainingOOMError("Training run ran out of memory for this configuration.")
            return TrainingResult(loss=4.2)

    runtime = ApiRuntime(
        db_path=tmp_path / "api.db",
        backend=FlakyBackend(),
        accept_all_keys=True,
    )
    client = TestClient(create_app(runtime))

    first = client.get("/loss", params=build_query(api_key="flaky-key", train_flops=int(1e15)))
    second = client.get("/loss", params=build_query(api_key="flaky-key", train_flops=int(3e15)))

    assert first.status_code == 503
    assert second.status_code == 200
    assert second.json() == {"loss": 4.2, "total_flops_used": float(int(3e15))}


def test_total_flops_accumulates_across_distinct_queries_for_one_key(tmp_path: Path) -> None:
    client, backend = make_client(tmp_path, loss=1.5)

    first = client.get("/loss", params=build_query(api_key="sum-key", train_flops=int(1e15)))
    second = client.get(
        "/loss",
        params=build_query(
            api_key="sum-key",
            d_model=768,
            num_layers=12,
            num_heads=12,
            train_flops=int(3e15),
        ),
    )
    total = client.get("/total_flops_used", params={"api_key": "sum-key"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(backend.calls) == 2
    assert total.status_code == 200
    assert total.json() == float(int(4e15))


def test_previous_runs_preserves_query_order(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, loss=4.2)

    first = build_query(api_key="history-key", d_model=256, num_layers=4, num_heads=4, train_flops=int(1e15))
    second = build_query(
        api_key="history-key",
        d_model=512,
        num_layers=8,
        num_heads=8,
        batch_size=256,
        learning_rate=5e-4,
        train_flops=int(3e15),
    )
    client.get("/loss", params=first)
    client.get("/loss", params=second)

    history = client.get("/previous_runs", params={"api_key": "history-key"})

    assert history.status_code == 200
    assert history.json() == {
        "previous_runs": [
            {
                "d_model": 256,
                "num_layers": 4,
                "num_heads": 4,
                "batch_size": 128,
                "learning_rate": 1e-3,
                "train_flops": int(1e15),
                "loss": 4.2,
            },
            {
                "d_model": 512,
                "num_layers": 8,
                "num_heads": 8,
                "batch_size": 256,
                "learning_rate": 5e-4,
                "train_flops": int(3e15),
                "loss": 4.2,
            },
        ]
    }


def test_previous_runs_rows_include_all_expected_public_fields(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path, loss=8.8)

    client.get("/loss", params=build_query(api_key="fields-key"))
    history = client.get("/previous_runs", params={"api_key": "fields-key"})

    assert history.status_code == 200
    row = history.json()["previous_runs"][0]
    assert set(row.keys()) == {
        "d_model",
        "num_layers",
        "num_heads",
        "batch_size",
        "learning_rate",
        "train_flops",
        "loss",
    }


def test_loss_returns_503_when_backend_is_not_ready(tmp_path: Path) -> None:
    runtime = ApiRuntime(
        db_path=tmp_path / "api.db",
        backend=UnavailableBackend(),
        accept_all_keys=True,
    )
    client = TestClient(create_app(runtime))

    response = client.get("/loss", params=build_query(api_key="stub-key"))

    assert response.status_code == 503
    assert response.json() == {"detail": {"message": "Training backend is not configured yet."}}
