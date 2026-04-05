from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from cs336_scaling.api_backend import BackendUnavailableError, TrainingResult
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
