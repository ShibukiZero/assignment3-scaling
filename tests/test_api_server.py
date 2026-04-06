from __future__ import annotations

import threading
from pathlib import Path

from fastapi.testclient import TestClient

from cs336_scaling.api_backend import BackendUnavailableError, TrainingOOMError, TrainingResult
from cs336_scaling.api_contract import TrainingConfig, build_training_config
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


class BlockingBackend:
    def __init__(self, loss: float = 9.9) -> None:
        self.loss = loss
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    def run(self, config: TrainingConfig) -> TrainingResult:
        with self._lock:
            self.calls += 1
            self.started.set()
        self.release.wait(timeout=5.0)
        return TrainingResult(loss=self.loss)


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


def test_loss_rejects_new_queries_that_exceed_scaling_law_budget_cap(tmp_path: Path) -> None:
    client, backend = make_client(tmp_path, loss=5.0)

    first = client.get(
        "/loss",
        params=build_query(
            api_key="cap-key",
            learning_rate=1e-3,
            train_flops=int(1e18),
        ),
    )
    second = client.get(
        "/loss",
        params=build_query(
            api_key="cap-key",
            learning_rate=9e-4,
            train_flops=int(1e18),
        ),
    )
    third = client.get(
        "/loss",
        params=build_query(
            api_key="cap-key",
            learning_rate=8e-4,
            train_flops=int(1e13),
        ),
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 422
    assert third.json() == {
        "detail": {
            "message": "API key would exceed the scaling law FLOPs budget cap of 2000000000000000000: cap-key"
        }
    }
    assert len(backend.calls) == 2


def test_loss_allows_cached_queries_even_after_budget_cap_is_reached(tmp_path: Path) -> None:
    client, backend = make_client(tmp_path, loss=8.0)
    cached_params = build_query(
        api_key="cap-cache-key",
        learning_rate=1e-3,
        train_flops=int(1e18),
    )

    first = client.get("/loss", params=cached_params)
    second = client.get(
        "/loss",
        params=build_query(
            api_key="cap-cache-key",
            learning_rate=9e-4,
            train_flops=int(1e18),
        ),
    )
    cached_again = client.get("/loss", params=cached_params)

    assert first.status_code == 200
    assert second.status_code == 200
    assert cached_again.status_code == 200
    assert cached_again.json() == {"loss": 8.0, "total_flops_used": float(int(2e18))}
    assert len(backend.calls) == 2


def test_runtime_serializes_same_api_key_queries_to_avoid_duplicate_training(tmp_path: Path) -> None:
    backend = BlockingBackend(loss=4.4)
    runtime = ApiRuntime(
        db_path=tmp_path / "api.db",
        backend=backend,
        accept_all_keys=True,
    )
    config = build_training_config(
        api_key="thread-key",
        d_model=512,
        num_layers=8,
        num_heads=8,
        batch_size=128,
        learning_rate=1e-3,
        train_flops=int(1e16),
    )

    results: list[tuple[float, bool]] = []

    def worker() -> None:
        result, cached = runtime.run_training_query(config)
        results.append((result.loss, cached))

    thread_a = threading.Thread(target=worker)
    thread_b = threading.Thread(target=worker)
    thread_a.start()
    backend.started.wait(timeout=5.0)
    thread_b.start()
    backend.release.set()
    thread_a.join(timeout=5.0)
    thread_b.join(timeout=5.0)

    assert backend.calls == 1
    assert sorted(results) == [(4.4, False), (4.4, True)]


def test_runtime_queues_distinct_configs_globally_while_one_training_is_running(tmp_path: Path) -> None:
    backend = BlockingBackend(loss=6.6)
    runtime = ApiRuntime(
        db_path=tmp_path / "api.db",
        backend=backend,
        accept_all_keys=True,
    )
    config_a = build_training_config(
        api_key="key-a",
        d_model=512,
        num_layers=8,
        num_heads=8,
        batch_size=128,
        learning_rate=1e-3,
        train_flops=int(1e16),
    )
    config_b = build_training_config(
        api_key="key-b",
        d_model=512,
        num_layers=8,
        num_heads=8,
        batch_size=128,
        learning_rate=9e-4,
        train_flops=int(3e16),
    )

    results: list[tuple[float, bool]] = []

    def worker(config: TrainingConfig) -> None:
        result, cached = runtime.run_training_query(config)
        results.append((result.loss, cached))

    thread_a = threading.Thread(target=worker, args=(config_a,))
    thread_b = threading.Thread(target=worker, args=(config_b,))
    thread_a.start()
    backend.started.wait(timeout=5.0)
    thread_b.start()

    # While the first training job is still blocked, the second should remain queued.
    assert backend.calls == 1

    backend.release.set()
    thread_a.join(timeout=5.0)
    thread_b.join(timeout=5.0)

    assert backend.calls == 2
    assert sorted(results) == [(6.6, False), (6.6, False)]


def test_queued_request_rechecks_budget_after_waiting(tmp_path: Path) -> None:
    backend = BlockingBackend(loss=7.7)
    runtime = ApiRuntime(
        db_path=tmp_path / "api.db",
        backend=backend,
        accept_all_keys=True,
    )
    config_a = build_training_config(
        api_key="queued-cap-key",
        d_model=512,
        num_layers=8,
        num_heads=8,
        batch_size=128,
        learning_rate=1e-3,
        train_flops=int(1e18),
    )
    config_b = build_training_config(
        api_key="queued-cap-key",
        d_model=512,
        num_layers=8,
        num_heads=8,
        batch_size=128,
        learning_rate=9e-4,
        train_flops=int(1e18),
    )
    config_c = build_training_config(
        api_key="queued-cap-key",
        d_model=512,
        num_layers=8,
        num_heads=8,
        batch_size=128,
        learning_rate=8e-4,
        train_flops=int(1e13),
    )

    errors: list[str] = []

    def worker(config: TrainingConfig) -> None:
        try:
            runtime.run_training_query(config)
        except Exception as exc:  # noqa: BLE001 - assertion helper in tests
            errors.append(str(exc))

    thread_a = threading.Thread(target=worker, args=(config_a,))
    thread_b = threading.Thread(target=worker, args=(config_b,))
    thread_c = threading.Thread(target=worker, args=(config_c,))
    thread_a.start()
    backend.started.wait(timeout=5.0)
    thread_b.start()
    thread_c.start()
    backend.release.set()
    thread_a.join(timeout=5.0)
    thread_b.join(timeout=5.0)
    thread_c.join(timeout=5.0)

    assert backend.calls == 2
    assert errors == [
        "422: {'message': 'API key would exceed the scaling law FLOPs budget cap of 2000000000000000000: queued-cap-key'}"
    ]


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
