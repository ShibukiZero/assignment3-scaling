from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from cs336_scaling.api_backend import BackendUnavailableError, TrainingOOMError, TrainingResult
from cs336_scaling.api_contract import TrainingConfig, build_training_config
from cs336_scaling.api_server import ApiRuntime, create_app
from cs336_scaling.api_store import ApiStore


class RecordingBackend:
    max_concurrency = 1

    def __init__(self, loss: float = 1.2345) -> None:
        self.loss = loss
        self.calls: list[TrainingConfig] = []

    def run(self, config: TrainingConfig) -> TrainingResult:
        self.calls.append(config)
        return TrainingResult(loss=self.loss)

    def begin_shutdown(self) -> None:
        return None

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        return True


class UnavailableBackend:
    max_concurrency = 1

    def run(self, config: TrainingConfig) -> TrainingResult:
        raise BackendUnavailableError("Training backend is not configured yet.")

    def begin_shutdown(self) -> None:
        return None

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        return True


class OOMBackend:
    max_concurrency = 1

    def run(self, config: TrainingConfig) -> TrainingResult:
        raise TrainingOOMError("Training run ran out of memory for this configuration.")

    def begin_shutdown(self) -> None:
        return None

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        return True


class BlockingBackend:
    max_concurrency = 1

    def __init__(self, loss: float = 9.9) -> None:
        self.loss = loss
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()
        self._calls_changed = threading.Condition(self._lock)

    def run(self, config: TrainingConfig) -> TrainingResult:
        with self._lock:
            self.calls += 1
            self.started.set()
            self._calls_changed.notify_all()
        self.release.wait(timeout=5.0)
        return TrainingResult(loss=self.loss)

    def wait_for_calls(self, expected_calls: int, timeout: float = 5.0) -> bool:
        with self._lock:
            return self._calls_changed.wait_for(lambda: self.calls >= expected_calls, timeout=timeout)

    def begin_shutdown(self) -> None:
        return None

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        return True


class ConcurrentBlockingBackend:
    max_concurrency = 2

    def __init__(self, *, loss: float = 3.3) -> None:
        self.loss = loss
        self.calls = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()
        self._active_changed = threading.Condition(self._lock)

    def run(self, config: TrainingConfig) -> TrainingResult:
        with self._lock:
            self.calls += 1
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            self.started.set()
            self._active_changed.notify_all()
        self.release.wait(timeout=5.0)
        with self._lock:
            self.active_calls -= 1
            self._active_changed.notify_all()
        return TrainingResult(loss=self.loss)

    def wait_for_active_calls(self, expected_calls: int, timeout: float = 5.0) -> bool:
        with self._lock:
            return self._active_changed.wait_for(
                lambda: self.active_calls >= expected_calls or self.max_active_calls >= expected_calls,
                timeout=timeout,
            )

    def begin_shutdown(self) -> None:
        return None

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        return True


class LockstepBackend:
    max_concurrency = 1

    def __init__(self, loss: float = 4.5) -> None:
        self.loss = loss
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, config: TrainingConfig) -> TrainingResult:
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=5.0)
        return TrainingResult(loss=self.loss)

    def begin_shutdown(self) -> None:
        return None

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        return True


class ShutdownAwareBackend:
    max_concurrency = 1

    def __init__(self, *, loss: float = 5.5) -> None:
        self.loss = loss
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self._queue = threading.Condition()
        self._jobs: list[tuple[TrainingConfig, threading.Event, dict[str, object]]] = []
        self._running = 0
        self._shutdown = False
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            with self._queue:
                while not self._jobs:
                    if self._shutdown:
                        return
                    self._queue.wait()
                config, event, state = self._jobs.pop(0)
                self._running += 1
            self.calls += 1
            self.started.set()
            self.release.wait(timeout=5.0)
            state["result"] = TrainingResult(loss=self.loss)
            state["running"] = True
            event.set()
            with self._queue:
                self._running -= 1
                if self._shutdown and self._running == 0 and not self._jobs:
                    self._queue.notify_all()

    def run(self, config: TrainingConfig) -> TrainingResult:
        if self._shutdown:
            raise BackendUnavailableError("Training backend is shutting down.")
        event = threading.Event()
        state: dict[str, object] = {}
        with self._queue:
            if self._shutdown:
                raise BackendUnavailableError("Training backend is shutting down.")
            self._jobs.append((config, event, state))
            self._queue.notify_all()
        event.wait(timeout=5.0)
        result = state.get("result")
        error = state.get("error")
        if error is not None:
            raise error  # type: ignore[misc]
        assert isinstance(result, TrainingResult)
        return result

    def begin_shutdown(self) -> None:
        with self._queue:
            self._shutdown = True
            while self._jobs:
                _, event, state = self._jobs.pop(0)
                state["error"] = BackendUnavailableError("Training backend is shutting down.")
                event.set()
            self._queue.notify_all()

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        with self._queue:
            if timeout is None:
                while self._running > 0 or self._jobs:
                    self._queue.wait()
                return True
            return self._running == 0 and not self._jobs


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


def test_loss_rejects_empty_api_key_as_client_error(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    response = client.get("/loss", params=build_query(api_key=""))

    assert response.status_code == 422
    assert response.json() == {"detail": {"message": "api_key must be provided."}}


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


def test_history_endpoints_reject_empty_api_key_as_client_error(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)

    total_response = client.get("/total_flops_used", params={"api_key": ""})
    history_response = client.get("/previous_runs", params={"api_key": ""})

    assert total_response.status_code == 422
    assert total_response.json() == {"detail": {"message": "api_key must be provided."}}
    assert history_response.status_code == 422
    assert history_response.json() == {"detail": {"message": "api_key must be provided."}}


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


def test_runtime_rechecks_cache_inside_lock_after_inflight_run_completes(tmp_path: Path) -> None:
    backend = LockstepBackend(loss=5.6)
    runtime = ApiRuntime(
        db_path=tmp_path / "api.db",
        backend=backend,
        accept_all_keys=True,
    )
    config = build_training_config(
        api_key="late-cache-key",
        d_model=512,
        num_layers=8,
        num_heads=8,
        batch_size=128,
        learning_rate=1e-3,
        train_flops=int(1e16),
    )

    first_result: list[tuple[float, bool]] = []

    def first_worker() -> None:
        result, cached = runtime.run_training_query(config)
        first_result.append((result.loss, cached))

    thread = threading.Thread(target=first_worker)
    thread.start()
    assert backend.started.wait(timeout=5.0)

    # Simulate a second request that missed the cache earlier, then resumes after
    # the first owner has already completed and cleared the inflight entry.
    cached_before_lock = runtime.store.get_run(config)
    assert cached_before_lock is None

    backend.release.set()
    thread.join(timeout=5.0)
    assert first_result == [(5.6, False)]

    second_result, second_cached = runtime.run_training_query(config)

    assert backend.calls == 1
    assert second_result.loss == 5.6
    assert second_cached is True


def test_runtime_allows_distinct_configs_to_enter_backend_concurrently(tmp_path: Path) -> None:
    backend = ConcurrentBlockingBackend(loss=6.6)
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
    backend.wait_for_active_calls(2, timeout=5.0)
    backend.release.set()
    thread_a.join(timeout=5.0)
    thread_b.join(timeout=5.0)

    assert backend.calls == 2
    assert backend.max_active_calls == 2
    assert sorted(results) == [(6.6, False), (6.6, False)]


def test_runtime_counts_reserved_flops_when_validating_new_queries(tmp_path: Path) -> None:
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
        train_flops=int(1e17),
    )

    errors: list[str] = []

    def worker(config: TrainingConfig) -> None:
        try:
            runtime.run_training_query(config)
        except Exception as exc:  # noqa: BLE001 - assertion helper in tests
            errors.append(str(exc))

    thread_a = threading.Thread(target=worker, args=(config_a,))
    thread_b = threading.Thread(target=worker, args=(config_b,))
    thread_a.start()
    backend.started.wait(timeout=5.0)
    thread_b.start()
    backend.wait_for_calls(2, timeout=5.0)
    deadline = time.monotonic() + 5.0
    while runtime.store.get_active_reserved_flops("queued-cap-key") < int(2e18):
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for reserved FLOPs to reach the expected value.")
        time.sleep(0.01)
    thread_c = threading.Thread(target=worker, args=(config_c,))
    thread_c.start()
    thread_c.join(timeout=5.0)
    assert not thread_c.is_alive(), "The capped request should fail before earlier jobs are released."
    backend.release.set()
    thread_a.join(timeout=5.0)
    thread_b.join(timeout=5.0)

    assert backend.calls == 2
    assert errors == [
        "422: {'message': 'API key would exceed the scaling law FLOPs budget cap of 2000000000000000000: queued-cap-key'}"
    ]


def test_runtime_rejects_new_queries_after_shutdown_begins(tmp_path: Path) -> None:
    runtime = ApiRuntime(
        db_path=tmp_path / "api.db",
        backend=RecordingBackend(),
        accept_all_keys=True,
    )
    client = TestClient(create_app(runtime))

    runtime.begin_shutdown()
    response = client.get("/loss", params=build_query(api_key="shutdown-key"))

    assert response.status_code == 503
    assert response.json() == {"detail": {"message": "Training backend is shutting down."}}


def test_shutdown_rejects_pending_requests_but_allows_running_request_to_finish(tmp_path: Path) -> None:
    backend = ShutdownAwareBackend(loss=9.1)
    runtime = ApiRuntime(
        db_path=tmp_path / "api.db",
        backend=backend,
        accept_all_keys=True,
    )
    config_a = build_training_config(
        api_key="shutdown-queue-key",
        d_model=512,
        num_layers=8,
        num_heads=8,
        batch_size=128,
        learning_rate=1e-3,
        train_flops=int(1e16),
    )
    config_b = build_training_config(
        api_key="shutdown-queue-key",
        d_model=768,
        num_layers=12,
        num_heads=12,
        batch_size=128,
        learning_rate=9e-4,
        train_flops=int(1e16),
    )

    results: list[tuple[float, bool]] = []
    errors: list[str] = []

    def worker(config: TrainingConfig) -> None:
        try:
            result, cached = runtime.run_training_query(config)
            results.append((result.loss, cached))
        except Exception as exc:  # noqa: BLE001 - assertion helper in tests
            errors.append(str(exc))

    thread_a = threading.Thread(target=worker, args=(config_a,))
    thread_b = threading.Thread(target=worker, args=(config_b,))
    thread_a.start()
    backend.started.wait(timeout=5.0)
    thread_b.start()
    runtime.begin_shutdown()
    backend.release.set()
    thread_a.join(timeout=5.0)
    thread_b.join(timeout=5.0)

    assert results == [(9.1, False)]
    assert errors == ["Training backend is shutting down."]


def test_runtime_recovers_incomplete_reservations_on_startup(tmp_path: Path) -> None:
    db_path = tmp_path / "api.db"
    store = ApiStore(db_path)
    config = build_training_config(
        api_key="recover-key",
        d_model=512,
        num_layers=8,
        num_heads=8,
        batch_size=128,
        learning_rate=1e-3,
        train_flops=int(1e18),
    )
    reservation_id = store.create_reservation(config)
    store.mark_reservation_running(reservation_id)

    runtime = ApiRuntime(
        db_path=db_path,
        backend=RecordingBackend(loss=2.2),
        accept_all_keys=True,
    )

    recovered = runtime.store.get_recovered_reservations("recover-key")

    assert len(recovered) == 1
    assert recovered[0]["status"] == "FAILED_RECOVERED"
    assert recovered[0]["train_flops"] == int(1e18)
    assert runtime.store.get_active_reserved_flops("recover-key") == 0


def test_recovered_reservations_do_not_block_future_budget_usage(tmp_path: Path) -> None:
    db_path = tmp_path / "api.db"
    store = ApiStore(db_path)
    config = build_training_config(
        api_key="recover-budget-key",
        d_model=512,
        num_layers=8,
        num_heads=8,
        batch_size=128,
        learning_rate=1e-3,
        train_flops=int(1e18),
    )
    reservation_id = store.create_reservation(config)
    store.mark_reservation_running(reservation_id)

    runtime = ApiRuntime(
        db_path=db_path,
        backend=RecordingBackend(loss=3.3),
        accept_all_keys=True,
    )

    runtime.ensure_budget_available("recover-budget-key", int(2e18))


def test_finalize_successful_run_clears_reserved_flops_when_completed_flops_are_recorded(tmp_path: Path) -> None:
    store = ApiStore(tmp_path / "api.db")
    config = build_training_config(
        api_key="finalize-key",
        d_model=512,
        num_layers=8,
        num_heads=8,
        batch_size=128,
        learning_rate=1e-3,
        train_flops=int(1e18),
    )
    reservation_id = store.create_reservation(config)
    store.mark_reservation_running(reservation_id)

    assert store.get_active_reserved_flops("finalize-key") == int(1e18)
    assert store.get_total_flops_used("finalize-key") is None

    store.finalize_successful_run(config, 6.7, reservation_id)

    assert store.get_active_reserved_flops("finalize-key") == 0
    assert store.get_total_flops_used("finalize-key") == int(1e18)
    reservations = store.get_reservations(api_key="finalize-key", limit=10)
    assert reservations[0].status == "SUCCEEDED"


def test_admin_reservations_exposes_status_counts_and_entries_for_api_key(tmp_path: Path) -> None:
    db_path = tmp_path / "api.db"
    runtime = ApiRuntime(
        db_path=db_path,
        backend=RecordingBackend(),
        accept_all_keys=True,
    )
    pending_id = runtime.store.create_reservation(
        build_training_config(
            api_key="inspect-key",
            d_model=512,
            num_layers=8,
            num_heads=8,
            batch_size=128,
            learning_rate=1e-3,
            train_flops=int(1e16),
        )
    )
    running_id = runtime.store.create_reservation(
        build_training_config(
            api_key="inspect-key",
            d_model=768,
            num_layers=12,
            num_heads=12,
            batch_size=128,
            learning_rate=9e-4,
            train_flops=int(3e16),
        )
    )
    failed_id = runtime.store.create_reservation(
        build_training_config(
            api_key="inspect-key",
            d_model=256,
            num_layers=4,
            num_heads=4,
            batch_size=128,
            learning_rate=8e-4,
            train_flops=int(1e15),
        )
    )
    runtime.store.mark_reservation_running(running_id)
    runtime.store.mark_reservation_failed(failed_id, failure_reason="synthetic failure")
    client = TestClient(create_app(runtime))

    response = client.get("/__admin__/reservations", params={"api_key": "inspect-key"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_key"] == "inspect-key"
    assert payload["active_reserved_flops"] == int(4e16)
    assert payload["status_counts"] == {"FAILED": 1, "PENDING": 1, "RUNNING": 1}
    assert [row["id"] for row in payload["reservations"]] == [failed_id, running_id, pending_id]
    assert payload["reservations"][0]["failure_reason"] == "synthetic failure"
    assert payload["reservations"][1]["status"] == "RUNNING"
    assert payload["reservations"][2]["status"] == "PENDING"


def test_admin_reservations_lists_global_recent_reservations_without_api_key(tmp_path: Path) -> None:
    db_path = tmp_path / "api.db"
    runtime = ApiRuntime(
        db_path=db_path,
        backend=RecordingBackend(),
        accept_all_keys=True,
    )
    first_id = runtime.store.create_reservation(
        build_training_config(
            api_key="key-a",
            d_model=512,
            num_layers=8,
            num_heads=8,
            batch_size=128,
            learning_rate=1e-3,
            train_flops=int(1e16),
        )
    )
    second_id = runtime.store.create_reservation(
        build_training_config(
            api_key="key-b",
            d_model=768,
            num_layers=12,
            num_heads=12,
            batch_size=128,
            learning_rate=9e-4,
            train_flops=int(3e16),
        )
    )
    runtime.store.mark_reservation_running(second_id)
    client = TestClient(create_app(runtime))

    response = client.get("/__admin__/reservations", params={"limit": 10})

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_key"] is None
    assert payload["active_reserved_flops"] is None
    assert payload["status_counts"] == {"PENDING": 1, "RUNNING": 1}
    assert [row["id"] for row in payload["reservations"]] == [second_id, first_id]


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
