from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query

from cs336_scaling.api_backend import (
    BackendUnavailableError,
    TrainingBackend,
    TrainingOOMError,
    get_training_backend,
)
from cs336_scaling.api_contract import build_training_config, validate_api_key
from cs336_scaling.api_store import ApiStore


DEFAULT_DB_PATH = Path("artifacts/api/api.db")
SCALING_LAW_FLOPS_BUDGET_CAP = int(2e18)
logger = logging.getLogger("uvicorn.error")


def parse_allowed_keys(raw_value: str | None) -> set[str]:
    if raw_value is None or raw_value.strip() == "":
        return set()
    return {token.strip() for token in raw_value.split(",") if token.strip()}


class ApiRuntime:
    def __init__(
        self,
        *,
        db_path: str | Path,
        backend: TrainingBackend,
        allowed_api_keys: set[str] | None = None,
        accept_all_keys: bool = True,
    ) -> None:
        self.store = ApiStore(db_path)
        recovered = self.store.recover_incomplete_reservations()
        self.backend = backend
        self.allowed_api_keys = set(allowed_api_keys or set())
        self.accept_all_keys = accept_all_keys
        self._state_lock = threading.Lock()
        self._shutting_down = False
        self._inflight_runs: dict[object, _InflightRun] = {}
        logger.info(
            "api runtime initialized with db_path=%s backend=%s max_concurrency=%s recovered_reservations=%s",
            Path(db_path).expanduser(),
            type(backend).__name__,
            getattr(backend, "max_concurrency", "unknown"),
            recovered,
        )

        for api_key in self.allowed_api_keys:
            self.store.register_api_key(api_key)

    def ensure_valid_api_key(self, api_key: str) -> str:
        api_key = validate_api_key(api_key)
        if self.accept_all_keys:
            self.store.register_api_key(api_key)
            return api_key
        if api_key in self.allowed_api_keys:
            self.store.register_api_key(api_key)
            return api_key
        raise HTTPException(status_code=422, detail={"message": f"Invalid API key provided: {api_key}"})

    def ensure_api_key_for_history(self, api_key: str) -> str:
        api_key = validate_api_key(api_key)
        if self.accept_all_keys:
            return api_key
        if api_key in self.allowed_api_keys or self.store.has_api_key(api_key):
            return api_key
        raise HTTPException(status_code=422, detail={"message": f"Invalid API key provided: {api_key}"})

    def ensure_budget_available(self, api_key: str, additional_flops: int) -> None:
        total_flops_used = self.store.get_total_flops_used(api_key) or 0
        reserved_flops = self.store.get_active_reserved_flops(api_key)
        logger.info(
            "budget check for api_key=%s completed_flops=%s reserved_flops=%s requested_flops=%s cap=%s",
            api_key,
            total_flops_used,
            reserved_flops,
            additional_flops,
            SCALING_LAW_FLOPS_BUDGET_CAP,
        )
        if total_flops_used + reserved_flops + additional_flops > SCALING_LAW_FLOPS_BUDGET_CAP:
            logger.warning(
                "budget rejected for api_key=%s completed_flops=%s reserved_flops=%s requested_flops=%s cap=%s",
                api_key,
                total_flops_used,
                reserved_flops,
                additional_flops,
                SCALING_LAW_FLOPS_BUDGET_CAP,
            )
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "API key would exceed the scaling law FLOPs budget cap of "
                        f"{SCALING_LAW_FLOPS_BUDGET_CAP}: {api_key}"
                    )
                },
            )

    def begin_shutdown(self) -> None:
        with self._state_lock:
            self._shutting_down = True
        logger.warning("api runtime entering shutdown mode")
        self.backend.begin_shutdown()

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        drained = self.backend.wait_for_shutdown(timeout=timeout)
        logger.warning("api runtime shutdown drain completed=%s", drained)
        return drained

    def run_training_query(self, config):
        cached_run = self.store.get_run(config)
        if cached_run is not None:
            logger.info(
                "cache hit for api_key=%s d_model=%s layers=%s heads=%s batch=%s lr=%s flops=%s",
                config.api_key,
                config.d_model,
                config.num_layers,
                config.num_heads,
                config.batch_size,
                config.learning_rate,
                config.train_flops,
            )
            return cached_run, True

        with self._state_lock:
            if self._shutting_down:
                logger.warning(
                    "rejecting new query during shutdown for api_key=%s d_model=%s layers=%s heads=%s batch=%s lr=%s flops=%s",
                    config.api_key,
                    config.d_model,
                    config.num_layers,
                    config.num_heads,
                    config.batch_size,
                    config.learning_rate,
                    config.train_flops,
                )
                raise BackendUnavailableError("Training backend is shutting down.")
            inflight = self._inflight_runs.get(config)
            if inflight is None:
                self.ensure_budget_available(config.api_key, config.train_flops)
                reservation_id = self.store.create_reservation(config)
                inflight = _InflightRun(reservation_id=reservation_id)
                self._inflight_runs[config] = inflight
                is_owner = True
                logger.info(
                    "created reservation=%s for api_key=%s d_model=%s layers=%s heads=%s batch=%s lr=%s flops=%s",
                    reservation_id,
                    config.api_key,
                    config.d_model,
                    config.num_layers,
                    config.num_heads,
                    config.batch_size,
                    config.learning_rate,
                    config.train_flops,
                )
            else:
                is_owner = False
                logger.info(
                    "dedupe wait on inflight reservation=%s for api_key=%s d_model=%s layers=%s heads=%s batch=%s lr=%s flops=%s",
                    inflight.reservation_id,
                    config.api_key,
                    config.d_model,
                    config.num_layers,
                    config.num_heads,
                    config.batch_size,
                    config.learning_rate,
                    config.train_flops,
                )

        if not is_owner:
            inflight.event.wait()
            if inflight.error is not None:
                raise inflight.error
            assert inflight.result is not None
            return inflight.result, True

        try:
            self.store.mark_reservation_running(inflight.reservation_id)
            logger.info("reservation=%s marked RUNNING for api_key=%s", inflight.reservation_id, config.api_key)
            result = self.backend.run(config)
            self.store.insert_run(config, result.loss)
            self.store.mark_reservation_succeeded(inflight.reservation_id)
            logger.info(
                "reservation=%s marked SUCCEEDED for api_key=%s loss=%s",
                inflight.reservation_id,
                config.api_key,
                result.loss,
            )
            inflight.result = result
            return result, False
        except Exception as exc:
            self.store.mark_reservation_failed(
                inflight.reservation_id,
                failure_reason=str(exc),
            )
            logger.warning(
                "reservation=%s marked FAILED for api_key=%s reason=%s",
                inflight.reservation_id,
                config.api_key,
                exc,
            )
            inflight.error = exc
            raise
        finally:
            with self._state_lock:
                self._inflight_runs.pop(config, None)
            inflight.event.set()


def create_runtime_from_env() -> ApiRuntime:
    db_path = Path(os.environ.get("CS336_API_DB_PATH", DEFAULT_DB_PATH)).expanduser()
    allowed_api_keys = parse_allowed_keys(os.environ.get("CS336_API_KEYS"))
    accept_all_keys = os.environ.get("CS336_API_ACCEPT_ALL_KEYS", "1") == "1"
    return ApiRuntime(
        db_path=db_path,
        backend=get_training_backend(),
        allowed_api_keys=allowed_api_keys,
        accept_all_keys=accept_all_keys,
    )


def assignment_error(status_code: int, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"message": message})


class _InflightRun:
    def __init__(self, *, reservation_id: int) -> None:
        self.reservation_id = reservation_id
        self.event = threading.Event()
        self.result = None
        self.error: Exception | None = None


def create_app(runtime: ApiRuntime | None = None) -> FastAPI:
    runtime = runtime or create_runtime_from_env()
    app = FastAPI(title="CS336 Assignment 3 Training API", version="0.1.0")

    @app.get("/loss")
    def get_loss(
        d_model: int = Query(...),
        num_layers: int = Query(...),
        num_heads: int = Query(...),
        batch_size: int = Query(...),
        learning_rate: float = Query(...),
        train_flops: int = Query(...),
        api_key: str = Query(...),
    ) -> dict[str, float]:
        resolved_api_key = runtime.ensure_valid_api_key(api_key)
        try:
            config = build_training_config(
                api_key=resolved_api_key,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                batch_size=batch_size,
                learning_rate=learning_rate,
                train_flops=train_flops,
            )
        except ValueError as exc:
            raise assignment_error(404, str(exc)) from exc

        try:
            result, _ = runtime.run_training_query(config)
        except BackendUnavailableError as exc:
            raise assignment_error(503, str(exc)) from exc
        except TrainingOOMError as exc:
            raise assignment_error(503, str(exc)) from exc
        total_flops_used = runtime.store.get_total_flops_used(resolved_api_key) or 0
        return {"loss": float(result.loss), "total_flops_used": float(total_flops_used)}

    @app.get("/total_flops_used")
    def get_total_flops_used(api_key: str = Query(...)) -> float:
        resolved_api_key = runtime.ensure_api_key_for_history(api_key)
        total_flops_used = runtime.store.get_total_flops_used(resolved_api_key)
        if total_flops_used is None:
            raise assignment_error(422, f"API key has no queries yet: {resolved_api_key}")
        return float(total_flops_used)

    @app.get("/previous_runs")
    def get_previous_runs(api_key: str = Query(...)) -> dict[str, list[dict[str, float | int]]]:
        resolved_api_key = runtime.ensure_api_key_for_history(api_key)
        previous_runs = runtime.store.get_previous_runs(resolved_api_key)
        if not previous_runs:
            raise assignment_error(422, f"API key has no queries yet: {resolved_api_key}")
        return {"previous_runs": [run.to_public_dict() for run in previous_runs]}

    @app.post("/__admin__/shutdown")
    def admin_shutdown() -> dict[str, str]:
        runtime.begin_shutdown()

        def _drain_and_stop() -> None:
            runtime.wait_for_shutdown(timeout=None)
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=_drain_and_stop, daemon=True).start()
        return {"message": "Shutdown initiated."}

    @app.get("/__admin__/reservations")
    def admin_reservations(
        api_key: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, object]:
        if api_key is not None:
            resolved_api_key = runtime.ensure_api_key_for_history(api_key)
        else:
            resolved_api_key = None

        reservations = runtime.store.get_reservations(
            api_key=resolved_api_key,
            limit=limit,
        )
        return {
            "api_key": resolved_api_key,
            "active_reserved_flops": (
                runtime.store.get_active_reserved_flops(resolved_api_key)
                if resolved_api_key is not None
                else None
            ),
            "status_counts": runtime.store.get_reservation_status_counts(resolved_api_key),
            "reservations": [reservation.to_public_dict() for reservation in reservations],
        }

    return app


app = create_app()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the assignment-style training API scaffold.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for uvicorn.")
    parser.add_argument("--port", type=int, default=8000, help="Port for uvicorn.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    uvicorn.run("cs336_scaling.api_server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
