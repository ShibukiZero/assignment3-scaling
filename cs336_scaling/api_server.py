from __future__ import annotations

import argparse
import os
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
        self.backend = backend
        self.allowed_api_keys = set(allowed_api_keys or set())
        self.accept_all_keys = accept_all_keys

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

        cached_run = runtime.store.get_run(config)
        if cached_run is not None:
            total_flops_used = runtime.store.get_total_flops_used(resolved_api_key) or 0
            return {"loss": float(cached_run.loss), "total_flops_used": float(total_flops_used)}

        try:
            result = runtime.backend.run(config)
        except BackendUnavailableError as exc:
            raise assignment_error(503, str(exc)) from exc
        except TrainingOOMError as exc:
            raise assignment_error(503, str(exc)) from exc

        runtime.store.insert_run(config, result.loss)
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
