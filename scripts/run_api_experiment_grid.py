#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

DEFAULT_EXPERIMENT_API_KEY = "cs336_assignment3_fixed_key"


@dataclass(frozen=True)
class AxisValue:
    id: str
    params: dict[str, Any]


@dataclass(frozen=True)
class Axis:
    name: str
    values: list[AxisValue]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Expand a Chapter 3 experiment grid JSON, query the training API with bounded concurrency, "
            "and write a compact JSON results file into the matching experiment artifact directory."
        )
    )
    parser.add_argument(
        "grid_config",
        help="Path to the experiment grid JSON file.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional API base URL override. Defaults to the grid JSON value.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key override. Defaults to the fixed experiment API key in this script.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override. Defaults to the directory that contains the grid JSON file.",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        help="Optional client concurrency override. Defaults to the detected local GPU count.",
    )
    return parser


def load_grid_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_axes(raw_axes: list[dict[str, Any]]) -> list[Axis]:
    axes: list[Axis] = []
    for raw_axis in raw_axes:
        values = [
            AxisValue(
                id=str(raw_value["id"]),
                params=dict(raw_value["params"]),
            )
            for raw_value in raw_axis["values"]
        ]
        axes.append(Axis(name=str(raw_axis["name"]), values=values))
    return axes


def expand_requests(
    *,
    experiment_id: str,
    endpoint: str,
    shared_params: dict[str, Any],
    axes: list[Axis],
    api_key: str,
) -> list[dict[str, Any]]:
    requests_plan: list[dict[str, Any]] = []
    for combination in itertools.product(*(axis.values for axis in axes)):
        params = dict(shared_params)
        combo_parts: list[str] = []
        axis_choices: dict[str, str] = {}
        for axis, chosen_value in zip(axes, combination, strict=True):
            params.update(chosen_value.params)
            combo_parts.append(chosen_value.id)
            axis_choices[axis.name] = chosen_value.id
        params["api_key"] = api_key
        request_id = "__".join(combo_parts)
        requests_plan.append(
            {
                "experiment_id": experiment_id,
                "request_id": request_id,
                "endpoint": endpoint,
                "axis_choices": axis_choices,
                "params": params,
            }
        )
    return requests_plan


def run_request(*, base_url: str, request_plan: dict[str, Any]) -> dict[str, Any]:
    endpoint = request_plan["endpoint"].lstrip("/")
    url = f"{base_url.rstrip('/')}/{endpoint}"
    response = requests.get(url, params=request_plan["params"], timeout=600)
    body_text = response.text
    try:
        body_json = response.json()
    except ValueError:
        body_json = None

    return {
        "request_id": request_plan["request_id"],
        "endpoint": request_plan["endpoint"],
        "axis_choices": request_plan["axis_choices"],
        "params_without_api_key": {
            key: value
            for key, value in request_plan["params"].items()
            if key != "api_key"
        },
        "status_code": response.status_code,
        "url": f"{url}?{urlencode(request_plan['params'])}",
        "body_json": body_json,
        "body_text": body_text if body_json is None else None,
    }


def detect_local_gpu_count() -> int:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None and cuda_visible_devices.strip() != "":
        tokens = [token.strip() for token in cuda_visible_devices.split(",") if token.strip()]
        if tokens:
            return len(tokens)

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return 1

    gpu_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return len(gpu_lines) if gpu_lines else 1


def run_requests_bounded(
    *,
    base_url: str,
    requests_plan: list[dict[str, Any]],
    max_inflight: int,
) -> list[dict[str, Any]]:
    if max_inflight <= 1:
        return [run_request(base_url=base_url, request_plan=request_plan) for request_plan in requests_plan]

    responses_by_id: dict[str, dict[str, Any]] = {}
    next_index = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_inflight) as executor:
        inflight: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}

        while next_index < len(requests_plan) or inflight:
            while next_index < len(requests_plan) and len(inflight) < max_inflight:
                request_plan = requests_plan[next_index]
                future = executor.submit(run_request, base_url=base_url, request_plan=request_plan)
                inflight[future] = request_plan
                next_index += 1

            if not inflight:
                time.sleep(1.0)
                continue

            done, _ = concurrent.futures.wait(
                list(inflight.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED,
                timeout=1.0,
            )
            for future in done:
                request_plan = inflight.pop(future)
                responses_by_id[request_plan["request_id"]] = future.result()

        return [responses_by_id[request_plan["request_id"]] for request_plan in requests_plan]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.grid_config)
    raw_config = load_grid_config(config_path)

    experiment_id = str(raw_config["experiment_id"])
    base_url = args.base_url or str(raw_config["base_url"])
    api_key = args.api_key or str(raw_config.get("api_key", DEFAULT_EXPERIMENT_API_KEY))

    endpoint = str(raw_config.get("endpoint", "/loss"))
    shared_params = dict(raw_config.get("shared_params", {}))
    axes = load_axes(list(raw_config["axes"]))
    client_config = dict(raw_config.get("client", {}))
    detected_gpu_count = detect_local_gpu_count()
    configured_max_inflight = client_config.get("max_inflight")
    if args.max_inflight is not None:
        max_inflight = args.max_inflight
    elif configured_max_inflight is None or int(configured_max_inflight) <= 0:
        max_inflight = detected_gpu_count
    else:
        max_inflight = int(configured_max_inflight)

    requests_plan = expand_requests(
        experiment_id=experiment_id,
        endpoint=endpoint,
        shared_params=shared_params,
        axes=axes,
        api_key=api_key,
    )

    output_dir = Path(args.output_dir) if args.output_dir is not None else config_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    responses = run_requests_bounded(
        base_url=base_url,
        requests_plan=requests_plan,
        max_inflight=max_inflight,
    )

    payload = {
        "experiment_id": experiment_id,
        "grid_config_path": str(config_path),
        "base_url": base_url,
        "api_key": api_key,
        "client": {
            "detected_local_gpu_count": detected_gpu_count,
            "max_inflight": max_inflight,
        },
        "num_requests": len(requests_plan),
        "request_plan": [
            {
                "request_id": request_plan["request_id"],
                "endpoint": request_plan["endpoint"],
                "axis_choices": request_plan["axis_choices"],
                "params_without_api_key": {
                    key: value
                    for key, value in request_plan["params"].items()
                    if key != "api_key"
                },
            }
            for request_plan in requests_plan
        ],
        "responses": responses,
    }
    results_path.write_text(json.dumps(payload, indent=2))

    print(f"Experiment finished: {experiment_id}")
    print(f"Requests sent: {len(requests_plan)}")
    print(f"Detected local GPU count: {detected_gpu_count}")
    print(f"Client max_inflight: {max_inflight}")
    print(f"Results path: {results_path}")


if __name__ == "__main__":
    main()
