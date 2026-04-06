#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests


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
            "Expand a Chapter 3 experiment grid JSON, query the training API sequentially, "
            "and write a compact JSON results file under .agents/logs/<experiment_id>/."
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
        help="Optional API key override. If omitted, the script reads the env var named in the grid JSON.",
    )
    parser.add_argument(
        "--output-root",
        default=".agents/logs",
        help="Root directory where experiment result directories will be created.",
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.grid_config)
    raw_config = load_grid_config(config_path)

    experiment_id = str(raw_config["experiment_id"])
    base_url = args.base_url or str(raw_config["base_url"])
    api_key = args.api_key
    if api_key is None:
        api_key_env = str(raw_config.get("api_key_env", "CS336_API_KEY"))
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"No API key provided. Set --api-key or export the env var {api_key_env}."
            )

    endpoint = str(raw_config.get("endpoint", "/loss"))
    shared_params = dict(raw_config.get("shared_params", {}))
    axes = load_axes(list(raw_config["axes"]))

    requests_plan = expand_requests(
        experiment_id=experiment_id,
        endpoint=endpoint,
        shared_params=shared_params,
        axes=axes,
        api_key=api_key,
    )

    output_dir = Path(args.output_root) / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    responses = [run_request(base_url=base_url, request_plan=request_plan) for request_plan in requests_plan]

    payload = {
        "experiment_id": experiment_id,
        "grid_config_path": str(config_path),
        "base_url": base_url,
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
    print(f"Results path: {results_path}")


if __name__ == "__main__":
    main()
