#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

MAX_LR = 1e-3
REF_N = 42_467_328.0
REF_LR = 9e-4
EXPONENT = -0.4716889721


def recommended_lr(n_params: float) -> float:
    uncapped = REF_LR * (n_params / REF_N) ** EXPONENT
    return min(MAX_LR, uncapped)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Return the Chapter 3 working learning-rate recommendation as a function of "
            "parameter count N, based on the large-N capped power-law fit from experiment "
            "3_2_2_large_n_lr_refine."
        )
    )
    parser.add_argument("n_params", type=float, help="Target non-embedding parameter count N.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lr = recommended_lr(args.n_params)

    if args.json:
        print(
            json.dumps(
                {
                    "n_params": args.n_params,
                    "recommended_learning_rate": lr,
                    "formula": "min(1e-3, 9e-4 * (N / 42467328)^(-0.4716889721))",
                },
                indent=2,
            )
        )
        return

    print(f"N = {args.n_params:.0f}")
    print(f"Recommended learning rate = {lr:.10f}")
    print("Formula = min(1e-3, 9e-4 * (N / 42467328)^(-0.4716889721))")


if __name__ == "__main__":
    main()
