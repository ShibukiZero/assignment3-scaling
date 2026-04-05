from __future__ import annotations

import argparse

from cs336_scaling.tokenization import (
    add_shared_corpus_args,
    format_gib,
    split_corpus_to_jsonl,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/eval JSONL splits from a raw corpus."
    )
    add_shared_corpus_args(parser)
    parser.add_argument(
        "--train-output",
        required=True,
        help="Output JSONL path for the train split.",
    )
    parser.add_argument(
        "--eval-output",
        required=True,
        help="Output JSONL path for the eval split.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.005,
        help="Document-level fraction reserved for eval. Default: 0.005 (0.5%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed used by the deterministic hash split.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    stats = split_corpus_to_jsonl(
        args.input,
        args.train_output,
        args.eval_output,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        text_key=args.text_key,
        parquet_batch_size=args.parquet_batch_size,
    )

    metadata = {
        "input_path": args.input,
        "train_output_path": args.train_output,
        "eval_output_path": args.eval_output,
        "num_input_files": stats.num_files,
        "num_input_documents": stats.num_input_documents,
        "num_train_documents": stats.num_train_documents,
        "num_eval_documents": stats.num_eval_documents,
        "train_utf8_bytes": stats.train_utf8_bytes,
        "eval_utf8_bytes": stats.eval_utf8_bytes,
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "text_key": args.text_key,
    }
    write_json(f"{args.train_output}.meta.json", metadata)

    print("Corpus split finished.")
    print(f"Train docs: {stats.num_train_documents}")
    print(f"Eval docs: {stats.num_eval_documents}")
    print(f"Train size: {format_gib(stats.train_utf8_bytes)}")
    print(f"Eval size: {format_gib(stats.eval_utf8_bytes)}")
    print(f"Train path: {args.train_output}")
    print(f"Eval path: {args.eval_output}")


if __name__ == "__main__":
    main()
