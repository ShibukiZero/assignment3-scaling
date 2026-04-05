from __future__ import annotations

import argparse

from cs336_scaling.tokenization import (
    add_shared_corpus_args,
    format_gib,
    parse_bytes_arg,
    sample_corpus_to_jsonl,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic random JSONL subset for tokenizer training."
    )
    add_shared_corpus_args(parser)
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for the sampled subset.",
    )
    parser.add_argument(
        "--target-bytes",
        default="3GiB",
        help="Approximate UTF-8 text budget to target, e.g. 2GiB, 3GiB, 3500000000.",
    )
    parser.add_argument(
        "--max-bytes",
        default="4GiB",
        help="Hard cap on sampled UTF-8 text size, e.g. 4GiB.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed used for deterministic sampling.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    target_bytes = parse_bytes_arg(args.target_bytes)
    max_bytes = parse_bytes_arg(args.max_bytes) if args.max_bytes else None
    stats = sample_corpus_to_jsonl(
        args.input,
        args.output,
        target_bytes=target_bytes,
        max_bytes=max_bytes,
        seed=args.seed,
        text_key=args.text_key,
        parquet_batch_size=args.parquet_batch_size,
    )

    metadata = {
        "input_path": args.input,
        "output_path": args.output,
        "num_input_files": stats.num_files,
        "num_input_documents": stats.num_input_documents,
        "num_output_documents": stats.num_output_documents,
        "total_input_utf8_bytes": stats.total_input_utf8_bytes,
        "total_output_utf8_bytes": stats.total_output_utf8_bytes,
        "sample_ratio": stats.sample_ratio,
        "seed": args.seed,
        "text_key": args.text_key,
        "target_bytes": target_bytes,
        "max_bytes": max_bytes,
    }
    write_json(f"{args.output}.meta.json", metadata)

    print("Corpus sampling finished.")
    print(f"Output docs: {stats.num_output_documents}")
    print(f"Output size: {format_gib(stats.total_output_utf8_bytes)}")
    print(f"Sample ratio: {stats.sample_ratio:.6f}")
    print(f"Sample path: {args.output}")


if __name__ == "__main__":
    main()
