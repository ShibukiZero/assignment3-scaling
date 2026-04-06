from __future__ import annotations

import argparse

from cs336_scaling.tokenization import (
    add_shared_corpus_args,
    parse_special_tokens,
    save_tokenizer_artifacts,
    train_byte_level_bpe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer for scaling experiments.")
    add_shared_corpus_args(parser)
    parser.add_argument("--output-dir", required=True, help="Directory where tokenizer artifacts will be saved.")
    parser.add_argument("--vocab-size", type=int, default=32_000, help="Final tokenizer vocabulary size.")
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum pair frequency required by the BPE trainer.",
    )
    parser.add_argument(
        "--special-tokens",
        default="<|endoftext|>",
        help="Comma-separated list of special tokens to reserve in the vocabulary.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    special_tokens = parse_special_tokens(args.special_tokens)

    tokenizer, stats = train_byte_level_bpe(
        args.input,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        min_frequency=args.min_frequency,
        text_key=args.text_key,
        parquet_batch_size=args.parquet_batch_size,
    )
    save_tokenizer_artifacts(
        tokenizer,
        args.output_dir,
        input_path=args.input,
        stats=stats,
        vocab_size_target=args.vocab_size,
        min_frequency=args.min_frequency,
        text_key=args.text_key,
        special_tokens=special_tokens,
    )

    print("Tokenizer training finished.")
    print(f"Documents: {stats.num_documents}")
    print(f"Files: {stats.num_files}")
    print(f"Saved artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
