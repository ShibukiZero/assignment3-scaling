from __future__ import annotations

import json
from array import array
from dataclasses import dataclass
from pathlib import Path


DTYPE_TO_ARRAY_CODE = {
    "uint16": "H",
    "uint32": "I",
}


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _load_array(path: Path, array_code: str) -> array:
    values = array(array_code)
    itemsize = values.itemsize
    file_size = path.stat().st_size
    if file_size % itemsize != 0:
        raise ValueError(
            f"File size for {path} is not divisible by {itemsize}-byte elements for dtype code {array_code}."
        )
    count = file_size // itemsize
    with path.open("rb") as handle:
        values.fromfile(handle, count)
    return values


@dataclass(frozen=True)
class TokenizedDataset:
    tokens: array
    offsets: array
    dtype: str

    @classmethod
    def from_meta(cls, meta_path: str | Path) -> TokenizedDataset:
        meta_path = Path(meta_path).expanduser()
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        base_dir = meta_path.parent

        dtype = str(metadata["dtype"])
        if dtype not in DTYPE_TO_ARRAY_CODE:
            raise ValueError(f"Unsupported token dtype in metadata: {dtype}")

        ids_path = _resolve_path(base_dir, str(metadata["ids_path"]))
        idx_path = _resolve_path(base_dir, str(metadata["idx_path"]))
        num_documents = int(metadata["num_documents"])
        num_tokens = int(metadata["num_tokens"])

        tokens = _load_array(ids_path, DTYPE_TO_ARRAY_CODE[dtype])
        offsets = _load_array(idx_path, "Q")

        if len(tokens) != num_tokens:
            raise ValueError(
                f"Metadata num_tokens={num_tokens} does not match loaded token count={len(tokens)}"
            )
        if len(offsets) != num_documents + 1:
            raise ValueError(
                f"Expected {num_documents + 1} offsets for {num_documents} documents, got {len(offsets)}"
            )
        if len(offsets) == 0 or offsets[0] != 0:
            raise ValueError("Offset table must start at 0.")
        if offsets[-1] != num_tokens:
            raise ValueError(
                f"Last offset must equal num_tokens={num_tokens}, got {offsets[-1]}"
            )
        for left, right in zip(offsets, offsets[1:]):
            if left > right:
                raise ValueError("Offsets must be monotonically non-decreasing.")

        return cls(tokens=tokens, offsets=offsets, dtype=dtype)

    @property
    def num_documents(self) -> int:
        return len(self.offsets) - 1

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    def get_document(self, index: int) -> list[int]:
        if index < 0 or index >= self.num_documents:
            raise IndexError(f"Document index out of range: {index}")
        start = self.offsets[index]
        end = self.offsets[index + 1]
        return list(self.tokens[start:end])

    def get_token_block(self, *, start: int, length: int) -> list[int]:
        if start < 0:
            raise IndexError(f"Token block start must be non-negative, got {start}")
        if length < 0:
            raise ValueError(f"Token block length must be non-negative, got {length}")
        end = start + length
        if end > self.num_tokens:
            raise IndexError(
                f"Requested token block [{start}, {end}) exceeds corpus length {self.num_tokens}"
            )
        return list(self.tokens[start:end])
