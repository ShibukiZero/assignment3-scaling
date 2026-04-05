from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cs336_scaling.api_contract import TrainingConfig


@dataclass(frozen=True)
class StoredRun:
    api_key: str
    d_model: int
    num_layers: int
    num_heads: int
    batch_size: int
    learning_rate: float
    train_flops: int
    loss: float

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "train_flops": self.train_flops,
            "loss": self.loss,
        }


class ApiStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    api_key TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_key TEXT NOT NULL,
                    d_model INTEGER NOT NULL,
                    num_layers INTEGER NOT NULL,
                    num_heads INTEGER NOT NULL,
                    batch_size INTEGER NOT NULL,
                    learning_rate REAL NOT NULL,
                    train_flops INTEGER NOT NULL,
                    loss REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(api_key, d_model, num_layers, num_heads, batch_size, learning_rate, train_flops),
                    FOREIGN KEY(api_key) REFERENCES api_keys(api_key)
                );
                """
            )

    def register_api_key(self, api_key: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO api_keys(api_key)
                VALUES (?)
                ON CONFLICT(api_key) DO NOTHING
                """,
                (api_key,),
            )

    def has_api_key(self, api_key: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM api_keys WHERE api_key = ?",
                (api_key,),
            ).fetchone()
        return row is not None

    def get_run(self, config: TrainingConfig) -> StoredRun | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT api_key, d_model, num_layers, num_heads, batch_size,
                       learning_rate, train_flops, loss
                FROM runs
                WHERE api_key = ?
                  AND d_model = ?
                  AND num_layers = ?
                  AND num_heads = ?
                  AND batch_size = ?
                  AND learning_rate = ?
                  AND train_flops = ?
                """,
                (
                    config.api_key,
                    config.d_model,
                    config.num_layers,
                    config.num_heads,
                    config.batch_size,
                    config.learning_rate,
                    config.train_flops,
                ),
            ).fetchone()
        return StoredRun(**dict(row)) if row is not None else None

    def insert_run(self, config: TrainingConfig, loss: float) -> None:
        self.register_api_key(config.api_key)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO runs(
                    api_key, d_model, num_layers, num_heads, batch_size,
                    learning_rate, train_flops, loss
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    config.api_key,
                    config.d_model,
                    config.num_layers,
                    config.num_heads,
                    config.batch_size,
                    config.learning_rate,
                    config.train_flops,
                    loss,
                ),
            )

    def get_total_flops_used(self, api_key: str) -> int | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT SUM(train_flops) AS total_flops_used FROM runs WHERE api_key = ?",
                (api_key,),
            ).fetchone()
        total = row["total_flops_used"] if row is not None else None
        return int(total) if total is not None else None

    def get_previous_runs(self, api_key: str) -> list[StoredRun]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT api_key, d_model, num_layers, num_heads, batch_size,
                       learning_rate, train_flops, loss
                FROM runs
                WHERE api_key = ?
                ORDER BY id ASC
                """,
                (api_key,),
            ).fetchall()
        return [StoredRun(**dict(row)) for row in rows]
