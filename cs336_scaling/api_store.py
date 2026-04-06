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


@dataclass(frozen=True)
class StoredReservation:
    id: int
    api_key: str
    d_model: int
    num_layers: int
    num_heads: int
    batch_size: int
    learning_rate: float
    train_flops: int
    status: str
    failure_reason: str | None
    created_at: str
    updated_at: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "api_key": self.api_key,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "train_flops": self.train_flops,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
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

                CREATE TABLE IF NOT EXISTS reservations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_key TEXT NOT NULL,
                    d_model INTEGER NOT NULL,
                    num_layers INTEGER NOT NULL,
                    num_heads INTEGER NOT NULL,
                    batch_size INTEGER NOT NULL,
                    learning_rate REAL NOT NULL,
                    train_flops INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    failure_reason TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
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

    def create_reservation(self, config: TrainingConfig) -> int:
        self.register_api_key(config.api_key)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO reservations(
                    api_key, d_model, num_layers, num_heads, batch_size,
                    learning_rate, train_flops, status
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
                    "PENDING",
                ),
            )
        return int(cursor.lastrowid)

    def mark_reservation_running(self, reservation_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE reservations
                SET status = 'RUNNING',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (reservation_id,),
            )

    def mark_reservation_succeeded(self, reservation_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE reservations
                SET status = 'SUCCEEDED',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (reservation_id,),
            )

    def mark_reservation_failed(
        self,
        reservation_id: int,
        *,
        status: str = "FAILED",
        failure_reason: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE reservations
                SET status = ?,
                    failure_reason = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, failure_reason, reservation_id),
            )

    def get_active_reserved_flops(self, api_key: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(SUM(train_flops), 0) AS total_reserved_flops
                FROM reservations
                WHERE api_key = ?
                  AND status IN ('PENDING', 'RUNNING')
                """,
                (api_key,),
            ).fetchone()
        return int(row["total_reserved_flops"]) if row is not None else 0

    def recover_incomplete_reservations(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE reservations
                SET status = 'FAILED_RECOVERED',
                    failure_reason = 'Recovered on startup after an incomplete previous run.',
                    updated_at = CURRENT_TIMESTAMP
                WHERE status IN ('PENDING', 'RUNNING')
                """
            )
        return int(cursor.rowcount)

    def get_recovered_reservations(self, api_key: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT api_key, d_model, num_layers, num_heads, batch_size,
                       learning_rate, train_flops, status, failure_reason
                FROM reservations
                WHERE api_key = ?
                  AND status = 'FAILED_RECOVERED'
                ORDER BY id ASC
                """,
                (api_key,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_reservations(
        self,
        *,
        api_key: str | None = None,
        limit: int = 100,
    ) -> list[StoredReservation]:
        normalized_limit = max(1, min(limit, 1000))
        query = """
            SELECT id, api_key, d_model, num_layers, num_heads, batch_size,
                   learning_rate, train_flops, status, failure_reason,
                   created_at, updated_at
            FROM reservations
        """
        params: tuple[Any, ...]
        if api_key is None:
            query += " ORDER BY id DESC LIMIT ?"
            params = (normalized_limit,)
        else:
            query += " WHERE api_key = ? ORDER BY id DESC LIMIT ?"
            params = (api_key, normalized_limit)

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [StoredReservation(**dict(row)) for row in rows]

    def get_reservation_status_counts(self, api_key: str | None = None) -> dict[str, int]:
        query = """
            SELECT status, COUNT(*) AS reservation_count
            FROM reservations
        """
        params: tuple[Any, ...]
        if api_key is None:
            query += " GROUP BY status ORDER BY status ASC"
            params = ()
        else:
            query += " WHERE api_key = ? GROUP BY status ORDER BY status ASC"
            params = (api_key,)

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return {str(row["status"]): int(row["reservation_count"]) for row in rows}

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
