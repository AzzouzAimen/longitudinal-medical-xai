"""
attention_store.py

 – Task A3 : Store attention weights in TimescaleDB
Topic M6 : Synthetic Thermal Time-Series
What this file does
-------------------
1. Creates the attention_maps table in TimescaleDB (once, on first import)
2. Provides insert_attention_map() to save weights after each inference

This is imported and called inside train_example.py after each forward pass.

Required by:
  A4 — query top-k attention windows
  B4 — Attention Alignment Accuracy
  B5 — heatmap generation

Usage
-----
  # In train_example.py, after model inference:
  from attention_store import AttentionStore

  store = AttentionStore(dsn)          # creates table if not exists
  store.insert(
      window_id  = 42,
      patient_id = 3,
      timestamps = [datetime1, datetime2, ...],   # 60 timestamps
      weights    = [0.01, 0.03, ...],             # 60 attention weights
      epoch      = 5,
  )
"""

import os
import logging
from datetime import datetime
from typing import List, Union

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("attention_store")


# ---------------------------------------------------------------------------
# SQL — table + indexes (created once)
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS attention_maps (
    id          BIGSERIAL    PRIMARY KEY,
    window_id   INTEGER      NOT NULL,
    patient_id  SMALLINT     NOT NULL,
    timestamp   TIMESTAMPTZ  NOT NULL,
    weight      REAL         NOT NULL,
    epoch       INTEGER,
    inferred_at TIMESTAMPTZ  DEFAULT NOW()
);
"""

_CREATE_INDEXES = [
    # A4: find top-k weights per window
    """
    CREATE INDEX IF NOT EXISTS idx_attn_window_weight
    ON attention_maps (window_id, patient_id, weight DESC);
    """,
    # B4/B5: retrieve all weights for a specific window
    """
    CREATE INDEX IF NOT EXISTS idx_attn_window_ts
    ON attention_maps (window_id, patient_id, timestamp ASC);
    """,
]


# ---------------------------------------------------------------------------
# AttentionStore
# ---------------------------------------------------------------------------

class AttentionStore:
    """
    Manages the attention_maps table in TimescaleDB.

    Parameters
    ----------
    dsn : str
        PostgreSQL/TimescaleDB connection string.
        If None, reads from environment variables.

    Example
    -------
    store = AttentionStore(dsn)

    # After model inference on a batch:
    store.insert(
        window_id  = int(meta["window_id"]),
        patient_id = int(meta["patient_id"]),
        timestamps = list(meta["timestamps"]),   # list of datetime, len=60
        weights    = attn_weights.tolist(),      # list of float, len=60
        epoch      = current_epoch,
    )
    """

    def __init__(self, dsn: str = None):
        self.dsn = dsn or _build_dsn()
        self._ensure_table()

    # ------------------------------------------------------------------
    # Table creation (runs once)
    # ------------------------------------------------------------------

    def _ensure_table(self):
        """Creates attention_maps table and indexes if they don't exist."""
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE)
                for idx_sql in _CREATE_INDEXES:
                    cur.execute(idx_sql)
            conn.commit()
            log.info("attention_maps table ready (created or already exists)")
        except Exception as e:
            conn.rollback()
            log.error(f"Failed to create attention_maps table: {e}")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Insert — single window
    # ------------------------------------------------------------------

    def insert(
        self,
        window_id:  int,
        patient_id: int,
        timestamps: List[datetime],
        weights:    Union[List[float], np.ndarray],
        epoch:      int = None,
    ):
        """
        Inserts attention weights for one window into TimescaleDB.

        Parameters
        ----------
        window_id  : int
            The window_id from windows_tsdb table.
        patient_id : int
            The patient this window belongs to.
        timestamps : list of datetime
            One datetime per timestep in the window (length = window_size = 60).
        weights : list or np.ndarray
            Attention weight per timestep (same length as timestamps).
            These are the mean attention weights across all heads and layers.
        epoch : int, optional
            Training epoch when this inference was made.
        """
        if len(timestamps) != len(weights):
            raise ValueError(
                f"timestamps ({len(timestamps)}) and weights ({len(weights)}) "
                "must have the same length."
            )

        weights = np.array(weights, dtype=np.float32)

        rows = [
            (window_id, patient_id, ts, float(w), epoch)
            for ts, w in zip(timestamps, weights)
        ]

        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO attention_maps
                        (window_id, patient_id, timestamp, weight, epoch)
                    VALUES %s
                    """,
                    rows,
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            log.warning(f"Failed to insert attention map for window {window_id}: {e}")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Insert — full batch (more efficient)
    # ------------------------------------------------------------------

    def insert_batch(
        self,
        window_ids:  List[int],
        patient_ids: List[int],
        timestamps_batch: List[List[datetime]],
        weights_batch:    Union[List[List[float]], np.ndarray],
        epoch:       int = None,
    ):
        """
        Inserts attention weights for a full batch of windows in one DB call.

        Parameters
        ----------
        window_ids       : list of int, length = batch_size
        patient_ids      : list of int, length = batch_size
        timestamps_batch : list of lists — timestamps_batch[i] = 60 timestamps for window i
        weights_batch    : (batch_size, window_size) array of attention weights
        epoch            : training epoch
        """
        rows = []
        for i, (wid, pid) in enumerate(zip(window_ids, patient_ids)):
            for ts, w in zip(timestamps_batch[i], weights_batch[i]):
                rows.append((wid, pid, ts, float(w), epoch))

        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO attention_maps
                        (window_id, patient_id, timestamp, weight, epoch)
                    VALUES %s
                    """,
                    rows,
                )
            conn.commit()
            log.debug(f"Inserted {len(rows)} attention rows for {len(window_ids)} windows")
        except Exception as e:
            conn.rollback()
            log.warning(f"Batch insert failed: {e}")
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Query helpers (used by A4)
    # ------------------------------------------------------------------

    def top_k_timestamps(self, window_id: int, patient_id: int, k: int = 5):
        """
        Returns the top-k timestamps with highest attention weights for a window.
        Used by A4 to validate which timesteps the model focused on.

        Returns list of (timestamp, weight) tuples, sorted by weight DESC.
        """
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT timestamp, weight
                    FROM attention_maps
                    WHERE window_id = %s AND patient_id = %s
                    ORDER BY weight DESC
                    LIMIT %s;
                    """,
                    (window_id, patient_id, k),
                )
                return cur.fetchall()
        finally:
            conn.close()

    def get_window_weights(self, window_id: int, patient_id: int):
        """
        Returns all (timestamp, weight) for a window ordered by time.
        Used by B5 for heatmap generation.
        """
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT timestamp, weight
                    FROM attention_maps
                    WHERE window_id = %s AND patient_id = %s
                    ORDER BY timestamp ASC;
                    """,
                    (window_id, patient_id),
                )
                return cur.fetchall()
        finally:
            conn.close()

    def count(self):
        """Returns total number of rows in attention_maps."""
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM attention_maps;")
                return cur.fetchone()[0]
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_dsn() -> str:
    host     = os.getenv("TSDB_HOST",     "localhost")
    port     = os.getenv("TSDB_PORT",     "5433")
    user     = os.getenv("TSDB_USER",     "postgres")
    password = os.getenv("TSDB_PASSWORD", "postgres")
    db       = os.getenv("TSDB_DB",       "m6_thermal_tsdb")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


# ---------------------------------------------------------------------------
# Extract attention weights from AnomalyTransformer
# ---------------------------------------------------------------------------

def extract_mean_attention(model, signals_batch):
    """
    Extracts mean attention weights from AnomalyTransformer for a batch.

    Takes the attention maps from all layers and heads,
    averages them, then averages across the query dimension
    to get one weight per timestep.

    Parameters
    ----------
    model         : AnomalyTransformer instance
    signals_batch : torch.Tensor of shape (B, C, T)

    Returns
    -------
    np.ndarray of shape (B, T) — one weight per timestep per sample
    """
    import torch

    model.eval()
    with torch.no_grad():
        # attn_maps shape: (n_layers, B, n_heads, T, T)
        attn_maps = model.get_attention_maps(signals_batch)

    # Average over layers and heads → (B, T, T)
    attn_mean = attn_maps.mean(dim=0).mean(dim=1)

    # Average over query dimension → (B, T)
    # Each value = how much attention timestep t received on average
    attn_per_timestep = attn_mean.mean(dim=1)

    return attn_per_timestep.cpu().numpy()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    print("Testing AttentionStore...")
    store = AttentionStore()
    print(f"Table created. Current row count: {store.count()}")

    # Insert a fake window to verify it works
    from datetime import timezone, timedelta
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [base + timedelta(seconds=i) for i in range(60)]
    weights    = np.random.dirichlet(np.ones(60)).tolist()  # sum to 1

    store.insert(
        window_id  = 9999,
        patient_id = 0,
        timestamps = timestamps,
        weights    = weights,
        epoch      = 0,
    )
    print(f"Inserted test window. Row count now: {store.count()}")

    top5 = store.top_k_timestamps(window_id=9999, patient_id=0, k=5)
    print(f"Top-5 timestamps for test window:")
    for ts, w in top5:
        print(f"  {ts}  weight={w:.4f}")

    print("\nAttentionStore A3 test passed.")