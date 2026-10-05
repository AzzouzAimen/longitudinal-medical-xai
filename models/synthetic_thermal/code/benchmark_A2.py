"""
benchmark_a2.py

 – Task A2 : DB vs NPY fetch speed comparison
Topic M6 : Synthetic Thermal Time-Series
What this script does
---------------------
Calls the existing benchmark_vs_npy() function from thermal_tsdb_dataset.py
across multiple batch sizes, then prints and saves a comparison table
(ms/batch and samples/s) for TimescaleDB vs raw .npy files.

Output
------
  results/a2_benchmark_results.csv   ← the comparison table
  results/a2_benchmark_results.json  ← full raw results

Usage
-----
  # From inside the Data-Loader folder:
  python benchmark_a2.py

  # With custom paths:
  python benchmark_a2.py --npy-dir "../Data-Wrangling/etl_output/npy" --patients 5
"""

import argparse
import json
import os
import time
import logging
from pathlib import Path

import numpy as np
import psycopg2
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark_a2")


def _build_dsn() -> str:
    host     = os.getenv("TSDB_HOST",     "localhost")
    port     = os.getenv("TSDB_PORT",     "5433")
    user     = os.getenv("TSDB_USER",     "postgres")
    password = os.getenv("TSDB_PASSWORD", "postgres")
    db       = os.getenv("TSDB_DB",       "m6_thermal_tsdb")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def get_all_patient_ids(dsn: str):
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT patient_id FROM subjects ORDER BY patient_id;")
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def run_benchmark(dsn, npy_dir, patient_ids, batch_sizes, n_runs=3):
    """
    Runs benchmark_vs_npy() for each batch size, n_runs times,
    and returns averaged results.
    """
    from thermal_tsdb_dataset import benchmark_vs_npy

    all_results = []

    for batch_size in batch_sizes:
        log.info(f"\n{'='*50}")
        log.info(f"Batch size = {batch_size}")
        log.info(f"{'='*50}")

        db_times, npy_times = [], []
        db_rates, npy_rates = [], []

        for run in range(n_runs):
            log.info(f"  Run {run+1}/{n_runs}...")
            result = benchmark_vs_npy(
                dsn=dsn,
                npy_dir=Path(npy_dir),
                patient_ids=patient_ids,
                batch_size=batch_size,
                num_workers=0,
                n_epochs=1,
            )
            db_times.append(result["db_epoch_s"])
            npy_times.append(result["npy_epoch_s"])
            db_rates.append(result["db_samples_per_s"])
            npy_rates.append(result["npy_samples_per_s"])

        # Calculate ms per batch
        # total batches = total samples / batch_size (approx)
        from thermal_tsdb_dataset import ThermalTSDBDataset
        ds = ThermalTSDBDataset(dsn=dsn, patient_ids=patient_ids)
        n_batches = len(ds) / batch_size

        db_ms_per_batch  = (np.mean(db_times)  * 1000) / n_batches
        npy_ms_per_batch = (np.mean(npy_times) * 1000) / n_batches

        row = {
            "batch_size":          batch_size,
            "db_epoch_s":          round(np.mean(db_times), 3),
            "npy_epoch_s":         round(np.mean(npy_times), 3),
            "db_ms_per_batch":     round(db_ms_per_batch, 2),
            "npy_ms_per_batch":    round(npy_ms_per_batch, 2),
            "db_samples_per_s":    round(np.mean(db_rates), 0),
            "npy_samples_per_s":   round(np.mean(npy_rates), 0),
            "speedup_npy_over_db": round(np.mean(db_times) / np.mean(npy_times), 2),
        }
        all_results.append(row)

    return all_results


def print_table(results):
    """Prints a formatted comparison table to the console."""
    print("\n" + "="*85)
    print("A2 BENCHMARK RESULTS — TimescaleDB vs Raw NPY Files")
    print("="*85)
    print(f"{'Batch':>6} | {'DB ms/batch':>12} | {'NPY ms/batch':>13} | "
          f"{'DB samples/s':>13} | {'NPY samples/s':>14} | {'NPY speedup':>11}")
    print("-"*85)
    for r in results:
        print(
            f"{r['batch_size']:>6} | "
            f"{r['db_ms_per_batch']:>12.2f} | "
            f"{r['npy_ms_per_batch']:>13.2f} | "
            f"{r['db_samples_per_s']:>13,.0f} | "
            f"{r['npy_samples_per_s']:>14,.0f} | "
            f"{r['speedup_npy_over_db']:>10.1f}x"
        )
    print("="*85)
    print("Interpretation: NPY speedup > 1 means raw files are faster than DB.")
    print("This is expected — DB adds query parsing overhead vs sequential disk reads.")
    print()


def save_csv(results, out_path: Path):
    import csv
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    log.info(f"CSV saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=" A2 — DB vs NPY Benchmark")
    p.add_argument("--npy-dir",   default="../Data-Wrangling/etl_output/npy",
                   help="Path to folder containing patient_XX_windows.npy files")
    p.add_argument("--patients",  type=int, default=None,
                   help="Number of patients to use (default: all)")
    p.add_argument("--runs",      type=int, default=3,
                   help="Number of runs per batch size for averaging (default: 3)")
    p.add_argument("--out-dir",   default="results")
    return p.parse_args()


def main():
    args   = parse_args()
    dsn    = _build_dsn()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get patient IDs
    all_patients = get_all_patient_ids(dsn)
    patient_ids  = all_patients[:args.patients] if args.patients else all_patients
    log.info(f"Using {len(patient_ids)} patients: {patient_ids}")

    # Verify NPY directory exists
    npy_dir = Path(args.npy_dir)
    if not npy_dir.exists():
        log.error(f"NPY directory not found: {npy_dir}")
        log.error("Run etl_pipeline.py first, then adjust --npy-dir to point to the output.")
        return

    # Batch sizes to test (as required by A2)
    batch_sizes = [16, 32, 64, 128]

    log.info(f"Starting benchmark: {len(batch_sizes)} batch sizes × {args.runs} runs each")
    results = run_benchmark(dsn, npy_dir, patient_ids, batch_sizes, n_runs=args.runs)

    # Print table to console
    print_table(results)

    # Save CSV
    csv_path = out_dir / "a2_benchmark_results.csv"
    save_csv(results, csv_path)

    # Save full JSON
    json_path = out_dir / "a2_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "patient_ids":  patient_ids,
                "batch_sizes":  batch_sizes,
                "n_runs":       args.runs,
                "npy_dir":      str(npy_dir),
            },
            "results": results
        }, f, indent=2)
    log.info(f"JSON saved → {json_path}")

    log.info("\nA2 complete. Outputs:")
    log.info(f"  Table (CSV)  → {csv_path}")
    log.info(f"  Raw data     → {json_path}")


if __name__ == "__main__":
    main()