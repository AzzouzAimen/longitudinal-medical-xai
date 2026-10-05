"""
Thermal database versus NumPy fetch benchmark.
Measures TimescaleDB (time_bucket) vs raw NPY fetch latency in ms/batch.
Output: results/benchmarks/
"""

import argparse, csv, json, logging, os, sys, time
from pathlib import Path
import numpy as np
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("db_benchmark")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "benchmarks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_dsn():
    return (f"postgresql://{os.getenv('TSDB_USER','postgres')}:{os.getenv('TSDB_PASSWORD','postgres')}"
            f"@{os.getenv('TSDB_HOST','localhost')}:{os.getenv('TSDB_PORT','5432')}"
            f"/{os.getenv('TSDB_DB','m6_thermal_tsdb')}")

def get_patient_ids(dsn, limit=None):
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute(f"SELECT patient_id::text FROM subjects ORDER BY patient_id{f' LIMIT {limit}' if limit else ''}")
        ids = [r[0] for r in cur.fetchall()]
    conn.close()
    return ids

def bench_tsdb(dsn, patient_ids, batch_size, n_runs=3):
    query = """
        SELECT w.window_id, w.patient_id, w.label,
               time_bucket('60 seconds', tr.timestamp) AS bucket,
               AVG(tr.left_temperature_norm), AVG(tr.right_temperature_norm)
        FROM windows_tsdb w
        JOIN thermal_readings tr
          ON tr.patient_id = w.patient_id
         AND tr.timestamp BETWEEN w.window_start AND w.window_end
        WHERE w.patient_id = ANY(%s::smallint[])
        GROUP BY w.window_id, w.patient_id, w.label, bucket
        ORDER BY w.window_id LIMIT %s
    """
    times = []
    conn = psycopg2.connect(dsn)
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(query, (patient_ids, batch_size))
            cur.fetchall()
        times.append((time.perf_counter() - t0) * 1000)
    conn.close()
    return {"method": "TimescaleDB", "batch_size": batch_size,
            "mean_ms": round(np.mean(times), 2), "std_ms": round(np.std(times), 2)}

def bench_npy(npy_dir, patient_ids, batch_size, n_runs=3):
    files = [npy_dir / f"patient_{str(p).zfill(2)}_windows.npy" for p in patient_ids if (npy_dir / f"patient_{str(p).zfill(2)}_windows.npy").exists()]
    if not files:
        log.error(f"No .npy files found in {npy_dir}")
        sys.exit(1)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for f in files:
            arr = np.load(f, mmap_mode="r")
            idx = np.random.randint(0, max(1, len(arr) - batch_size))
            arr[idx: idx + batch_size].copy()
        times.append((time.perf_counter() - t0) * 1000)
    return {"method": "NPY files", "batch_size": batch_size,
            "mean_ms": round(np.mean(times), 2), "std_ms": round(np.std(times), 2)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy-dir", default=os.getenv("NPY_DIR"))
    p.add_argument("--patients", type=int, default=None)
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()

    dsn        = get_dsn()
    npy_dir    = Path(args.npy_dir)
    patients   = get_patient_ids(dsn, args.patients)
    batch_sizes = [16, 32, 64, 128]

    tsdb_res, npy_res = [], []
    for bs in batch_sizes:
        log.info(f"Batch {bs}...")
        tsdb_res.append(bench_tsdb(dsn, patients, bs, args.runs))
        npy_res.append( bench_npy(npy_dir, patients, bs, args.runs))

    print(f"\n{'Batch':>6} | {'TSDB ms':>10} | {'NPY ms':>10} | {'Speedup':>10}")
    print("-" * 45)
    for t, n in zip(tsdb_res, npy_res):
        speedup = t["mean_ms"] / n["mean_ms"] if n["mean_ms"] else 0
        print(f"{t['batch_size']:>6} | {t['mean_ms']:>10.2f} | {n['mean_ms']:>10.2f} | {speedup:>9.2f}x")

    rows = [{"batch_size": t["batch_size"], "tsdb_mean_ms": t["mean_ms"],
             "npy_mean_ms": n["mean_ms"],
             "speedup": round(t["mean_ms"]/n["mean_ms"], 3) if n["mean_ms"] else None}
            for t, n in zip(tsdb_res, npy_res)]

    csv_path = OUTPUT_DIR / "db_pipeline_benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    json_path = OUTPUT_DIR / "db_pipeline_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump({"tsdb": tsdb_res, "npy": npy_res}, f, indent=2)

    log.info(f"Saved → {csv_path}, {json_path}")

if __name__ == "__main__":
    main()
