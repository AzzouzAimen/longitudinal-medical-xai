"""Selective Apnea-ECG download, WFDB inspection, and report generation."""
import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "apnea"
OUTPUTS = ROOT / "outputs" / "data_preparation"
RECORDS = ["a01", "a02", "a03", "c01", "c02", "c03"]


def _download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return False
    urllib.request.urlretrieve(url, destination)
    return True


def download():
    """Use a Kaggle archive URL (or KAGGLE_API_DATASET via kaggle CLI) when supplied.

    Kaggle URLs are intentionally configurable because private/download URLs expire.
    Set PDM_APNEA_ARCHIVE_URL to the benchmarked Apnea-ECG v1.0.0 mirror archive.
    """
    url = os.getenv("PDM_APNEA_ARCHIVE_URL")
    if not url:
        # Verified PhysioNet v1.0.0 fallback. This downloads only the requested
        # WFDB triplets and preserves the canonical raw file layout.
        for record in RECORDS:
            for suffix in (".hea", ".dat", ".apn"):
                _download(f"https://physionet.org/files/apnea-ecg/1.0.0/{record}{suffix}", DATA / f"{record}{suffix}")
        return
    archive = DATA / "apnea_ecg_v1.0.0.zip"
    _download(url, archive)
    with zipfile.ZipFile(archive) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        wanted = []
        for record in RECORDS:
            for suffix in (".hea", ".dat", ".apn"):
                if record + suffix in names:
                    wanted.append(names[record + suffix])
        if not wanted:
            raise RuntimeError("Archive contains none of the requested Apnea-ECG records")
        for name in wanted:
            target = DATA / Path(name).name
            if not target.exists():
                with zf.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def inspect_records():
    rows = []
    for record in RECORDS:
        base = DATA / record
        if not (base.with_suffix(".hea")).exists():
            continue
        rec = wfdb.rdrecord(str(base))
        ann = wfdb.rdann(str(base), "apn") if base.with_suffix(".apn").exists() else None
        labels = list(ann.symbol) if ann else []
        rows.append({
            "recording_id": record, "duration": rec.sig_len / rec.fs,
            "channels": rec.sig_name, "sampling_rates": [rec.fs] * rec.n_sig,
            "number_of_minute_annotations": len(labels), "A_count": labels.count("A"), "N_count": labels.count("N"),
            "Resp C": any(x.lower() == "resp c" for x in rec.sig_name),
            "Resp A": any(x.lower() == "resp a" for x in rec.sig_name),
            "Resp N": any(x.lower() == "resp n" for x in rec.sig_name),
            "SpO2": any("spo2" in x.lower() for x in rec.sig_name),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download()
    rows = inspect_records()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "apnea_inspection.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Apnea records inspected: {len(rows)}")
    for row in rows:
        print(row["recording_id"], row["duration"], row["channels"], "A=", row["A_count"], "N=", row["N_count"])


if __name__ == "__main__":
    main()
