"""Selective Sleep-EDF Cassette download, EDF inspection, and report generation."""
import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mne

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sleep_edf"
OUTPUTS = ROOT / "outputs" / "data_preparation"
PAIRS = [
    ("SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf"), ("SC4002E0-PSG.edf", "SC4002EC-Hypnogram.edf"),
    ("SC4011E0-PSG.edf", "SC4011EH-Hypnogram.edf"), ("SC4012E0-PSG.edf", "SC4012EC-Hypnogram.edf"),
    ("SC4021E0-PSG.edf", "SC4021EH-Hypnogram.edf"), ("SC4022E0-PSG.edf", "SC4022EJ-Hypnogram.edf"),
]
EXPERIMENTAL_BASE = datetime(2000, 1, 1, tzinfo=timezone.utc)
RECORDING_METADATA_PATH = OUTPUTS / "sleep_recording_metadata.json"


def _record_length_seconds(raw) -> float:
    """Return the EDF data-record duration used by per-channel metadata."""

    extra = raw._raw_extras[0]
    value = extra.get("record_length", 1.0)
    values = value if hasattr(value, "__len__") and not isinstance(value, str) else [value]
    for candidate in values:
        try:
            candidate = float(candidate)
        except (TypeError, ValueError):
            continue
        if candidate > 0:
            return candidate
    return 1.0


def channel_sampling_rates(raw) -> dict[str, float]:
    """Read each channel's native rate from EDF samples-per-record metadata."""

    extra = raw._raw_extras[0]
    names = list(raw.ch_names)
    n_samps = extra.get("n_samps")
    record_length = _record_length_seconds(raw)
    if n_samps is not None and len(n_samps) == len(names):
        rates = {
            channel: float(samples) / record_length
            for channel, samples in zip(names, n_samps)
            if float(samples) > 0
        }
        if len(rates) == len(names) and all(rate > 0 for rate in rates.values()):
            return rates
    # EDF metadata is authoritative when present.  This fallback is only for
    # readers that do not expose samples-per-record, and is still per-channel
    # rather than an assumed Sleep-EDF constant.
    fallback = float(raw.info["sfreq"])
    return {channel: fallback for channel in names}


def meaningful_recording_rate(rates: dict[str, float]) -> float | None:
    """Return a recording rate only when every channel shares one native rate."""

    values = list(rates.values())
    if not values:
        return None
    first = values[0]
    return float(first) if all(abs(value - first) <= 1e-9 for value in values[1:]) else None


def _measurement_timestamp(raw) -> datetime | None:
    """Normalize an EDF measurement date, rejecting unusable metadata."""

    value = raw.info.get("meas_date")
    if value is None:
        return None
    if isinstance(value, tuple):
        try:
            value = datetime.fromtimestamp(float(value[0]) + float(value[1]) / 1e6, tz=timezone.utc)
        except (TypeError, ValueError, OSError, OverflowError):
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    # EDF dates before the Unix epoch are not useful for ordering this
    # experimental dataset; they are treated like missing metadata.
    return value if value.year >= 1970 else None


def timestamp_for_recording(raw, pair_index: int) -> tuple[datetime, str]:
    """Use EDF time when reliable, otherwise a deterministic isolated night."""

    measured = _measurement_timestamp(raw)
    if measured is not None:
        return measured, "edf_measurement"
    return EXPERIMENTAL_BASE + timedelta(days=2 * int(pair_index)), "normalized_experimental"


def write_recording_metadata(rows, path: Path = RECORDING_METADATA_PATH) -> Path:
    """Persist stable Sleep recording metadata for downstream population."""

    canonical = []
    for row in rows:
        rates = row.get("channel_sampling_rates", row.get("sampling_rates", {}))
        start = row.get("recording_start_time", row.get("measurement_start_time"))
        if isinstance(start, datetime):
            start = start.isoformat()
        canonical.append({
            "recording_id": str(row["recording_id"]),
            "channel_sampling_rates": {str(k): float(v) for k, v in rates.items()},
            "recording_sampling_rate_hz": row.get("recording_sampling_rate_hz"),
            "recording_start_time": start,
            "timestamp_provenance": row.get("timestamp_provenance", "normalized_experimental"),
            "duration_seconds": float(row["duration_seconds"] if "duration_seconds" in row else row.get("duration", 0.0)),
            "raw_sampling_rate_hz": row.get("raw_sampling_rate_hz"),
            "modeled_sampling_rates": row.get("modeled_sampling_rates", {}),
            "aux_sampling_rates": row.get("aux_sampling_rates", {}),
        })
    canonical.sort(key=lambda item: item["recording_id"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canonical, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_recording_metadata(path: Path = RECORDING_METADATA_PATH) -> dict[str, dict]:
    """Load the stable Sleep metadata sidecar keyed by recording ID."""

    path = Path(path)
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Sleep recording metadata sidecar: {path}") from exc
    if not isinstance(rows, list):
        raise ValueError("Sleep recording metadata sidecar must contain a list")
    return {str(row["recording_id"]): dict(row) for row in rows if isinstance(row, dict) and "recording_id" in row}


def download():
    url = os.getenv("PDM_SLEEP_EDF_ARCHIVE_URL")
    DATA.mkdir(parents=True, exist_ok=True)
    if not url:
        # Verified PhysioNet Sleep-EDF Expanded v1.0.0 fallback. Download only
        # the six requested PSG/hypnogram pairs.
        for psg, hyp in PAIRS:
            for name in (psg, hyp):
                target = DATA / name
                if not target.exists():
                    urllib.request.urlretrieve(
                        f"https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/{name}", target
                    )
        return
    archive = DATA / "sleep_cassette.zip"
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for psg, hyp in PAIRS:
            for name in (psg, hyp):
                if name not in names:
                    raise RuntimeError(f"Archive missing required file {name}")
                target = DATA / name
                if not target.exists():
                    with zf.open(names[name]) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)


def inspect_records():
    rows = []
    for pair_index, (psg_name, hyp_name) in enumerate(PAIRS):
        psg = DATA / psg_name
        hyp = DATA / hyp_name
        if not psg.exists() or not hyp.exists():
            continue
        raw = mne.io.read_raw_edf(psg, preload=False, verbose=False)
        ann = mne.read_annotations(hyp)
        rates = channel_sampling_rates(raw)
        measurement_time, timestamp_provenance = timestamp_for_recording(raw, pair_index)
        stages = [str(x).strip() for x in ann.description]
        durations = {}
        for stage, duration in zip(stages, ann.duration):
            durations[stage] = durations.get(stage, 0.0) + float(duration)
        rows.append({"recording_id": psg_name.removesuffix("-PSG.edf"), "subject": psg_name[:5],
                     "night": psg_name[5:7], "duration": raw.n_times / raw.info["sfreq"],
                     "channels": raw.ch_names, "sampling_rates": rates,
                     "recording_sampling_rate_hz": meaningful_recording_rate(rates),
                     "measurement_start_time": measurement_time.isoformat(),
                     "timestamp_provenance": timestamp_provenance,
                     "sleep_stage_counts": {k: stages.count(k) for k in sorted(set(stages))},
                     "sleep_stage_durations_seconds": durations})
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--download", action="store_true"); args = parser.parse_args()
    if args.download: download()
    rows = inspect_records(); OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "sleep_edf_inspection.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_recording_metadata(rows)
    print(f"Sleep-EDF records inspected: {len(rows)}")
    for row in rows: print(row["recording_id"], row["duration"], len(row["channels"]), row["sleep_stage_counts"])


if __name__ == "__main__": main()
