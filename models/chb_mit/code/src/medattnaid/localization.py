from collections import Counter, defaultdict
from .metrics import wilson_interval


SENSOR_TO_REGION = {
    "FP1-F7": "left_frontal",
    "F7-T7": "left_temporal",
    "T7-P7": "left_temporo_parietal",
    "P7-O1": "left_posterior",
    "FP1-F3": "left_frontal",
    "F3-C3": "left_central",
    "C3-P3": "left_central_parietal",
    "P3-O1": "left_posterior",
    "FP2-F4": "right_frontal",
    "F4-C4": "right_central",
    "C4-P4": "right_central_parietal",
    "P4-O2": "right_posterior",
    "FP2-F8": "right_frontal",
    "F8-T8": "right_temporal",
    "T8-P8": "right_temporo_parietal",
    "P8-O2": "right_posterior",
    "FZ-CZ": "midline_central",
    "CZ-PZ": "midline_parietal",
}


GROUND_TRUTH_REGIONS = {
    "chb01_03": {"side": "LEFT", "region": "left_temporal"},
    "chb01_04": {"side": "LEFT", "region": "left_temporal"},
}


def localize(window_results, sensor_to_region=None, ground_truth=None):
    sensor_to_region = sensor_to_region or SENSOR_TO_REGION
    ground_truth = ground_truth or GROUND_TRUTH_REGIONS
    per_recording = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in window_results:
        key = row["recording_id"]
        channel = row["channel"]
        per_recording[key][channel][1] += 1
        if row["is_anomaly"]:
            per_recording[key][channel][0] += 1
    correct_side = 0
    correct_region = 0
    total = 0
    details = []
    for recording_id, channels in per_recording.items():
        if recording_id not in ground_truth:
            continue
        ranked = sorted(
            channels.items(),
            key=lambda item: item[1][0] / max(item[1][1], 1),
            reverse=True,
        )
        top_regions = [sensor_to_region.get(ch, "unknown") for ch, _ in ranked[:5]]
        side_counts = Counter("LEFT" if r.startswith("left") else "RIGHT" if r.startswith("right") else "MIDLINE" for r in top_regions)
        predicted_side = side_counts.most_common(1)[0][0]
        predicted_region = Counter(top_regions).most_common(1)[0][0]
        gt = ground_truth[recording_id]
        correct_side += int(predicted_side == gt["side"])
        correct_region += int(predicted_region == gt["region"])
        total += 1
        details.append({
            "recording_id": recording_id,
            "predicted_side": predicted_side,
            "ground_truth_side": gt["side"],
            "predicted_region": predicted_region,
            "ground_truth_region": gt["region"],
        })
    side_ci = wilson_interval(correct_side, total)
    region_ci = wilson_interval(correct_region, total)
    return {
        "details": details,
        "side": {"correct": correct_side, "total": total, "accuracy": correct_side / total if total else 0.0, "ci": side_ci},
        "region": {"correct": correct_region, "total": total, "accuracy": correct_region / total if total else 0.0, "ci": region_ci},
    }
