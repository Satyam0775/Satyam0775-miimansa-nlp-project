import os
import json
import pandas as pd

def read_ground_truth_spans(ann_file):
    """
    Read ground truth spans from CADEC .ann file in 'original/'.
    Returns set of (label, text).
    """
    spans = set()
    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            label_info = parts[1].split()
            label = label_info[0]
            text = parts[2].strip()
            spans.add((label, text))
    return spans


def load_predicted_spans(pred_json_file):
    """
    Load predicted spans from Task 2 JSON output.
    Expects list of dicts: {"id", "label", "span", "text"}.
    """
    with open(pred_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    spans = []
    for item in data:
        if isinstance(item, dict):
            spans.append((item["label"], item["text"]))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            spans.append((item[0], item[-1]))
    return spans


def normalize_span(span):
    """
    Normalize label + text (case + whitespace).
    """
    label, text = span
    return (label.strip().lower(), " ".join(text.strip().lower().split()))


def evaluate_predictions(filename, data_dir, output_dir):
    """
    Evaluate Task 2 predictions against ground truth (original/).
    Returns metrics dict and saves CSV + JSON in outputs/task3/.
    """
    gt_ann_file = os.path.join(data_dir, "original", f"{filename}.ann")
    pred_json_file = os.path.join(output_dir, "task2", f"{filename}_predictions.json")

    if not os.path.exists(gt_ann_file):
        raise FileNotFoundError(f"Ground truth not found: {gt_ann_file}")
    if not os.path.exists(pred_json_file):
        raise FileNotFoundError(f"Predictions not found: {pred_json_file}")

    gt_spans = read_ground_truth_spans(gt_ann_file)
    pred_spans = load_predicted_spans(pred_json_file)

    # Normalize
    gt_spans_norm = set(normalize_span(s) for s in gt_spans)
    pred_spans_norm = set(normalize_span(s) for s in pred_spans)

    # Metrics
    tp = len(gt_spans_norm & pred_spans_norm)
    fp = len(pred_spans_norm - gt_spans_norm)
    fn = len(gt_spans_norm - pred_spans_norm)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {
        "File": filename,
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

    # Save
    out_dir = os.path.join(output_dir, "task3")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{filename}_metrics.csv")
    json_path = os.path.join(out_dir, f"{filename}_metrics.json")

    pd.DataFrame([results]).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Task 3 results saved:\n- {csv_path}\n- {json_path}")
    return results
