import os
import json
import pandas as pd
from evaluation import read_ground_truth_spans, load_predicted_spans, evaluate_predictions

# -------------------------------
# Paths
# -------------------------------
DATA_DIR = r"C:\Users\satya\Downloads\Miimansa Problem\Assignment\data\CADEC.v2"
OUTPUT_DIR = r"C:\Users\satya\Downloads\Miimansa Problem\Assignment\outputs"
SAMPLED_FILES_LIST = os.path.join(OUTPUT_DIR, "task5", "step5_sampled_files.txt")

TASK5_OUT_CSV = os.path.join(OUTPUT_DIR, "task5", "task5_metrics.csv")
TASK5_OUT_JSON = os.path.join(OUTPUT_DIR, "task5", "task5_metrics.json")
os.makedirs(os.path.dirname(TASK5_OUT_CSV), exist_ok=True)

# -------------------------------
# Load sampled file list
# -------------------------------
with open(SAMPLED_FILES_LIST, "r", encoding="utf-8") as f:
    sampled_files = [line.strip() for line in f.readlines()]

print(f"📄 Evaluating {len(sampled_files)} files from sampled list...")

# -------------------------------
# Run evaluation for each file
# -------------------------------
all_results = []
for filename in sampled_files:
    gt_ann_file = os.path.join(DATA_DIR, "original", f"{filename}.ann")
    pred_json_file = os.path.join(OUTPUT_DIR, "task2", f"{filename}_predictions.json")

    if not os.path.exists(gt_ann_file) or not os.path.exists(pred_json_file):
        print(f"⚠️ Skipping {filename}, missing files.")
        continue

    results = evaluate_predictions(filename, DATA_DIR, OUTPUT_DIR)
    all_results.append(results)

# -------------------------------
# Save aggregated results
# -------------------------------
df = pd.DataFrame(all_results)
df.to_csv(TASK5_OUT_CSV, index=False)
with open(TASK5_OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print("\n✅ Task 5 evaluation complete!")
print(f"- Results saved to: {TASK5_OUT_CSV}")
print(f"- Results saved to: {TASK5_OUT_JSON}")

# Show overall averages
print("\n--- Overall Metrics (across 50 files) ---")
print(df[["Precision", "Recall", "F1-score"]].mean())
