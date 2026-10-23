import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

audit_path = PROJECT_ROOT / "reports" / "dataset_audit.json"
with open(audit_path, "r", encoding="utf-8") as f:
    audit = json.load(f)

print("=== SUMMARY ===")
print(f"Passed: {audit['summary']['passed']}")
print(f"Failed: {audit['summary']['failed']}")
print(f"Gate Passed: {audit['summary']['gate_passed']}")

print()
print("=== PATIENT ID ANALYSIS ===")
pa = audit["patient_id_analysis"]
print(f"Total samples: {pa['total_samples']}")
print(f"Unique patients: {pa['unique_patients']}")
print(f"Multi-sample patients: {pa['multi_sample_patients']}")
print(f"Sample types: {pa['sample_types']}")

print()
print("=== CLASS DISTRIBUTION ===")
cd = audit["class_distribution"]
for cancer, info in cd["target_classes"].items():
    print(f"  {cancer} (label {info['label']}): {info['actual_count']} (expected: {info['expected_count']}) match={info['match']}")
print(f"  Total target: {cd['total_target_samples']}")
print(f"  Total all: {cd['total_all_samples']}")

print()
print("=== LABEL MAPPINGS ===")
lm = audit["label_mappings"]
print(f"Verified: {lm['mappings_verified']}")
print(f"Discrepancies: {lm['discrepancies']}")
for label, projects in lm["actual_mappings"].items():
    print(f"  Label {label}: {projects}")

print()
print("=== DATA QUALITY ===")
for mod, mq in audit["data_quality"].items():
    print(f"  {mod}: IDs_unique={mq['feature_ids_unique']}, "
          f"non_numeric={mq['non_numeric_cells']}, "
          f"non_finite={mq['non_finite_cells']}, "
          f"constant_first100={mq['constant_features_in_first_100_samples']}, "
          f"rows={mq['rows_checked']}")

print()
print("=== WARNINGS ===")
for w in audit.get("warnings", []):
    print(f"  {w}")

# Check patient multi-sample for our 4 target classes specifically
print()
print("=== TARGET CLASS PATIENT CHECK ===")
verification_file = PROJECT_ROOT / "reports" / "tcga_label_verification.csv"
if not verification_file.exists():
    verification_file = Path(r"D:\Personal Files\BCA\5TH SEM\Major_Project_Research\Dataset_Report\tcga_label_verification.csv")
target_labels = {2, 8, 13, 27}
from collections import Counter

patient_samples = {}
if verification_file.exists():
    with open(verification_file, "r") as f:
        reader = csv.DictReader(f)
    for row in reader:
        lbl = int(row["label"])
        if lbl in target_labels:
            pid = row["patient_id"]
            sid = row["sample_id"]
            if pid not in patient_samples:
                patient_samples[pid] = []
            patient_samples[pid].append(sid)

multi = {k: v for k, v in patient_samples.items() if len(v) > 1}
print(f"Target class patients: {len(patient_samples)}")
print(f"Multi-sample patients in target: {len(multi)}")
if multi:
    for pid, sids in list(multi.items())[:10]:
        print(f"  {pid}: {sids}")
