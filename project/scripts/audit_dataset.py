"""
Stage A: Comprehensive Dataset Audit Script
=============================================
Verifies sample alignment, label mappings, data integrity, feature statistics,
and class distributions across all four omics modalities.

Outputs:
  - reports/dataset_audit.json
  - reports/class_distribution.csv
  - reports/sample_alignment_report.csv
"""

import csv
import json
import os
import sys
import math
from pathlib import Path
from collections import Counter, OrderedDict
from datetime import datetime

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = Path(
    r"D:\Personal Files\BCA\5TH SEM\Major_Project_Research\Dataset_2"
    r"\MLOmics\Main_Dataset\Classification_datasets\Pan-cancer\Original"
)

REPORT_DIR = Path(
    r"D:\Personal Files\BCA\5TH SEM\Major_Project_Research\Dataset_Report"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODALITY_FILES = OrderedDict([
    ("mRNA",  DATA_DIR / "Pan-cancer_mRNA.csv"),
    ("miRNA", DATA_DIR / "Pan-cancer_miRNA.csv"),
    ("CNV",   DATA_DIR / "Pan-cancer_CNV.csv"),
    ("Methy", DATA_DIR / "Pan-cancer_Methy.csv"),
])

LABEL_FILE = DATA_DIR / "Pan-cancer_label_num.csv"
VERIFICATION_FILE = REPORT_DIR / "tcga_label_verification.csv"
CROSSTAB_FILE = REPORT_DIR / "tcga_label_crosstab.csv"

# Expected label→project mappings for our 4 classes
EXPECTED_MAPPINGS = {
    2:  "TCGA-BRCA",
    8:  "TCGA-HNSC",
    13: "TCGA-LGG",
    27: "TCGA-THCA",
}

EXPECTED_COUNTS = {
    "BRCA": 757,
    "HNSC": 506,
    "LGG":  506,
    "THCA": 494,
}

TARGET_LABELS = {2, 8, 13, 27}

LABEL_TO_CANCER = {
    2:  "BRCA",
    8:  "HNSC",
    13: "LGG",
    27: "THCA",
}


# ─── Utility Functions ──────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[AUDIT] {msg}")


def read_csv_header(filepath: Path) -> list[str]:
    """Read only the header row of a CSV file."""
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def count_csv_rows(filepath: Path) -> int:
    """Count data rows (excluding header) without loading into memory."""
    with open(filepath, "r", newline="") as f:
        return sum(1 for _ in f) - 1


def stream_csv_rows(filepath: Path):
    """Yield rows as lists, one at a time, including header."""
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            yield row


# ─── Audit Checks ───────────────────────────────────────────────────────────

def audit_label_file(audit: dict) -> tuple[list[str], list[int]]:
    """Audit the label file and return (sample_ids_from_labels, labels)."""
    log("Checking label file...")
    
    header = read_csv_header(LABEL_FILE)
    row_count = count_csv_rows(LABEL_FILE)
    
    audit["label_file"] = {
        "path": str(LABEL_FILE),
        "header": header,
        "data_rows": row_count,
        "expected_header": ["Label"],
    }
    
    labels = []
    with open(LABEL_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            labels.append(int(row[0]))
    
    unique_labels = sorted(set(labels))
    label_counts = Counter(labels)
    
    audit["label_file"]["unique_labels"] = unique_labels
    audit["label_file"]["label_counts"] = dict(sorted(label_counts.items()))
    audit["label_file"]["total_samples"] = len(labels)
    
    log(f"  Label file: {len(labels)} rows, {len(unique_labels)} unique labels")
    log(f"  Labels: {unique_labels}")
    
    return labels


def audit_modality_headers(audit: dict) -> dict[str, list[str]]:
    """Audit headers of all modality files. Return {modality: [sample_ids]}."""
    log("Checking modality file headers...")
    
    modality_headers = {}
    modality_sample_ids = {}
    
    for mod_name, mod_file in MODALITY_FILES.items():
        header = read_csv_header(mod_file)
        feature_col = header[0]
        sample_ids = header[1:]
        
        row_count = count_csv_rows(mod_file)
        
        modality_headers[mod_name] = {
            "path": str(mod_file),
            "feature_column_name": feature_col,
            "total_columns": len(header),
            "sample_count": len(sample_ids),
            "feature_count_rows": row_count,
            "first_5_samples": sample_ids[:5],
            "last_3_samples": sample_ids[-3:],
        }
        
        modality_sample_ids[mod_name] = sample_ids
        
        log(f"  {mod_name}: {len(header)} cols ({len(sample_ids)} samples), "
            f"{row_count} feature rows, feature_col='{feature_col}'")
    
    audit["modalities"] = modality_headers
    return modality_sample_ids


def audit_sample_alignment(
    audit: dict,
    modality_sample_ids: dict[str, list[str]],
    labels: list[int],
) -> list[str]:
    """Verify that sample IDs are identical across all modalities and label file."""
    log("Checking sample alignment across modalities...")
    
    alignment = {
        "all_modalities_identical": True,
        "label_count_matches_samples": True,
        "details": {},
        "issues": [],
    }
    
    # Reference: first modality
    ref_name = list(modality_sample_ids.keys())[0]
    ref_ids = modality_sample_ids[ref_name]
    
    for mod_name, sample_ids in modality_sample_ids.items():
        identical = (sample_ids == ref_ids)
        same_set = (set(sample_ids) == set(ref_ids))
        has_duplicates = len(sample_ids) != len(set(sample_ids))
        has_blanks = any(s.strip() == "" for s in sample_ids)
        
        alignment["details"][mod_name] = {
            "sample_count": len(sample_ids),
            "identical_order_to_reference": identical,
            "same_sample_set_as_reference": same_set,
            "has_duplicate_sample_ids": has_duplicates,
            "has_blank_sample_ids": has_blanks,
        }
        
        if not identical:
            alignment["all_modalities_identical"] = False
            if not same_set:
                alignment["issues"].append(
                    f"{mod_name} has different sample IDs than {ref_name}"
                )
            else:
                alignment["issues"].append(
                    f"{mod_name} has same samples but different order than {ref_name}"
                )
        
        if has_duplicates:
            alignment["issues"].append(f"{mod_name} has duplicate sample IDs")
        
        if has_blanks:
            alignment["issues"].append(f"{mod_name} has blank sample IDs")
    
    # Check label count
    if len(labels) != len(ref_ids):
        alignment["label_count_matches_samples"] = False
        alignment["issues"].append(
            f"Label file has {len(labels)} rows but modalities have {len(ref_ids)} samples"
        )
    
    audit["sample_alignment"] = alignment
    
    if alignment["all_modalities_identical"]:
        log("  ✓ All modalities share identical sample IDs in identical order")
    else:
        log("  ✗ Sample alignment issues found!")
        for issue in alignment["issues"]:
            log(f"    - {issue}")
    
    return ref_ids


def audit_label_mappings(
    audit: dict,
    sample_ids: list[str],
    labels: list[int],
) -> None:
    """Verify label→TCGA project mappings using the verification CSV."""
    log("Verifying label→TCGA project mappings...")
    
    mapping_audit = {
        "verification_file_exists": VERIFICATION_FILE.exists(),
        "crosstab_file_exists": CROSSTAB_FILE.exists(),
        "mappings_verified": False,
        "expected_mappings": {str(k): v for k, v in EXPECTED_MAPPINGS.items()},
        "actual_mappings": {},
        "discrepancies": [],
    }
    
    if not VERIFICATION_FILE.exists():
        mapping_audit["discrepancies"].append("Verification CSV not found")
        audit["label_mappings"] = mapping_audit
        log("  ✗ Verification file not found!")
        return
    
    # Read verification CSV
    verification_header = read_csv_header(VERIFICATION_FILE)
    log(f"  Verification CSV columns: {verification_header}")
    
    mapping_audit["verification_columns"] = verification_header
    
    # Expected: sample_id, label, patient_id, tcga_project
    expected_cols = {"sample_id", "label", "patient_id", "tcga_project"}
    actual_cols = set(verification_header)
    if not expected_cols.issubset(actual_cols):
        missing = expected_cols - actual_cols
        mapping_audit["discrepancies"].append(
            f"Verification CSV missing columns: {missing}"
        )
    
    # Build label→project mapping from verification data
    label_project_pairs = Counter()
    sample_count_in_verification = 0
    unmatched_samples = 0
    
    col_indices = {col: i for i, col in enumerate(verification_header)}
    
    with open(VERIFICATION_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            sample_count_in_verification += 1
            label_val = int(row[col_indices["label"]])
            tcga_project = row[col_indices["tcga_project"]]
            if tcga_project:
                label_project_pairs[(label_val, tcga_project)] += 1
            else:
                unmatched_samples += 1
    
    mapping_audit["verification_sample_count"] = sample_count_in_verification
    mapping_audit["unmatched_samples"] = unmatched_samples
    
    # Check each target label
    for label_num in sorted(TARGET_LABELS):
        projects_for_label = {
            proj: count
            for (lbl, proj), count in label_project_pairs.items()
            if lbl == label_num
        }
        
        mapping_audit["actual_mappings"][str(label_num)] = projects_for_label
        
        expected_project = EXPECTED_MAPPINGS.get(label_num)
        
        if len(projects_for_label) == 0:
            mapping_audit["discrepancies"].append(
                f"Label {label_num}: no TCGA project found in verification"
            )
        elif len(projects_for_label) > 1:
            mapping_audit["discrepancies"].append(
                f"Label {label_num}: maps to multiple projects: {projects_for_label}"
            )
        else:
            actual_project = list(projects_for_label.keys())[0]
            if actual_project != expected_project:
                mapping_audit["discrepancies"].append(
                    f"Label {label_num}: expected {expected_project}, "
                    f"got {actual_project}"
                )
    
    if not mapping_audit["discrepancies"]:
        mapping_audit["mappings_verified"] = True
        log("  ✓ All label→TCGA project mappings verified")
    else:
        log("  ✗ Mapping discrepancies found:")
        for d in mapping_audit["discrepancies"]:
            log(f"    - {d}")
    
    audit["label_mappings"] = mapping_audit


def audit_class_distribution(
    audit: dict,
    labels: list[int],
) -> None:
    """Check class counts for the four target cancer types."""
    log("Checking class distributions...")
    
    label_counts = Counter(labels)
    
    target_distribution = {}
    discrepancies = []
    
    for label_num in sorted(TARGET_LABELS):
        cancer_name = LABEL_TO_CANCER[label_num]
        actual_count = label_counts.get(label_num, 0)
        expected_count = EXPECTED_COUNTS[cancer_name]
        
        target_distribution[cancer_name] = {
            "label": label_num,
            "actual_count": actual_count,
            "expected_count": expected_count,
            "match": actual_count == expected_count,
        }
        
        if actual_count != expected_count:
            discrepancies.append(
                f"{cancer_name} (label {label_num}): "
                f"expected {expected_count}, got {actual_count}"
            )
    
    total_target = sum(
        target_distribution[c]["actual_count"]
        for c in target_distribution
    )
    
    class_dist = {
        "target_classes": target_distribution,
        "total_target_samples": total_target,
        "total_all_samples": len(labels),
        "non_target_samples": len(labels) - total_target,
        "all_counts_match": len(discrepancies) == 0,
        "discrepancies": discrepancies,
    }
    
    audit["class_distribution"] = class_dist
    
    if class_dist["all_counts_match"]:
        log(f"  ✓ All target class counts match expectations")
    else:
        for d in discrepancies:
            log(f"  ✗ {d}")
    
    log(f"  Total target: {total_target}, Total all: {len(labels)}, "
        f"Non-target: {len(labels) - total_target}")
    
    # Save class_distribution.csv
    csv_path = OUTPUT_DIR / "class_distribution.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cancer_type", "numeric_label", "tcga_project",
            "actual_count", "expected_count", "match"
        ])
        for cancer_name, info in target_distribution.items():
            writer.writerow([
                cancer_name,
                info["label"],
                EXPECTED_MAPPINGS[info["label"]],
                info["actual_count"],
                info["expected_count"],
                info["match"],
            ])
    
    log(f"  Saved: {csv_path}")


def audit_data_quality_streaming(audit: dict) -> None:
    """Stream through each modality to check for data quality issues.
    
    Checks per modality:
    - Feature identifier uniqueness
    - Non-numeric values (sample a few rows)
    - Non-finite values (NaN, Inf)
    - Constant features (all same value across first N samples)
    - Row width consistency
    """
    log("Streaming data quality checks (sampling approach)...")
    
    quality = {}
    
    for mod_name, mod_file in MODALITY_FILES.items():
        log(f"  Checking {mod_name}...")
        
        mod_quality = {
            "feature_ids_unique": True,
            "duplicate_feature_ids": [],
            "non_numeric_cells": 0,
            "non_finite_cells": 0,
            "inconsistent_row_widths": [],
            "constant_features": 0,
            "rows_checked": 0,
            "sample_non_numeric_values": [],
        }
        
        feature_ids = []
        expected_width = None
        
        # Track feature value ranges for constant detection
        feature_min = {}
        feature_max = {}
        
        with open(mod_file, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_width = len(header)
            
            for row_idx, row in enumerate(reader):
                # Check row width
                if len(row) != expected_width:
                    mod_quality["inconsistent_row_widths"].append({
                        "row": row_idx + 2,
                        "width": len(row),
                        "expected": expected_width,
                    })
                
                # Feature ID
                feature_id = row[0]
                feature_ids.append(feature_id)
                
                # Check numeric values (sample every 10th row for speed)
                if row_idx % 10 == 0:
                    for col_idx in range(1, min(len(row), expected_width)):
                        val_str = row[col_idx].strip()
                        if val_str == "":
                            mod_quality["non_numeric_cells"] += 1
                            continue
                        try:
                            val = float(val_str)
                            if not math.isfinite(val):
                                mod_quality["non_finite_cells"] += 1
                        except ValueError:
                            mod_quality["non_numeric_cells"] += 1
                            if len(mod_quality["sample_non_numeric_values"]) < 5:
                                mod_quality["sample_non_numeric_values"].append({
                                    "row": row_idx + 2,
                                    "col": col_idx,
                                    "value": val_str[:50],
                                })
                
                # Track min/max for constant feature detection (first 100 samples)
                for col_idx in range(1, min(len(row), 101)):
                    try:
                        val = float(row[col_idx])
                        if feature_id not in feature_min:
                            feature_min[feature_id] = val
                            feature_max[feature_id] = val
                        else:
                            if val < feature_min[feature_id]:
                                feature_min[feature_id] = val
                            if val > feature_max[feature_id]:
                                feature_max[feature_id] = val
                    except (ValueError, IndexError):
                        pass
                
                mod_quality["rows_checked"] = row_idx + 1
        
        # Check feature ID uniqueness
        feature_id_counts = Counter(feature_ids)
        duplicates = {k: v for k, v in feature_id_counts.items() if v > 1}
        if duplicates:
            mod_quality["feature_ids_unique"] = False
            mod_quality["duplicate_feature_ids"] = [
                {"id": k, "count": v}
                for k, v in sorted(duplicates.items(), key=lambda x: -x[1])[:10]
            ]
        
        # Count constant features (based on first 100 samples)
        constant_count = sum(
            1 for fid in feature_min
            if feature_min[fid] == feature_max[fid]
        )
        mod_quality["constant_features_in_first_100_samples"] = constant_count
        mod_quality["total_features_checked"] = len(feature_min)
        
        quality[mod_name] = mod_quality
        
        log(f"    Features: {mod_quality['rows_checked']}, "
            f"IDs unique: {mod_quality['feature_ids_unique']}, "
            f"Non-numeric (sampled): {mod_quality['non_numeric_cells']}, "
            f"Non-finite (sampled): {mod_quality['non_finite_cells']}, "
            f"Constant (first 100): {constant_count}")
    
    audit["data_quality"] = quality


def audit_patient_id_structure(
    audit: dict,
    sample_ids: list[str],
) -> None:
    """Analyze TCGA barcode structure to check for multi-sample patients."""
    log("Analyzing patient ID structure...")
    
    # TCGA barcode: TCGA-XX-XXXX-01
    # Patient ID: TCGA-XX-XXXX (first 3 parts)
    # Sample type: 01 = primary tumor, 06 = metastatic, etc.
    
    patient_to_samples = {}
    sample_type_counter = Counter()
    malformed_barcodes = []
    
    for sid in sample_ids:
        parts = sid.split("-")
        if len(parts) >= 4 and parts[0] == "TCGA":
            patient_id = "-".join(parts[:3])
            sample_type = parts[3][:2] if len(parts[3]) >= 2 else parts[3]
            
            if patient_id not in patient_to_samples:
                patient_to_samples[patient_id] = []
            patient_to_samples[patient_id].append(sid)
            sample_type_counter[sample_type] += 1
        else:
            malformed_barcodes.append(sid)
    
    multi_sample_patients = {
        pid: sids for pid, sids in patient_to_samples.items()
        if len(sids) > 1
    }
    
    patient_analysis = {
        "total_samples": len(sample_ids),
        "unique_patients": len(patient_to_samples),
        "multi_sample_patients": len(multi_sample_patients),
        "sample_types": dict(sample_type_counter.most_common()),
        "malformed_barcodes_count": len(malformed_barcodes),
        "malformed_barcodes_examples": malformed_barcodes[:10],
    }
    
    if multi_sample_patients:
        patient_analysis["multi_sample_examples"] = {
            pid: sids
            for pid, sids in list(multi_sample_patients.items())[:10]
        }
    
    audit["patient_id_analysis"] = patient_analysis
    
    log(f"  Unique patients: {len(patient_to_samples)}")
    log(f"  Multi-sample patients: {len(multi_sample_patients)}")
    log(f"  Sample types: {dict(sample_type_counter.most_common())}")
    
    if multi_sample_patients:
        log("  ⚠ Multi-sample patients found — patient-level splitting required!")
        for pid, sids in list(multi_sample_patients.items())[:5]:
            log(f"    {pid}: {sids}")


def save_sample_alignment_report(
    sample_ids: list[str],
    labels: list[int],
    modality_sample_ids: dict[str, list[str]],
) -> None:
    """Save per-sample alignment report."""
    log("Saving sample alignment report...")
    
    csv_path = OUTPUT_DIR / "sample_alignment_report.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        mod_names = list(modality_sample_ids.keys())
        header = ["position", "sample_id_from_mRNA", "label"]
        for mod_name in mod_names:
            header.append(f"{mod_name}_matches")
        header.append("all_aligned")
        
        writer.writerow(header)
        
        ref_ids = modality_sample_ids[mod_names[0]]
        
        for i in range(len(ref_ids)):
            row = [i, ref_ids[i], labels[i] if i < len(labels) else "MISSING"]
            
            all_match = True
            for mod_name in mod_names:
                mod_ids = modality_sample_ids[mod_name]
                matches = (i < len(mod_ids) and mod_ids[i] == ref_ids[i])
                row.append(matches)
                if not matches:
                    all_match = False
            
            row.append(all_match)
            writer.writerow(row)
    
    log(f"  Saved: {csv_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("STAGE A: COMPREHENSIVE DATASET AUDIT")
    log("=" * 70)
    log(f"Timestamp: {datetime.now().isoformat()}")
    log(f"Data directory: {DATA_DIR}")
    log(f"Report directory: {REPORT_DIR}")
    log(f"Output directory: {OUTPUT_DIR}")
    log("")
    
    audit = {
        "timestamp": datetime.now().isoformat(),
        "data_directory": str(DATA_DIR),
        "report_directory": str(REPORT_DIR),
        "checks_passed": [],
        "checks_failed": [],
        "checks_incomplete": [],
        "warnings": [],
    }
    
    # 1. Check file existence
    log("--- Check 1: File Existence ---")
    for name, path in MODALITY_FILES.items():
        if path.exists():
            audit["checks_passed"].append(f"{name} file exists: {path.name}")
        else:
            audit["checks_failed"].append(f"{name} file NOT FOUND: {path}")
    
    if LABEL_FILE.exists():
        audit["checks_passed"].append(f"Label file exists: {LABEL_FILE.name}")
    else:
        audit["checks_failed"].append(f"Label file NOT FOUND: {LABEL_FILE}")
        log("FATAL: Label file not found. Cannot proceed.")
        return
    
    # 2. Audit label file
    log("\n--- Check 2: Label File ---")
    labels = audit_label_file(audit)
    
    # 3. Audit modality headers
    log("\n--- Check 3: Modality Headers ---")
    modality_sample_ids = audit_modality_headers(audit)
    
    # 4. Check feature column naming inconsistency
    log("\n--- Check 4: Feature Column Naming ---")
    feature_col_names = {
        mod: info["feature_column_name"]
        for mod, info in audit["modalities"].items()
    }
    unique_names = set(feature_col_names.values())
    if len(unique_names) == 1:
        audit["checks_passed"].append(
            f"All modalities use same feature column name: {unique_names.pop()}"
        )
    else:
        audit["warnings"].append(
            f"Feature column naming inconsistency: {feature_col_names}"
        )
        log(f"  ⚠ Inconsistent feature column names: {feature_col_names}")
    
    # 5. Sample alignment
    log("\n--- Check 5: Sample Alignment ---")
    sample_ids = audit_sample_alignment(audit, modality_sample_ids, labels)
    
    if audit["sample_alignment"]["all_modalities_identical"]:
        audit["checks_passed"].append("All modalities have identical sample IDs in identical order")
    else:
        audit["checks_failed"].append(
            f"Sample alignment issues: {audit['sample_alignment']['issues']}"
        )
    
    if audit["sample_alignment"]["label_count_matches_samples"]:
        audit["checks_passed"].append("Label count matches sample count")
    else:
        audit["checks_failed"].append("Label count does not match sample count")
    
    # 6. Label mapping verification
    log("\n--- Check 6: Label→TCGA Mapping ---")
    audit_label_mappings(audit, sample_ids, labels)
    
    if audit["label_mappings"]["mappings_verified"]:
        audit["checks_passed"].append("All 4 target label→TCGA project mappings verified")
    else:
        if audit["label_mappings"]["discrepancies"]:
            audit["checks_failed"].extend(audit["label_mappings"]["discrepancies"])
        else:
            audit["checks_incomplete"].append("Label mapping verification incomplete")
    
    # 7. Class distribution
    log("\n--- Check 7: Class Distribution ---")
    audit_class_distribution(audit, labels)
    
    if audit["class_distribution"]["all_counts_match"]:
        audit["checks_passed"].append("All target class counts match expectations")
    else:
        audit["checks_failed"].extend(audit["class_distribution"]["discrepancies"])
    
    # 8. Data quality checks
    log("\n--- Check 8: Data Quality (Streaming) ---")
    audit_data_quality_streaming(audit)
    
    for mod_name, mq in audit["data_quality"].items():
        if mq["feature_ids_unique"]:
            audit["checks_passed"].append(f"{mod_name}: feature IDs unique")
        else:
            audit["checks_failed"].append(
                f"{mod_name}: duplicate feature IDs: {mq['duplicate_feature_ids'][:3]}"
            )
        
        if mq["non_numeric_cells"] == 0:
            audit["checks_passed"].append(f"{mod_name}: no non-numeric cells (sampled)")
        else:
            audit["warnings"].append(
                f"{mod_name}: {mq['non_numeric_cells']} non-numeric cells (sampled)"
            )
        
        if mq["non_finite_cells"] == 0:
            audit["checks_passed"].append(f"{mod_name}: no non-finite cells (sampled)")
        else:
            audit["warnings"].append(
                f"{mod_name}: {mq['non_finite_cells']} non-finite cells (sampled)"
            )
        
        if not mq["inconsistent_row_widths"]:
            audit["checks_passed"].append(f"{mod_name}: consistent row widths")
        else:
            audit["checks_failed"].append(
                f"{mod_name}: inconsistent row widths at rows {mq['inconsistent_row_widths'][:5]}"
            )
    
    # 9. Patient ID analysis
    log("\n--- Check 9: Patient ID Structure ---")
    audit_patient_id_structure(audit, sample_ids)
    
    if audit["patient_id_analysis"]["multi_sample_patients"] > 0:
        audit["warnings"].append(
            f"{audit['patient_id_analysis']['multi_sample_patients']} patients "
            f"have multiple samples — patient-level splitting required"
        )
    else:
        audit["checks_passed"].append("No multi-sample patients detected")
    
    # 10. Save alignment report
    log("\n--- Saving Reports ---")
    save_sample_alignment_report(sample_ids, labels, modality_sample_ids)
    
    # 11. Summary
    log("\n" + "=" * 70)
    log("AUDIT SUMMARY")
    log("=" * 70)
    log(f"Checks passed:     {len(audit['checks_passed'])}")
    log(f"Checks failed:     {len(audit['checks_failed'])}")
    log(f"Checks incomplete: {len(audit['checks_incomplete'])}")
    log(f"Warnings:          {len(audit['warnings'])}")
    
    if audit["checks_failed"]:
        log("\nFAILED CHECKS:")
        for f in audit["checks_failed"]:
            log(f"  ✗ {f}")
    
    if audit["warnings"]:
        log("\nWARNINGS:")
        for w in audit["warnings"]:
            log(f"  ⚠ {w}")
    
    audit["summary"] = {
        "passed": len(audit["checks_passed"]),
        "failed": len(audit["checks_failed"]),
        "incomplete": len(audit["checks_incomplete"]),
        "warnings": len(audit["warnings"]),
        "gate_passed": len(audit["checks_failed"]) == 0,
    }
    
    # Save JSON audit report
    json_path = OUTPUT_DIR / "dataset_audit.json"
    with open(json_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    log(f"\nSaved: {json_path}")
    
    if audit["summary"]["gate_passed"]:
        log("\n✓ STAGE A GATE: PASSED — All required checks verified.")
    else:
        log("\n✗ STAGE A GATE: FAILED — Review failed checks before proceeding.")
    
    return audit


if __name__ == "__main__":
    main()
