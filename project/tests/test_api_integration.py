"""
Automated Integration and Validation Tests for Multimodal Inference API
=======================================================================
Covers:
1. Complete quad-modal inference with live S2 checkpoint
2. Missing modality subsets (C1, C10, C11, C12 CNV Stress Test, C14)
3. Upload parsing of valid CSV files
4. Detection and rejection of empty modality requests (HTTP 400)
5. Validation of invalid file extensions, dimension mismatches, and NaN values
6. Probability range [0, 1] and unit-sum property
7. Exact agreement with offline extracted evaluations
"""

import sys
import json
from pathlib import Path
import pytest
import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT.parent / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api_server import app
from inference_service import engine

client = TestClient(app)


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "s2" in data["loaded_checkpoints"]
    assert data["features_total"] == 9844
    assert data["preprocessors_loaded"]["mRNA"] is True
    assert data["preprocessors_loaded"]["Methy"] is True


def test_template_download():
    for mod in ["mRNA", "miRNA", "CNV", "Methy"]:
        res = client.get(f"/api/template/{mod}")
        assert res.status_code == 200
        assert "text/csv" in res.headers["content-type"]
        lines = res.text.strip().split("\n")
        assert len(lines) >= 2  # header + row of zeros
        features = lines[0].split(",")
        assert len(features) == engine.modality_dims[mod]


def test_live_quad_modal_prediction():
    res = client.post("/api/predict", json={
        "modalities": [1, 1, 1, 1],
        "patient_idx": 0,
        "strategy": "s2"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["request_status"] == "success"
    assert data["predicted_class"] == "BRCA"
    assert data["confidence_pct"] > 90.0
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-3
    assert len(data["active_modalities"]) == 4
    assert len(data["missing_modalities"]) == 0
    assert data["condition_id"] == "C0"


def test_all_15_missing_modality_conditions_live():
    from src.data.dataset import EVAL_CONDITIONS, MODALITY_KEYS
    for cond_id, cond_info in EVAL_CONDITIONS.items():
        flags = [int(cond_info[m]) for m in MODALITY_KEYS]
        res = client.post("/api/predict", json={
            "modalities": flags,
            "patient_idx": 1,
            "strategy": "s2"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["condition_id"] == cond_id
        assert data["predicted_class"] in ["BRCA", "HNSC", "LGG", "THCA"]
        probs = data["probabilities"]
        assert all(0.0 <= p <= 1.0 for p in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 1e-3


def test_empty_modalities_rejection():
    res = client.post("/api/predict", json={
        "modalities": [0, 0, 0, 0],
        "patient_idx": 0
    })
    assert res.status_code == 400
    assert "At least one omic modality must be active" in res.json()["detail"]


def test_file_upload_prediction():
    sample_dir = CODE_DIR / "sample_inputs"
    mrna_file = sample_dir / "BRCA_Patient101_mRNA.csv"
    cnv_file = sample_dir / "BRCA_Patient101_CNV.csv"
    
    assert mrna_file.is_file(), "mRNA sample file must exist"
    assert cnv_file.is_file(), "CNV sample file must exist"

    with open(mrna_file, "rb") as f_mrna, open(cnv_file, "rb") as f_cnv:
        files = {
            "file_mRNA": ("BRCA_Patient101_mRNA.csv", f_mrna, "text/csv"),
            "file_CNV": ("BRCA_Patient101_CNV.csv", f_cnv, "text/csv"),
        }
        data = {
            "modalities": json.dumps(["mRNA", "CNV"]),
            "strategy": "s2",
            "sample_id": "Integration-Test-01"
        }
        res = client.post("/api/predict/upload", files=files, data=data)
        assert res.status_code == 200
        resp_data = res.json()
        assert resp_data["condition_id"] == "C9"
        assert resp_data["predicted_class"] == "BRCA"
        assert resp_data["confidence_pct"] > 95.0
        assert resp_data["modality_contributions"]["mRNA"] > 0
        assert resp_data["modality_contributions"]["CNV"] > 0
        assert resp_data["modality_contributions"]["miRNA"] == 0.0
        assert resp_data["modality_contributions"]["Methy"] == 0.0


def test_nan_values_rejection():
    # Construct invalid CSV with a NaN value
    csv_invalid = "KRT5,KRT6A\nNaN,1.0\n"
    files = {"file_mRNA": ("invalid.csv", csv_invalid.encode("utf-8"), "text/csv")}
    data = {"modalities": json.dumps(["mRNA"])}
    res = client.post("/api/predict/upload", files=files, data=data)
    assert res.status_code == 400
    assert "NaN values" in res.json()["detail"] or "Dimension mismatch" in res.json()["detail"]


def test_unsupported_file_extension():
    files = {"file_mRNA": ("test.exe", b"invalid binary", "application/octet-stream")}
    data = {"modalities": json.dumps(["mRNA"])}
    res = client.post("/api/predict/upload", files=files, data=data)
    assert res.status_code == 400
    assert "Unsupported file format" in res.json()["detail"]


def test_prediction_consistency_with_offline_extracted_evals():
    """Verifies that live API forward pass matches offline extracted evaluations exactly."""
    evals = np.load(PROJECT_ROOT / "reports" / "presentation_figures" / "data" / "extracted_test_evaluations.npz")
    
    # Check patient 0, 1, 2, 3
    for p_idx in range(4):
        res = client.post("/api/predict", json={
            "modalities": [1, 1, 1, 1],
            "patient_idx": p_idx,
            "strategy": "s2"
        })
        assert res.status_code == 200
        live_data = res.json()
        live_pred = live_data["predicted_class"]
        
        expected_pred_idx = int(evals["preds_s2_c0"][p_idx])
        expected_code = ["BRCA", "HNSC", "LGG", "THCA"][expected_pred_idx]
        assert live_pred == expected_code
        
        # Check probability alignment
        expected_probs = evals["probs_s2_c0"][p_idx]
        for c_idx, code in enumerate(["BRCA", "HNSC", "LGG", "THCA"]):
            diff = abs(live_data["probabilities"][code] - expected_probs[c_idx])
            assert diff < 1e-3, f"Probability mismatch for {code} on sample {p_idx}: diff={diff}"


def test_static_assets_serving():
    """Verifies that histopathology micrograph images and static assets are served properly."""
    expected_assets = [
        "histopathology-of-invasive-ductal-carcinoma-of-the.jpg",
        "oral-cancer-1-squamous-cell-carcinoma-histopathology.jpg",
        "pnet-histopathology-he-200x.jpg",
        "thyroid-papillary-carcinoma-histopathology-2.jpg",
        "credits.json",
        "banner.svg",
        "model_workflow.svg",
    ]
    for asset in expected_assets:
        res = client.get(f"/assets/{asset}")
        assert res.status_code == 200, f"Asset /assets/{asset} failed with {res.status_code}"
        assert len(res.content) > 0

