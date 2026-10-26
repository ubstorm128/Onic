"""
Multimodal AI Framework Inference API Server
============================================
Exposes REST endpoints for real-time cancer subtype inference,
file upload of high-dimensional multi-omics data (mRNA, miRNA, CNV, Methy),
dynamic modality dropout inference under missing modalities,
and TCGA Pan-Cancer patient evaluation.

Run via:
    python api_server.py
or:
    python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
"""

import io
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Sequence

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent / "project"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_service import (
    engine,
    MODALITY_KEYS,
    MODALITY_INFO,
    CLASS_MAPPING,
    EVAL_CONDITIONS,
)

app = FastAPI(
    title="Robust Multimodal AI for Cancer Classification API",
    description=(
        "Live REST API serving the trained Cross-Attention Multimodal Cancer Classification "
        "model under arbitrary missing modalities."
    ),
    version="2.1.0",
)

# Enable CORS for local development and web origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    modalities: Optional[List[Union[int, str]]] = Field(
        default=[1, 1, 1, 1],
        description="Binary flags [mRNA, miRNA, CNV, Methy] (1=active, 0=absent) or list of modality names."
    )
    inputs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dictionary mapping modality name to numerical feature list or {feature: value} mapping."
    )
    patient_idx: Optional[int] = Field(
        default=None,
        ge=0,
        le=339,
        description="Optional test cohort patient index (0-339) to evaluate with genuine held-out multi-omic features."
    )
    strategy: str = Field(
        default="s2",
        description="Training strategy: 's2' (Modality Dropout proposed) or 's1' (Complete only baseline)."
    )
    is_preprocessed: bool = Field(
        default=False,
        description="True if inputs are already z-score standardized / min-max scaled; False if raw."
    )


class PredictResponse(BaseModel):
    request_status: str
    condition_id: str
    condition_description: str
    active_modalities: List[str]
    missing_modalities: List[str]
    predicted_class: str
    class_name: str
    organ: str
    confidence_pct: float
    probabilities: Dict[str, float]
    status_text: str
    status_class: str
    modality_contributions: Dict[str, float]
    biomarkers: List[Dict[str, Any]]
    attention_weights: List[List[float]]
    model_checkpoint: str
    strategy: str
    patient_id: str
    ground_truth: str
    latency_ms: float
    validation_warnings: List[str]


# ---------------------------------------------------------------------------
# Frontend Static File Serving
# ---------------------------------------------------------------------------
@app.get("/")
@app.get("/index.html")
def serve_index():
    return FileResponse(CODE_DIR / "index.html")


@app.get("/style.css")
def serve_css():
    return FileResponse(CODE_DIR / "style.css", media_type="text/css")


@app.get("/script.js")
def serve_js():
    return FileResponse(CODE_DIR / "script.js", media_type="application/javascript")


from fastapi.staticfiles import StaticFiles
if (CODE_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=CODE_DIR / "assets"), name="assets")
    app.mount("/Code/assets", StaticFiles(directory=CODE_DIR / "assets"), name="code_assets")
if (CODE_DIR / "sample_inputs").is_dir():
    app.mount("/sample_inputs", StaticFiles(directory=CODE_DIR / "sample_inputs"), name="sample_inputs")
    app.mount("/Code/sample_inputs", StaticFiles(directory=CODE_DIR / "sample_inputs"), name="code_sample_inputs")
if (CODE_DIR / "templates").is_dir():
    app.mount("/templates", StaticFiles(directory=CODE_DIR / "templates"), name="templates")
    app.mount("/Code/templates", StaticFiles(directory=CODE_DIR / "templates"), name="code_templates")


# ---------------------------------------------------------------------------
# API Documentation & Health Check
# ---------------------------------------------------------------------------
@app.get("/api")
def read_root():
    return {
        "status": "online",
        "service": "Robust Multimodal AI Pan-Cancer Classification",
        "version": "2.1.0",
        "model_architecture": "MultimodalCrossAttentionClassifier",
        "loaded_checkpoints": engine.loaded_checkpoints,
        "endpoints": {
            "health": "/api/health",
            "modalities": "/api/modalities",
            "predict_json": "/api/predict",
            "predict_upload": "/api/predict/upload",
            "patients": "/api/patients",
            "sample_demo": "/api/sample-demo/{sample_idx}",
            "template": "/api/template/{modality}",
            "explainability": "/api/explainability",
            "benchmarks": "/api/benchmarks",
        },
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_architecture": "MultimodalCrossAttentionClassifier",
        "framework": "Proposed Cross-Attention with Learned Mask Tokens (Strategy S2)",
        "cohort": "TCGA Pan-Cancer Multi-Omics",
        "classes": {
            k: f"{v['code']} ({v['name']})" for k, v in CLASS_MAPPING.items()
        },
        "modalities": MODALITY_INFO,
        "features_total": sum(MODALITY_INFO[m]["features"] for m in MODALITY_KEYS),
        "test_patients_count": len(engine.test_cohort.get("labels", [])) if engine.test_cohort else 340,
        "loaded_checkpoints": engine.loaded_checkpoints,
        "preprocessors_loaded": {m: (m in engine.preprocessors) for m in MODALITY_KEYS},
        "device": str(engine.device),
    }


@app.get("/api/modalities")
def list_modalities():
    return {
        "modalities": MODALITY_INFO,
        "feature_counts": {m: MODALITY_INFO[m]["features"] for m in MODALITY_KEYS},
    }


@app.get("/api/template/{modality}")
def download_template(modality: str):
    """Downloads a CSV template populated with the exact feature header row for the given modality."""
    mod_key = None
    for k in MODALITY_KEYS:
        if k.lower() == modality.lower():
            mod_key = k
            break

    if not mod_key:
        raise HTTPException(
            status_code=404,
            detail=f"Modality '{modality}' not found. Available: {MODALITY_KEYS}"
        )

    try:
        csv_content = engine.get_csv_template(mod_key)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=template_{mod_key}.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/patients")
def list_patients():
    return {
        "total_test_samples": len(engine.test_cohort.get("labels", [])) if engine.test_cohort else 340,
        "patients": engine.sample_patients_catalog,
    }


@app.get("/api/sample-demo/{sample_idx}")
def get_sample_demo(sample_idx: int):
    """Returns raw/unprocessed feature sample values for a demonstration test patient."""
    if not engine.test_cohort or "labels" not in engine.test_cohort:
        raise HTTPException(status_code=503, detail="Test cohort data not loaded.")

    p_idx = max(0, min(sample_idx, len(engine.test_cohort["labels"]) - 1))
    sample_data = {}

    for m in MODALITY_KEYS:
        arr = engine.test_cohort[m][p_idx : p_idx + 1]
        prep = engine.preprocessors.get(m)
        if prep and prep.scaler is not None:
            raw_vals = prep.scaler.inverse_transform(arr).squeeze(0).tolist()
        else:
            raw_vals = arr.squeeze(0).tolist()
        sample_data[m] = {
            "feature_count": len(raw_vals),
            "sample_head": raw_vals[:10],
            "first_features": prep.selected_feature_names[:10] if prep else [],
        }

    label = int(engine.test_cohort["labels"][p_idx])
    patient_id = f"TCGA-PATIENT-{p_idx + 1:03d}"
    for cat in engine.sample_patients_catalog:
        if cat["idx"] == p_idx:
            patient_id = cat["sample_id"]
            break

    return {
        "patient_index": p_idx,
        "patient_id": patient_id,
        "ground_truth": CLASS_MAPPING[label]["code"],
        "cancer_name": CLASS_MAPPING[label]["name"],
        "modalities": sample_data,
    }


@app.get("/api/explainability")
def get_explainability():
    if not engine.explainability_report:
        raise HTTPException(status_code=503, detail="Explainability report not available")
    return engine.explainability_report


@app.get("/api/benchmarks")
def get_benchmarks():
    comp_path = PROJECT_ROOT / "artifacts" / "backups" / "metrics_backup" / "missing_modality_comparison.json"
    if not comp_path.is_file():
        comp_path = PROJECT_ROOT / "artifacts" / "metrics" / "missing_modality_comparison.json"
    if comp_path.is_file():
        with open(comp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=503, detail="Benchmark comparison metrics not available")


# ---------------------------------------------------------------------------
# Prediction Inference Endpoints
# ---------------------------------------------------------------------------
def _resolve_modality_flags(modalities: Optional[Sequence[Union[int, str]]]) -> List[int]:
    """Helper to convert modality flags or list of strings to binary [0, 1] integers."""
    if not modalities:
        return [1, 1, 1, 1]

    if all(isinstance(x, (int, float)) for x in modalities):
        flags = [int(bool(x)) for x in modalities]
        if len(flags) != 4:
            raise HTTPException(status_code=400, detail="Modalities array must have length 4.")
        return flags

    # List of modality name strings: e.g. ["mRNA", "CNV"]
    str_set = {str(x).lower() for x in modalities}
    return [1 if m.lower() in str_set else 0 for m in MODALITY_KEYS]


@app.post("/api/predict", response_model=PredictResponse)
def predict_json(request: PredictRequest):
    """
    Submits numerical omics feature vectors or selects a test cohort patient case.
    Executes live PyTorch forward pass with the trained MultimodalCrossAttentionClassifier.
    """
    flags = _resolve_modality_flags(request.modalities)
    if sum(flags) == 0:
        raise HTTPException(status_code=400, detail="At least one omic modality must be active for inference.")

    strategy = request.strategy.lower()
    if strategy not in ["s1", "s2"]:
        strategy = "s2"

    try:
        # Case A: User supplied explicit input feature vectors or dictionaries
        if request.inputs:
            modality_arrays = {}
            warnings_list = []
            for i, mod in enumerate(MODALITY_KEYS):
                if flags[i] == 1:
                    if mod not in request.inputs:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Modality '{mod}' is marked active but missing in 'inputs' payload."
                        )
                    vec, warns = engine.parse_vector_input(
                        modality=mod,
                        values=request.inputs[mod],
                        is_preprocessed=request.is_preprocessed,
                    )
                    modality_arrays[mod] = vec
                    warnings_list.extend(warns)

            sample_meta = {
                "sample_id": f"Custom-Patient-{int(time.time()) % 10000}",
                "ground_truth": "Unlabeled User Sample",
                "warnings": warnings_list,
            }
            res = engine.run_inference(
                modality_arrays=modality_arrays,
                active_flags=flags,
                strategy=strategy,
                sample_meta=sample_meta,
            )
            return PredictResponse(**res)

        # Case B: User specified a test cohort patient index for verified evaluation
        elif request.patient_idx is not None:
            res = engine.predict_test_cohort_sample(
                sample_idx=request.patient_idx,
                active_flags=flags,
                strategy=strategy,
            )
            return PredictResponse(**res)

        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'inputs' (feature vectors) or 'patient_idx' (test cohort sample index) must be provided."
            )
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution error: {e}")


@app.post("/api/predict/upload", response_model=PredictResponse)
async def predict_upload(
    file_mRNA: Optional[UploadFile] = File(None),
    file_miRNA: Optional[UploadFile] = File(None),
    file_CNV: Optional[UploadFile] = File(None),
    file_Methy: Optional[UploadFile] = File(None),
    modalities: Optional[str] = Form(None),
    strategy: str = Form("s2"),
    sample_id: Optional[str] = Form(None),
):
    """
    Submits real CSV or TSV files for any combination of the 4 omics modalities.
    Validates file format, feature headers, dimensions, NaN/Inf, applies training scalers,
    and returns cancer classification predictions from the trained PyTorch model.
    """
    uploaded_files = {
        "mRNA": file_mRNA,
        "miRNA": file_miRNA,
        "CNV": file_CNV,
        "Methy": file_Methy,
    }

    # Determine which modalities were actually provided
    flags = [0, 0, 0, 0]
    if modalities:
        try:
            parsed_mods = json.loads(modalities)
            flags = _resolve_modality_flags(parsed_mods)
        except Exception:
            flags = _resolve_modality_flags(modalities.split(","))
    else:
        # Default flags based on non-null uploaded files
        for i, mod in enumerate(MODALITY_KEYS):
            if uploaded_files[mod] is not None:
                flags[i] = 1

    if sum(flags) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided. Please upload at least one CSV/TSV modality file (mRNA, miRNA, CNV, or Methy)."
        )

    # Process each uploaded file
    modality_arrays = {}
    validation_warnings = []

    for i, mod in enumerate(MODALITY_KEYS):
        if flags[i] == 1:
            up_file = uploaded_files[mod]
            if up_file is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Modality '{mod}' is selected as active, but no file was uploaded for it."
                )

            # Check filename extension
            fn = up_file.filename or f"{mod}.csv"
            ext = Path(fn).suffix.lower()
            if ext not in [".csv", ".tsv", ".txt"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format '{ext}' for {mod}. Only .csv, .tsv, and .txt files are supported."
                )

            # Read file bytes with 25MB limit
            content = await up_file.read()
            if len(content) > 25 * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded file '{fn}' exceeds 25 MB size limit."
                )

            try:
                processed_vec, warns = engine.parse_and_validate_file(
                    modality=mod,
                    file_bytes=content,
                    filename=fn,
                )
                modality_arrays[mod] = processed_vec
                validation_warnings.extend(warns)
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error validating {mod} in '{fn}': {e}")

    sample_label = sample_id if sample_id else "Uploaded-Patient-Sample"
    sample_meta = {
        "sample_id": sample_label,
        "ground_truth": "Unlabeled Clinical Upload",
        "warnings": validation_warnings,
    }

    try:
        strat = "s2" if strategy.lower() == "s2" else "s1"
        res = engine.run_inference(
            modality_arrays=modality_arrays,
            active_flags=flags,
            strategy=strat,
            sample_meta=sample_meta,
        )
        return PredictResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {e}")


# ---------------------------------------------------------------------------
# Server Launcher
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("  Multimodal AI Framework Inference API Server")
    print("  Loaded Model: Proposed Cross-Attention (Strategy S2)")
    print("  Target Cohort: TCGA Pan-Cancer (BRCA, HNSC, LGG, THCA)")
    print("  Serving at: http://127.0.0.1:8000")
    print("  Swagger UI: http://127.0.0.1:8000/docs")
    print("=" * 65 + "\n")
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(CODE_DIR))
