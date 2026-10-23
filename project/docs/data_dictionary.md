# Data Dictionary

**Project:** A Robust Multimodal AI Framework for Multi-Cancer Classification Under Missing Modalities  
**Last Updated:** 2026-10-15

---

## 1. Source Dataset

### MLOmics Pan-cancer Classification Dataset

**Origin:** The Cancer Genome Atlas (TCGA) via the MLOmics benchmark collection  
**Archive:** `28729127.zip` (~8.83 GB)  
**Location:** `D:\Personal Files\BCA\5TH SEM\Major_Project_Research\Dataset_2\MLOmics\Main_Dataset\Classification_datasets\Pan-cancer\Original\`

---

## 2. Original Data Files

### 2.1 Modality Matrices

All four matrices share the same structure: **features (rows) × samples (columns)**.

| File | Modality | Description | Features | Samples | Feature Column |
|---|---|---|---:|---:|---|
| `Pan-cancer_mRNA.csv` | mRNA expression | Messenger RNA gene expression levels | 3,217 | 8,314 | `sample` |
| `Pan-cancer_miRNA.csv` | miRNA expression | MicroRNA expression levels | 383 | 8,314 | `sample` |
| `Pan-cancer_CNV.csv` | Copy Number Variation | DNA copy number alterations | 3,105 | 8,314 | `Sample` |
| `Pan-cancer_Methy.csv` | DNA Methylation | DNA methylation beta values | 3,139 | 8,314 | `sample` |

**Matrix orientation:**
- **Row 1 (header):** `[feature_col, sample_id_1, sample_id_2, ..., sample_id_8314]`
- **Rows 2+:** `[feature_name, value_1, value_2, ..., value_8314]`
- The first column contains the feature identifier (gene name, miRNA name, etc.)
- All remaining columns are patient sample measurements

> [!IMPORTANT]
> These matrices are transposed relative to the typical (samples × features) convention. Each column after the first represents one patient; each row represents one feature/gene.

### 2.2 Label File

| File | Description |
|---|---|
| `Pan-cancer_label_num.csv` | Single column `Label` with 8,314 integer values (0–31) |

**Structure:**
- Single column named `Label`
- No sample ID column — alignment is **positional**: row *i* of the label file corresponds to column *i+1* of each modality matrix (i.e., the *(i+1)*th column header is the sample ID)
- Values are integers 0 through 31, each uniquely mapping to one TCGA cancer project

---

## 3. Sample Identifiers

### 3.1 TCGA Barcode Format

All sample IDs follow the TCGA barcode convention:

```
TCGA-XX-XXXX-NN
 │     │    │   └─ Sample type (2 digits)
 │     │    └───── Patient case ID (4 alphanumeric)
 │     └────────── Tissue Source Site (2 chars)
 └──────────────── TCGA prefix
```

**Examples:**
- `TCGA-AO-A0JJ-01` → Patient `TCGA-AO-A0JJ`, sample type `01` (primary tumor)
- `TCGA-EE-A29D-06` → Patient `TCGA-EE-A29D`, sample type `06` (metastatic)

### 3.2 Sample Type Codes

| Code | Meaning | Count in Dataset |
|---|---|---:|
| `01` | Primary solid tumor | 7,800 |
| `06` | Metastatic | 351 |
| `03` | Primary blood-derived cancer | 163 |

### 3.3 Patient Uniqueness

Each of the 8,314 sample IDs maps to a unique patient ID. No patient has multiple samples in this dataset.

---

## 4. Cancer Type Labels

### 4.1 Full Label Mapping (32 Classes)

| Label | TCGA Project | Cancer Name | Samples |
|---:|---|---|---:|
| 0 | TCGA-ACC | Adrenocortical carcinoma | 76 |
| 1 | TCGA-BLCA | Bladder urothelial carcinoma | 401 |
| **2** | **TCGA-BRCA** | **Breast invasive carcinoma** | **757** |
| 3 | TCGA-CESC | Cervical squamous cell carcinoma | 291 |
| 4 | TCGA-CHOL | Cholangiocarcinoma | 36 |
| 5 | TCGA-COAD | Colon adenocarcinoma | 285 |
| 6 | TCGA-DLBC | Diffuse large B-cell lymphoma | 47 |
| 7 | TCGA-ESCA | Esophageal carcinoma | 180 |
| **8** | **TCGA-HNSC** | **Head and neck squamous cell carcinoma** | **506** |
| 9 | TCGA-KICH | Kidney chromophobe | 65 |
| 10 | TCGA-KIRC | Kidney renal clear cell carcinoma | 306 |
| 11 | TCGA-KIRP | Kidney renal papillary cell carcinoma | 268 |
| 12 | TCGA-LAML | Acute myeloid leukemia | 163 |
| **13** | **TCGA-LGG** | **Brain lower-grade glioma** | **506** |
| 14 | TCGA-LIHC | Liver hepatocellular carcinoma | 357 |
| 15 | TCGA-LUAD | Lung adenocarcinoma | 448 |
| 16 | TCGA-LUSC | Lung squamous cell carcinoma | 356 |
| 17 | TCGA-MESO | Mesothelioma | 87 |
| 18 | TCGA-OV | Ovarian serous cystadenocarcinoma | 9 |
| 19 | TCGA-PAAD | Pancreatic adenocarcinoma | 176 |
| 20 | TCGA-PCPG | Pheochromocytoma and paraganglioma | 161 |
| 21 | TCGA-PRAD | Prostate adenocarcinoma | 484 |
| 22 | TCGA-READ | Rectum adenocarcinoma | 91 |
| 23 | TCGA-SARC | Sarcoma | 250 |
| 24 | TCGA-SKCM | Skin cutaneous melanoma | 351 |
| 25 | TCGA-STAD | Stomach adenocarcinoma | 365 |
| 26 | TCGA-TGCT | Testicular germ cell tumors | 133 |
| **27** | **TCGA-THCA** | **Thyroid carcinoma** | **494** |
| 28 | TCGA-THYM | Thymoma | 119 |
| 29 | TCGA-UCEC | Uterine corpus endometrial carcinoma | 411 |
| 30 | TCGA-UCS | Uterine carcinosarcoma | 55 |
| 31 | TCGA-UVM | Uveal melanoma | 80 |

### 4.2 Target Four-Class Subset (Bold rows above)

| Label | Short Name | Full Name | Samples | % of Target |
|---:|---|---|---:|---:|
| 2 | BRCA | Breast invasive carcinoma | 757 | 33.5% |
| 8 | HNSC | Head and neck squamous cell carcinoma | 506 | 22.4% |
| 13 | LGG | Brain lower-grade glioma | 506 | 22.4% |
| 27 | THCA | Thyroid carcinoma | 494 | 21.8% |
| | | **Total** | **2,263** | **100%** |

**Class balance:** Moderate imbalance. BRCA has ~1.5× the samples of THCA. Not severe enough to mandate aggressive oversampling, but stratified splitting is essential.

---

## 5. Verification Files (Dataset_Report/)

### 5.1 `sample_label_pairs.csv`

| Column | Type | Description |
|---|---|---|
| `sample_id` | string | TCGA barcode (e.g., `TCGA-AO-A0JJ-01`) |
| `label` | integer | Numeric class label (0–31) |

**Rows:** 8,314  
**Source:** Extracted from mRNA headers (sample IDs) paired with label file (positional alignment)

### 5.2 `tcga_label_verification.csv`

| Column | Type | Description |
|---|---|---|
| `sample_id` | string | TCGA barcode |
| `label` | integer | Numeric class label |
| `patient_id` | string | TCGA patient ID (first 3 barcode parts) |
| `tcga_project` | string | TCGA project code from GDC API (e.g., `TCGA-BRCA`) |

**Rows:** 8,314  
**Source:** GDC API batch queries for independent verification  
**Missing project matches:** 0 (all 8,314 matched)

### 5.3 `tcga_label_crosstab.csv`

Cross-tabulation of `label` (rows) × `tcga_project` (columns). Perfect diagonal matrix — each label maps to exactly one project.

---

## 6. Filtered Dataset (Dataset_Report/filtered_4cancer/)

Pre-created by `create_4cancer_dataset.py`. Contains the 4-class subset.

| File | Description | Features | Samples |
|---|---|---:|---:|
| `labels_4cancer.csv` | Labels with sample IDs, project, cancer type/name | — | 2,263 |
| `mRNA_4cancer.csv` | Filtered mRNA matrix | 3,217 | 2,263 |
| `miRNA_4cancer.csv` | Filtered miRNA matrix | 383 | 2,263 |
| `CNV_4cancer.csv` | Filtered CNV matrix | 3,105 | 2,263 |
| `Methy_4cancer.csv` | Filtered methylation matrix | 3,139 | 2,263 |
| `dataset_audit.csv` | Audit of filtered creation | — | — |
| `dataset_metadata.json` | Machine-readable metadata | — | — |

### 6.1 `labels_4cancer.csv` Columns

| Column | Type | Description |
|---|---|---|
| `sample_id` | string | TCGA barcode |
| `label` | integer | Numeric label (2, 8, 13, or 27) |
| `tcga_project` | string | TCGA project code |
| `cancer_type` | string | Short name (BRCA, HNSC, LGG, THCA) |
| `cancer_name` | string | Full cancer name |

---

## 7. Feature Types by Modality

| Modality | Feature Identifiers | Measurement Type | Typical Scale | Notes |
|---|---|---|---|---|
| mRNA | Gene symbols | Normalized expression | Varies (likely FPKM/TPM or log-transformed) | 3,217 genes (subset of genome) |
| miRNA | miRNA identifiers | Normalized expression | Varies | 383 miRNAs |
| CNV | Gene/region identifiers | Copy number values | Centered around 0 (log2 ratio) or 2 (absolute copies) | 3,105 features |
| Methy | CpG site identifiers | Methylation beta values | 0.0–1.0 (proportion methylated) | 3,139 CpG sites |

> [!NOTE]
> Exact preprocessing applied by the MLOmics dataset authors is not fully documented. Scale and normalization must be empirically verified during Stage C preprocessing.

---

## 8. Key Constraints and Assumptions

1. **No natural missing modalities:** All 8,314 samples have data in all four modality matrices. Missing modalities will be **simulated** for robustness experiments.
2. **Positional label alignment:** Labels are aligned by array position, not by explicit sample ID joins. This has been verified via independent GDC API cross-referencing.
3. **No multi-sample patients:** Each sample represents a unique patient. Patient-level splitting reduces to sample-level splitting.
4. **Feature-column naming:** CNV uses `Sample` (capital S); others use `sample` (lowercase). Handle case-insensitively.
5. **Original data immutability:** The files in `Original/` must never be modified.
