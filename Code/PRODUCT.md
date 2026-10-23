# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Academic evaluators, defense committee members, biomedical researchers, and students reviewing the Bachelor of Computer Applications (BCA) Major Project. They need to quickly understand the scientific validity, architectural novelty, and empirical resilience of the multimodal deep learning framework under incomplete diagnostic records.

## Product Purpose

Demonstrates a resilient pan-cancer classification framework capable of delivering accurate cancer subtype predictions across 4 target malignancies (BRCA, HNSC, LGG, THCA) even when diagnostic omic modalities (such as transcriptomic mRNA gene expression, microRNA regulation, somatic copy number alterations, or epigenetic DNA methylation) are partially missing. Success means clearly proving that the model degrades gracefully rather than failing catastrophically like standard early/late fusion architectures.

## Positioning

Unlike conventional multimodal cancer diagnosis models that crash or experience catastrophic performance collapse when an input stream is unobserved, this framework incorporates dynamic masked attention tokens, learned default missingness representations, and cross-modal attention to achieve 99.56% peak Macro-F1 under complete data and retain an +8.52% empirical gain under extreme single-modality stress tests (C12: CNV Only).

## Operating Context

- Academic evaluation & viva defense presentations
- Interactive web-based demonstration and topology inspection
- High-contrast slide presentation environments and desktop/laptop screens

## Capabilities and Constraints

- **Confirmed Capabilities**:
  - Live interactive 4-omics neural routing simulator with toggleable modalities (mRNA, miRNA, CNV, DNA Methylation).
  - FastAPI inference backend (`api_server.py`) serving live PyTorch model evaluations, latency telemetry, and biomarker attributions.
  - Interactive TCGA test patient case selector (BRCA, HNSC, LGG, THCA) demonstrating patient-specific multi-omic predictions.
  - Diagnostic prediction card with real-time confidence meter, subtype classification, and 4-modality contribution breakdown.
  - Empirical ablation benchmark tables comparing Baseline S1 (Complete Only) against Proposed S2 (Modality Dropout) across 15 combinatorial missingness conditions (C0–C14).
  - Hybrid offline fallback: seamless local fallback if the API server is offline, auto-upgrading to `🟢 Live Model Online` upon connection.
  - Integrated dark/light mode toggle with default light mode.
- **Constraints**:
  - Pure Vanilla CSS and lightweight native JavaScript without external framework dependencies (React/Tailwind not used).
  - Web client operates standalone in modern browsers with responsive layout support.

## Brand Commitments

- **Project Name**: Robust Multimodal AI for Multi-Cancer Classification
- **Institutional Identity**: Faculty of Computer Technology, Assam down town University, Guwahati
- **Contributors**: Aniket Saha (Data Engineering & Deep Learning Architecture), Udit Bora (Lead Developer & Interface Engineering), Kritartha Barman (Evaluation Protocols & Leak-Free Pipeline), Mr. Wanraplang Nongbri (Academic Supervisor)
- **Tone**: Rigorous, scientific, modern, research-grade, and precise.

## Evidence on Hand

- Verified multi-omic TCGA pan-cancer cohort benchmarks (2,263 patient samples, 4 cancer classes, 9,844 molecular features).
- Empirical ablation accuracy metrics across all 15 combinatorial conditions: C0 Complete (99.56% F1), C1 Missing mRNA (99.45%), C11 Methy Only (93.37%), and C12 CNV Only (74.12%, +8.52% gain over S1).
- Working client and server codebase: [index.html](file:///d:/PLACE/BCA/MajorProject/Onic/Code/index.html), [style.css](file:///d:/PLACE/BCA/MajorProject/Onic/Code/style.css), [script.js](file:///d:/PLACE/BCA/MajorProject/Onic/Code/script.js), [api_server.py](file:///d:/PLACE/BCA/MajorProject/Onic/Code/api_server.py).

## Product Principles

1. **Truth Over Decoration**: All metrics, modality dropout behaviors, and diagnostic states must reflect genuine computational oncology realities.
2. **Resilience First**: Emphasize graceful degradation under data scarcity at every level of the UX.
3. **Clarity for Evaluation**: Structure information so academic evaluators can immediately locate architecture details, empirical validation data, and project contributions.
4. **Fluid Presentation**: Keep the interface responsive, tactile, and smooth for seamless demonstration flow during viva or review.

## Accessibility & Inclusion

- Keyboard navigability across all interactive elements, presets, chips, and sections (`Tab`, `Space`, `Enter`, `Arrow` keys).
- Semantic ARIA labels on theme toggles, modality buttons, and navigation controls.
- Full support for `prefers-reduced-motion` to disable animations for users who prefer static interactions.
