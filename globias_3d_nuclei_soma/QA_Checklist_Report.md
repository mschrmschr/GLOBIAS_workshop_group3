# QA Checklist Report
**Project:** globias_3d_nuclei_soma  
**Date:** 2026-11-12  
**Workflow type:** New Workflow  
**Overall status:**
- Workflow: 5/6 Minimal passed | 2/4 Recommended passed
- Image Publishing: 9/13 Minimal passed | 0/1 Recommended passed

---

## PART C: SCIENTIFIC PLAUSIBILITY (Scored First)

**Deliverable measured:** `processed_images/batch/F*/*_objects.csv` (40 per-field object tables; nuclei + soma)  
**Measurement summary:** min=90, median=1537.0, max=5973, total=101,773, zero/empty files=0.

**PLAUSIBILITY VERDICT (from summarize_deliverables):**  
INCOMPLETE — the structural checks found nothing wrong, but no quantity was supplied so the RESULT ITSELF was never checked against anything. This is NOT a pass. Supply expected_per_file (a stated quantity from the user's request), then call this tool again. If the request genuinely states no quantity, say explicitly in the report that plausibility could not be verified.

| Item | Status | Evidence |
|------|--------|----------|
| Deliverable exists and is non-empty | ✅ PASS | 40 object CSVs matched; aggregate zero/empty files = 0. |
| Object counts are within an order of magnitude of what the user asked for | ⚠️ PARTIAL | User request states no expected per-field count; tool returned INCOMPLETE for quantity-based plausibility. |
| No unexplained collapse or explosion between passes | ✅ PASS | Logged revision changed pilot soma from 149 to 133 objects (~1.1x), not a >10x unexplained shift. |
| Counts are consistent with the images themselves | ⚠️ PARTIAL | Representative visual audits are acceptable, but no numeric target was supplied for objective plausibility anchoring. |

---

## PART A: WORKFLOW STANDARDS

### MINIMAL Requirements

| Item | Status | Evidence |
|------|--------|----------|
| Cite components and platform | ✅ PASS | Workflow_Documentation + `logs/environment_metadata.log` cite Fiji/ImageJ, Python envs, micro_sam, and core libraries with versions. |
| Describe sequence | ✅ PASS | Workflow_Documentation Section 3 provides ordered steps from IO validation through batch segmentation, stats, and plotting. |
| Key settings | ✅ PASS | Key thresholds/settings documented (e.g., nuclei min 25 voxels; soma range 300–18,310 voxels; preprocessing sigmas/alphas). |
| Example data and code | ✅ PASS | Example scripts are present in `scripts/python/` and `scripts/imagej/`; sample raw images present in `raw_images/`. |
| Manual ROI | ⚠️ PARTIAL | Human review checkpoints are documented, but no explicit manual ROI-selection procedure is described (appears not used). |
| Exact versions | ✅ PASS | Exact versions are recorded in documentation and logs (Fiji/ImageJ, Python, micro_sam, pandas/numpy/scipy/matplotlib/seaborn). |

### RECOMMENDED Requirements

| Item | Status | Evidence |
|------|--------|----------|
| All settings documented | ⚠️ PARTIAL | Core analysis settings are documented; not every auxiliary/display parameter is centralized in one parameter appendix. |
| Public example data and code | ❌ FAIL | Workflow_Documentation reproducibility section contains `[TO BE FILLED]` for public repository/DOI. |
| Rationale | ✅ PASS | Workflow_Documentation Section 9 explains why pilot-first, 3D AIS, fixed filters, and field-level statistics were used. |
| Limitations | ✅ PASS | Workflow_Documentation Section 10 details marker identity, calibration uncertainty, segmentation limits, and power constraints. |

### IDEAL Requirements

| Item | Status | Evidence |
|------|--------|----------|
| Screen recording or tutorial | ❌ FAIL | No tutorial/screencast file or link found in project files. |
| Easy install / container | ❌ FAIL | No Dockerfile/requirements/environment-export install instructions are present in the project root. |

---

## PART B: IMAGE PUBLISHING STANDARDS

### Image Format — MINIMAL

| Item | Status | Evidence |
|------|--------|----------|
| Focus on relevant content | ⚠️ PARTIAL | QC and representative panels are clear, but no explicit publication cropping/zoom policy is documented. |
| Separate individual images | ✅ PASS | Per-field outputs are saved separately under `processed_images/batch/<field_id>/`. |
| Show example image | ✅ PASS | Representative raw examples are present (`raw_images/F0003_DAPI.tif`, `raw_images/F0003_stain.tif`). |

### Image Colors & Channels — MINIMAL

| Item | Status | Evidence |
|------|--------|----------|
| Annotation of channels visible | ✅ PASS | Figure 5 labels channels as “DAPI (raw)” and “Soma marker (raw)” and caption documents channel provenance. |
| Report B&C adjustments | ✅ PASS | Caption documents display scaling (p1.0–p99.8) and scripts document normalization/background operations. |
| Grayscale for each channel | ✅ PASS | Channels are stored and processed as separate single-channel TIFF stacks (DAPI and stain). |
| Same adjustments for comparisons | ⚠️ PARTIAL | Same processing pipeline is fixed batch-wide, but figure display scaling is per-panel robust scaling rather than identical absolute limits. |
| Color-blind accessible | ✅ PASS | Plotting script sets `sns.set_palette('colorblind')`. |

### Image Colors & Channels — RECOMMENDED

| Item | Status | Evidence |
|------|--------|----------|
| Provide intensity scales (scale bars) | ⚠️ PARTIAL | Pixel scale bars are present in Figure 5 (“100 px”), but physical calibration is explicitly uncertain and not intensity-calibrated. |

### Image Annotation — MINIMAL

| Item | Status | Evidence |
|------|--------|----------|
| Add scale information | ⚠️ PARTIAL | Scale bars are added in Figure 5 via custom drawing, but only in pixel units with calibration caveat and not ImageJ `Scale Bar...` command. |
| Explain all annotations | ⚠️ PARTIAL | Figure 5 caption explains labels/bars; annotation semantics for all QC overlays are not fully centralized in one legend/spec. |
| Annotations legible | ✅ PASS | Figure script uses large annotation fonts (24–28 px) and visible bar thickness; publication plots use readable default fonts. |
| Annotations don't obscure data | ✅ PASS | Figure 3 annotation box was moved outside data panel in final plotting script to avoid obscuring points/labels. |

### Image Availability — MINIMAL

| Item | Status | Evidence |
|------|--------|----------|
| Images shared (lossless compression) | ✅ PASS | Analysis imagery is TIFF/PNG/SVG; no JPEG outputs were found in the project tree. |

---

## Action Items

### WORKFLOW - Critical Failures
List every ❌ FAIL from workflow MINIMAL requirements:
- [ ] None.

### WORKFLOW - Recommended Improvements
List every ❌ FAIL or ⚠️ PARTIAL from workflow RECOMMENDED requirements:
- [ ] All settings documented: add a single consolidated parameter appendix covering all script-level parameters (including display/QC defaults).
- [ ] Public example data and code: provide a public repository/archive URL or DOI in Workflow_Documentation.

### IMAGE PUBLISHING - Critical Failures
List every ❌ FAIL from image publishing MINIMAL requirements:
- [ ] None.

### IMAGE PUBLISHING - Recommended Improvements
List every ❌ FAIL or ⚠️ PARTIAL from image publishing RECOMMENDED requirements:
- [ ] Provide intensity scales (scale bars): add physically calibrated scale information (or explicitly calibrated intensity/scale references) beyond pixel-only bars where possible.
