# Workflow Documentation

**Project name:** globias_3d_nuclei_soma  
**Date:** 2026-11-12  
**Workflow type:** New reproducible 3D fluorescence-image analysis workflow

---

## 1. Scientific Goal

Analyze 20 registered pairs of 3D fluorescence TIFF stacks to:

1. Segment and count DAPI-positive nuclei in each field.
2. Segment and count objects positive for the user-described somatic marker.
3. Calculate field-level nuclei-to-soma count ratios.
4. Test whether nuclei and soma-marker counts are associated across fields.
5. Characterize soma-marker-positive object sizes when segmentation quality permits.

Because the exact somatic marker identity was not available, these detections are described conservatively as **soma-marker-positive** or **soma-like 3D objects**, not confirmed anatomical whole cells.

## 2. Software Components & Versions

| Component | Version | Role |
|---|---:|---|
| Fiji/ImageJ | 2.16.0 / ImageJ 1.54p | Human inspection of raw, corrected, segmented, and figure outputs |
| Python, main analysis environment | 3.13.14 | Input verification, statistics, plotting, and reporting |
| Python, `napari-mcp` environment | 3.11.15 | micro_sam segmentation and related image processing |
| micro_sam | 1.8.2 | True-3D automatic instance segmentation |
| scikit-image | 0.26.0 in `napari-mcp` | Image processing, labels, measurements, and QC generation |
| NumPy | 2.5.1 in project environment log | Numerical processing |
| pandas | 3.0.5 | Tabular aggregation and analysis |
| SciPy | 1.18.0 | Statistical analysis |
| Matplotlib | 3.11.1 | Publication figures |
| seaborn | 0.13.2 | Publication-figure styling |
| NVIDIA CUDA-capable GPU | Available; CUDA used during segmentation | micro_sam inference acceleration |
| Operating system | Linux 6.8.0-111-generic x86_64, glibc 2.41 | Runtime platform |

The version record is in `logs/environment_metadata.log`. micro_sam and scikit-image versions were independently confirmed from the installed `napari-mcp` environment.

## 3. Processing Sequence

1. **Workspace and provenance setup** — Created structured folders for scripts, data, processed images, figures, logs, and raw-image references.
2. **Input integrity check** — Matched 20 DAPI stacks to 20 stain stacks by field ID; verified readability, one-to-one pairing, and equal `84 × 1000 × 1000` ZYX dimensions.
3. **Representative input review** — Assessed contrast, background, artifacts, focus, and target visibility. The review indicated uneven diffuse background, variable signal, and punctate artifacts, supporting a conservative pilot-first workflow.
4. **DAPI preprocessing** — Applied XY low-frequency haze correction (`sigma_xy=40 px`, subtraction strength `alpha=0.90`) followed by robust whole-volume normalization. No slice-wise equalization was applied.
5. **Soma-stain preprocessing** — Applied the same haze correction, mild XY-only Gaussian smoothing (`sigma_zyx=(0, 0.6, 0.6)`), and robust whole-volume normalization. No slice-wise equalization was applied.
6. **Human preprocessing review** — The user inspected raw and corrected F0003 stacks in Fiji and approved both channel-specific corrections.
7. **Pilot nuclei segmentation** — Ran true-3D micro_sam automatic instance segmentation with stock light-microscopy model `vit_b_lm` on CUDA. Retained nuclei of at least 25 voxels.
8. **Human nuclei review** — The user inspected raw data, 3D labels, representative-plane overlays, and an XY projection and accepted the pilot method despite known occasional misses/merges.
9. **Pilot soma segmentation** — Ran true-3D micro_sam AIS with `vit_b_lm` on CUDA. An initial minimum-volume filter retained objects of at least 300 voxels.
10. **Soma-filter revision** — The user rejected oversized false regions on low-signal slices 56 and 77. A robust IQR-derived upper cutoff of 18,310 voxels removed 16 oversized components. The final approved soma range was 300–18,310 voxels.
11. **Human pilot approval** — The user judged the revised soma result acceptable and approved application of the fixed pilot settings to all 20 fields.
12. **Batch segmentation and measurement** — Applied the approved channel-specific preprocessing and 3D segmentation settings to all field pairs. Saved preprocessed stacks, uint32 labels, object measurements, metadata, and QC images per field.
13. **Border handling** — Retained and flagged border-touching objects for total counts. Used border-excluded objects for the primary soma-size distribution and nonborder sensitivity summaries.
14. **Field-level aggregation** — Calculated nuclei counts, soma counts, safe nuclei-to-soma ratios, and object-volume summaries by field.
15. **Statistics** — Treated the 20 fields, rather than pooled individual objects, as replicates. Calculated descriptive summaries, bootstrap confidence intervals, Pearson and Spearman count associations, missingness checks, and IQR outlier reports.
16. **Plotting** — Generated four 300-DPI PNG figures and editable SVG counterparts. Figure 3 was revised to move the statistics box outside the data panel so that field F0072 remained visible.
17. **Visual result review** — A representative low-/typical-/high-count batch audit found no systematic batch-level segmentation failure. All final PNG figures passed visual review.

## 4. Key Settings & Parameters

| Parameter | Value | Script | Rationale |
|---|---|---|---|
| Representative pilot field | F0003 | Phase 4a–4e scripts | Review preprocessing and labels before committing to a long 3D batch |
| DAPI haze-correction scale | `sigma_xy=40 px` | `phase4a_preprocess_dapi_F0003_for_micro_sam.py`; batch script | Remove broad XY background while preserving nuclear-scale structures |
| DAPI background subtraction | `alpha=0.90` | Same as above | Conservative correction chosen after visual inspection |
| Soma haze correction | `sigma_xy=40 px`, `alpha=0.90` | `phase4b_preprocess_soma_stain_F0003_for_micro_sam.py`; batch script | Reduce uneven diffuse background |
| Soma smoothing | Gaussian `sigma_zyx=(0, 0.6, 0.6)` | Same as above | Mildly suppress punctate XY noise without smoothing across Z |
| Slice equalization | None | Preprocessing and batch scripts | Avoid introducing artificial slice-to-slice intensity changes |
| Normalization | Robust whole-volume normalization to float range | Preprocessing and batch scripts | Normalize each stain independently while preserving within-volume Z variation |
| Segmentation method | True-3D micro_sam AIS | Phase 4c–4f scripts | Handle volumetric objects and avoid independent 2D slice counting |
| micro_sam model | Stock `vit_b_lm` | Phase 4c–4f scripts | Light-microscopy model selected for both fluorescence channels |
| Inference device | CUDA GPU | Phase 4c–4f scripts | Accelerate large-volume 3D segmentation |
| Nuclei minimum size | 25 voxels | Phase 4c and batch script | Remove very small components while retaining dim/small nuclei candidates |
| Soma accepted size | 300–18,310 voxels, inclusive | Phase 4d–4f scripts | Lower limit removed small puncta; IQR-derived upper limit removed user-identified oversized false regions |
| Count border policy | Retain and flag | Batch/statistics scripts | Avoid silently discarding objects from per-field counts |
| Primary soma-size subset | 4,519 nonborder objects | Statistics/plotting scripts | Reduce truncation bias in size distributions |
| Primary size unit | Voxels | All measurement outputs | Reliable Z spacing was not available |
| Provisional XY scale | 0.324512482 µm/pixel | Statistics metadata/outputs | Numeric TIFF XY scale provisionally interpreted as physical calibration |
| Provisional Z spacing | 1.0 µm | Statistics metadata/outputs | Explicit assumption only; not supported by reliable acquisition metadata |
| Significance threshold | `alpha=0.05` | Statistics script | Prespecified interpretation threshold |
| Confidence intervals | Bootstrap 95% CI for medians; 95% CI for Pearson correlation | Statistics script | Quantify field-level uncertainty |

## 5. Input Data

- **Image type:** Registered 3D fluorescence tissue fields.
- **Channels:** DAPI nuclei channel and separate somatic-marker channel.
- **Storage:** One single-channel TIFF stack per marker and field.
- **Format:** 16-bit TIFF, `84 × 1000 × 1000` ZYX per stack.
- **Dataset size:** 20 registered field pairs, 40 TIFF stacks total.
- **Field IDs:** F0003, F0006, F0009, F0011, F0012, F0021, F0027, F0041, F0050, F0070, F0072, F0073, F0077, F0078, F0079, F0080, F0084, F0093, F0094, F0096.
- **Locations:**
  - `/data/tasks/Globias_workshop/DAPI/`
  - `/data/tasks/Globias_workshop/stain/`
- **Input manifest:** `data/input_integrity_manifest.csv`

## 6. Output Data

### Batch segmentation and measurements

- Combined nuclei objects: `processed_images/batch/batch_nuclei_objects_combined.csv`
- Combined soma objects: `processed_images/batch/batch_soma_objects_combined.csv`
- Per-field counts/status: `processed_images/batch/batch_per_field_counts_status.csv`
- Failure table: `processed_images/batch/batch_failures.csv`
- Batch summary: `processed_images/batch/batch_summary.json` and `.txt`
- Per-field images and tables: `processed_images/batch/<field_id>/`

### Statistical outputs

- Main statistics table: `data/Statistics_Results.csv`
- Field counts and ratios: `data/field_counts_with_safe_ratios.csv`
- Missingness report: `data/missingness_report.csv`
- IQR outlier report: `data/iqr_outlier_report.csv`
- Soma object-level volume summary: `data/soma_voxel_volume_object_level_summary.csv`
- Field-level soma median/IQR table: `data/soma_field_median_iqr_by_field.csv`

### Figures

Each figure is supplied as a 300-DPI PNG and editable SVG:

1. `figures/figure1_per_field_counts_panels.*`
2. `figures/figure2_per_field_ratios_with_bootstrap_ci.*`
3. `figures/figure3_counts_scatter_association.*`
4. `figures/figure4_soma_voxel_distribution_and_field_iqr.*`

### Main quantitative results

- **Completed fields:** 20/20; no partial or failed fields.
- **Retained nuclei:** 96,590 total.
- **Retained soma-marker objects:** 5,183 total.
- **Nonborder soma objects in the primary size distribution:** 4,519.
- **Nuclei per field:** median 5,126; bootstrap 95% CI 4,553–5,337; mean 4,829.5; SD 946.70; range 2,511–5,973.
- **Soma-marker objects per field:** median 238.5; bootstrap 95% CI 180.5–270.5; mean 259.15; SD 131.92; range 90–563.
- **Count association, total counts:** Pearson `r=0.318520`, 95% CI `[-0.144347, 0.667023]`, `p=0.17109`; Spearman `rho=0.401806`, `p=0.0790697`.
- **Nonborder sensitivity:** Pearson `r=0.327036`, `p=0.1593`; Spearman `rho=0.352764`, `p=0.127119`.
- No association test reached `alpha=0.05`. This is reported as **no statistically significant field-level association**, not evidence that no biological relationship exists.

## 7. Statistical Analysis

- **Replicate unit:** Field (`n=20`), not pooled segmented objects.
- **Descriptive statistics:** Mean, standard deviation, median, range, and bootstrap 95% confidence intervals where applicable.
- **Ratios:** Nuclei-to-soma count ratios calculated safely per field; total and nonborder summaries reported separately.
- **Association tests:**
  - Pearson correlation for linear association between field-level counts.
  - Spearman rank correlation as a rank-based sensitivity analysis.
- **Significance threshold:** `p < 0.05`.
- **Outlier handling:** IQR reports were generated. The soma pilot's upper size cutoff was derived from a robust IQR upper fence and then fixed at 18,310 voxels before batch processing. Statistical field observations were not removed merely because they were extreme.
- **Border sensitivity:** Primary total-count analyses were accompanied by nonborder count sensitivity analyses.
- **Soma size:** Primary distribution uses nonborder objects and voxel volume on a logarithmic display scale.

## 8. Manual Steps and User Verification

Human review was an explicit part of the analysis rather than relying solely on automated visual assessment:

1. The user inspected raw and corrected F0003 DAPI data in Fiji and approved preprocessing.
2. The user inspected raw and corrected F0003 soma-stain data and approved preprocessing.
3. The user reviewed F0003 nuclei labels and overlays and accepted the segmentation method.
4. The user rejected the first soma result because slices 56 and 77 contained oversized false regions.
5. The workflow applied an upper size filter, regenerated masks and before/after QC images, and presented the revision.
6. The user judged the revised soma result acceptable and approved fixed settings for the 20-field batch, statistics, plotting, documentation, and QA.

Automated visual review was advisory and complementary. It did not override measured metadata or the user's direct inspection.

## 9. Rationale

- **Pilot before batch:** Each 3D stack is large, and the input review identified uneven background and heterogeneous signal. A reviewed pilot reduced the risk of applying unsuitable settings to all fields.
- **Channel-specific preprocessing:** DAPI nuclei and the somatic marker have different morphology and noise. The soma channel therefore received mild XY denoising while DAPI did not.
- **No Z smoothing or slice equalization:** These choices protected real Z-dependent signal variation and avoided imposing continuity unsupported by the uncertain Z calibration.
- **True-3D segmentation:** Objects span multiple slices; 3D AIS avoids counting the same structure independently on adjacent planes.
- **Fixed size filtering after approval:** The lower limits rejected debris; the soma upper limit directly addressed oversized artifacts observed by the user. The approved limits were then held constant for reproducibility.
- **Border-aware reporting:** Total counts retain edge objects, while nonborder size distributions reduce bias from truncated components.
- **Field-level statistics:** Fields are the appropriate replicate unit for the requested count association; treating thousands of segmented objects as independent replicates would inflate apparent statistical power.
- **Voxel volume as primary:** Reliable Z spacing was unavailable, making voxel counts more defensible than physical volumes.

## 10. Quality Control, Limitations, and Known Issues

### Quantitative and operational checks

- All 20/20 DAPI/stain pairs were readable, uniquely paired, and shape-compatible.
- Batch processing completed for all fields with no partial or failed fields.
- uint32 instance labels, per-object tables, field summaries, and QC outputs were produced.
- Representative low-, typical-, and high-soma-count fields showed no systematic batch-level segmentation failure.
- All final PNG figures passed visual review for legibility and unobscured data.

### Limitations

1. **Unknown marker identity:** The stain cannot be interpreted as a specific cell type or confirmed whole-cell anatomy. Results concern operationally defined soma-marker-positive objects.
2. **Uncertain physical calibration:** XY scale was inferred from a numeric TIFF value and Z spacing was assumed. Voxel volume is primary; any µm³ value is explicitly provisional.
3. **Nuclei errors remain possible:** Pilot QC suggested occasional missed or merged nuclei, particularly on weak edge planes. The user accepted this limitation.
4. **Soma definition is operational:** Soma labels are the approved micro_sam output constrained to 300–18,310 voxels. They may include touching/merged detections or omit faint peripheral signal.
5. **Visual QC is sampled:** Representative compilations cannot prove every object in every plane is correct.
6. **No manual ground truth:** Segmentation accuracy metrics such as precision, recall, Dice score, or panoptic quality could not be calculated.
7. **Border objects:** Border-touching objects are valid for operational total counts but may be truncated; they are excluded from the primary size distribution.
8. **Statistical power:** With 20 fields, the correlation confidence interval is broad. The nonsignificant result is not proof of no relationship.

### Resolved execution issues

- The first nuclei pilot stopped because `numpy as np` was not explicitly imported in the segmentation environment. Explicit imports fixed the issue; the successful rerun generated the approved outputs.
- Plotting initially failed due to incorrect ratio-row lookup. Exact `Statistics_Results.csv` identifiers were used in the corrected script. No underlying measurements or statistics changed.
- Figure 3 initially placed a statistics box over field F0072. The box was moved to an external right margin; all points and field labels are visible in the final figure.

## 11. Reproducibility

- **Project root:** `/app/data/projects/globias_3d_nuclei_soma/`
- **Code location:** `scripts/python/`
- **Input verification script:** `scripts/python/phase3_verify_input_integrity_io.py`
- **Pilot preprocessing scripts:** `phase4a_preprocess_dapi_F0003_for_micro_sam.py`, `phase4b_preprocess_soma_stain_F0003_for_micro_sam.py`
- **Pilot segmentation scripts:** `phase4c_pilot_segment_F0003_dapi_nuclei_micro_sam_3d.py`, `phase4d_pilot_segment_F0003_soma_micro_sam_3d.py`
- **Soma revision script:** `phase4e_revise_F0003_soma_upper_size_filter_and_qc.py`
- **Batch script:** `phase4f_batch_segment_20_fields_micro_sam_3d_nuclei_soma.py`
- **Statistics script:** `phase4g_stage1_statistics_counts_ratios_soma_volume.py`
- **Plotting script:** `phase4h_stage2_publication_plots_counts_ratios_scatter_soma_volume.py`
- **Example/complete input data:** `/data/tasks/Globias_workshop/`
- **Environment record:** `logs/environment_metadata.log`
- **State and decision record:** Project state ledger in the project workspace.
- **Public access:** [TO BE FILLED — add DOI, repository, or archive URL if this project is shared publicly]
- **Container/install specification:** [TO BE FILLED — add a portable environment export or container definition before external reproduction]

For reproduction, retain the original field naming convention, preserve DAPI/stain registration, apply the fixed preprocessing and segmentation settings without field-specific tuning, and interpret physical volumes only after independently confirming both XY and Z calibration.