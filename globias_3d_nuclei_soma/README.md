# 3D nuclei and soma-marker counting in tissue (GloBIAS 2026, group 3)

Analysis of 20 registered pairs of 3D fluorescence stacks (DAPI + a somatic membrane stain,
84 × 1000 × 1000 ZYX, 16-bit). Nuclei and soma-marker-positive objects were segmented in true 3D with
micro_sam (`vit_b_lm`, automatic instance segmentation, GPU), counted per field, compared, and the soma
object volumes summarised. The workflow was run by the Agentic-J agent, and the user reviewed every
pilot result in Fiji before the batch run.

This folder holds the code, tables, figures, reports and **downscaled** QC previews. The large image
files (preprocessed float32 stacks, uint32 3D label stacks and full-resolution QC images) are on
Zenodo: **[Zenodo DOI — add after upload]**. The raw input stacks are not redistributed.

## Main results (n = 20 fields)

| Quantity | Value |
|---|---|
| Nuclei, total | 96,590 · median 5,126 per field (bootstrap 95% CI 4,553–5,337) |
| Soma-marker objects, total | 5,183 · median 238.5 per field (180.5–270.5) |
| Nuclei / soma ratio | median 22.4 (18.1–28.0), range 9.0–35.7 |
| Count association | Pearson r = 0.32 (95% CI −0.14 to 0.67), p = 0.17 · Spearman ρ = 0.40, p = 0.079 |
| Soma volume (nonborder, n = 4,519) | median 2,616 voxels (≈ 275 µm³ **provisional**) |

No association reached α = 0.05. With 20 fields, that is not evidence of no relationship.

## Layout

| Path | Contents |
|---|---|
| `Workflow_Documentation.md` | Full method: versions, ordered steps, every parameter, rationale, limitations |
| `QA_Checklist_Report.md` | Agent's QA against workflow and image-publishing checklists |
| `state_ledger.json` | Agent's decision record: goal, metadata, decisions, completed steps, vision-judge verdicts |
| `scripts/python/` | Phase 3–4h scripts (IO check, preprocessing, pilot + batch segmentation, stats, plots); `archive/` = earlier versions |
| `scripts/imagej/` | Groovy script for Figure 5 |
| `data/` | `Statistics_Results.csv`, per-field counts and ratios, outlier and missingness reports, IO manifest |
| `figures/` | Figures 1–5 (PNG 300 dpi + SVG) |
| `processed_images/batch/` | Per-object CSVs, metadata JSON and QC previews per field, combined object tables, batch summary |
| `processed_images/pilot/F0003/` | Pilot outputs, incl. the soma size-filter revision (before/after QC) |
| `presentation_figures/` | Validation figures made for the workshop slides, plus the script that renders them from the TIFFs |
| `logs/` | Environment versions, IO summary, agent usage log |

QC PNGs in this repository were downscaled to at most 2000 px for size; the full-resolution versions are
inside the Zenodo zips. TIFF paths referenced in CSV/JSON files (`/app/data/projects/...`) refer to the
analysis container; they map to `processed_images/...` in the Zenodo archives.

## Key settings

- Preprocessing (both channels): subtract a Gaussian-blurred background (σ_xy = 40 px, α = 0.9),
  robust whole-volume normalisation to [0, 1]; no slice equalisation. Soma stain only: Gaussian
  σ_zyx = (0, 0.6, 0.6).
- Segmentation: micro_sam 1.8.2, `vit_b_lm`, AIS in 3D, CUDA.
- Size filters: nuclei ≥ 25 voxels; soma 300–18,310 voxels (upper bound = IQR fence on the F0003 pilot,
  added after the user flagged false oversized regions on slices 56 and 77).
- Border-touching objects are kept and flagged for counts and excluded from size distributions.
- Statistics: field is the replicate; bootstrap CIs (10,000 resamples); Pearson + Spearman.

## Missing information and limitations

- **Marker identity unknown**: objects are "soma-marker-positive objects", not confirmed cells.
- **Calibration unknown**: XY 0.3245 µm/px read from a TIFF tag whose unit is "pixel"; Z = 1.0 µm assumed.
  Voxels are the primary unit; any µm³ value is provisional.
- **No ground truth**: no precision/recall/Dice; quality rests on visual review by the user.
- No expected counts, acquisition metadata (objective, microscope) or biological grouping were available.
- Known QC-image issue: in the batch `*_QC_planes_overlay.png` / `*_QC_xy_mip_overlay.png`, the
  "boundary overlay" panels render as solid red; use the middle "mask" panel or `presentation_figures/`.
