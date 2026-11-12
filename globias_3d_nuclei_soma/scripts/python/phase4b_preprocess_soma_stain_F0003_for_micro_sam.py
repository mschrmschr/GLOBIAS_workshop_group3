import json
import tifffile
from skimage import filters

# =========================
# CONFIG
# =========================
INPUT_TIF = r'/data/tasks/Globias_workshop/stain/F0003_stain.tif'
OUTPUT_DIR = r'/app/data/projects/globias_3d_nuclei_soma/processed_images/pilot/F0003'

OUTPUT_PREPROC_TIF = os.path.join(
    OUTPUT_DIR,
    'F0003_stain_preprocessed_for_micro_sam_3d_float32_0to1.tif'
)
OUTPUT_METADATA_JSON = os.path.join(
    OUTPUT_DIR,
    'F0003_stain_preprocessing_metadata.json'
)
OUTPUT_QC_PNG = os.path.join(
    OUTPUT_DIR,
    'F0003_stain_preprocessing_qc_raw_vs_corrected.png'
)

# Conservative, data-guided defaults
P_LOW_NORM = 0.5
P_HIGH_NORM = 99.8
BACKGROUND_SIGMA_Z = 1.0
BACKGROUND_SIGMA_XY_DIVISOR = 25.0
ALPHA_LOW_HAZE = 0.70
ALPHA_HIGH_HAZE = 0.90
HAZE_THRESHOLD = 0.15

# Requirement: assess drift but avoid slice-wise normalization/equalization
APPLY_SLICE_EQUALIZATION = False

# Optional mild denoise for punctate speckles only (smaller than soma boundaries)
USE_PUNCTA_DENOISE_IF_NEEDED = True
PUNCTA_TRIGGER_FRACTION = 0.020
MILD_DENOISE_SIGMA_ZYX = (0.0, 0.6, 0.6)

EPS = 1e-8


def percentiles_dict(arr, plist=(0, 0.5, 1, 5, 50, 95, 99, 99.8, 100)):
    vals = np.percentile(arr, plist)
    return {f'p{p}': float(v) for p, v in zip(plist, vals)}


os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# 1) Load raw ZYX stack (unchanged on disk)
# -------------------------
raw = tifffile.imread(INPUT_TIF)
print(f'Loaded raw stack: shape={raw.shape}, dtype={raw.dtype}', flush=True)

if raw.ndim != 3:
    raise ValueError(f"VERIFICATION FAILED: Expected 3D ZYX stack (ndim=3), got ndim={raw.ndim}.")

z, y, x = raw.shape
if min(z, y, x) <= 0:
    raise ValueError('VERIFICATION FAILED: Empty dimension detected in input stack.')

raw_f = raw.astype(np.float32, copy=False)
if not np.isfinite(raw_f).all():
    raise ValueError('VERIFICATION FAILED: Input contains non-finite values.')

raw_pct = percentiles_dict(raw_f)
raw_dynamic = max(raw_pct['p99.8'] - raw_pct['p0.5'], EPS)

# -------------------------
# 2) Assess per-slice intensity drift (no slice-wise correction applied)
# -------------------------
slice_p10 = np.percentile(raw_f, 10, axis=(1, 2))
slice_p50 = np.percentile(raw_f, 50, axis=(1, 2))
slice_p90 = np.percentile(raw_f, 90, axis=(1, 2))

z_idx = np.arange(z, dtype=np.float32)
slope_per_slice = float(np.polyfit(z_idx, slice_p50, 1)[0]) if z > 1 else 0.0
slice_median_span = float(np.percentile(slice_p50, 95) - np.percentile(slice_p50, 5))
drift_fraction = float(slice_median_span / raw_dynamic)

print(
    f'Per-slice drift assessment: slope_median_per_z={slope_per_slice:.6f}, '
    f'drift_fraction={drift_fraction:.4f}',
    flush=True
)

if APPLY_SLICE_EQUALIZATION:
    raise ValueError('VERIFICATION FAILED: This workflow is configured to avoid slice-wise normalization.')

work = raw_f.copy()
print('Slice-wise equalization intentionally disabled to preserve inter-slice intensity relationships.', flush=True)

# -------------------------
# 3) Quantify punctate noise and membrane-like edge continuity proxies
# -------------------------
hi_freq = work - filters.gaussian(work, sigma=(0.0, 1.0, 1.0), preserve_range=True)
puncta_amp = float(np.percentile(np.abs(hi_freq), 99.9))
puncta_fraction = float(puncta_amp / (raw_dynamic + EPS))

rep_idx = np.array([0, int(np.argmin(slice_p50)), z // 2, z - 1], dtype=int)
rep_idx = np.unique(np.clip(rep_idx, 0, z - 1))
edge_strength = []
edge_density = []
for zi in rep_idx:
    grad = filters.sobel(work[zi])
    thr = float(np.percentile(grad, 90))
    edge_strength.append(float(np.mean(grad)))
    edge_density.append(float(np.mean(grad > thr)))

membrane_edge_strength_mean = float(np.mean(edge_strength))
membrane_edge_density_mean = float(np.mean(edge_density))

print(
    f'Puncta/membrane assessment: puncta_fraction={puncta_fraction:.4f}, '
    f'edge_strength_mean={membrane_edge_strength_mean:.6f}, '
    f'edge_density_mean={membrane_edge_density_mean:.4f}',
    flush=True
)

# Optional mild denoise only for strong puncta, with sub-boundary scale
mild_denoise_used = bool(USE_PUNCTA_DENOISE_IF_NEEDED and puncta_fraction >= PUNCTA_TRIGGER_FRACTION)
if mild_denoise_used:
    work = filters.gaussian(work, sigma=MILD_DENOISE_SIGMA_ZYX, preserve_range=True).astype(np.float32)
    print(
        f'Applied mild puncta denoise with sigma_zyx={MILD_DENOISE_SIGMA_ZYX} '
        f'(trigger={PUNCTA_TRIGGER_FRACTION:.3f}).',
        flush=True
    )
else:
    print('No denoising applied (puncta below trigger or denoise disabled).', flush=True)

# -------------------------
# 4) Conservative 3D low-frequency haze correction
# -------------------------
bg_sigma_xy = max(16.0, min(y, x) / BACKGROUND_SIGMA_XY_DIVISOR)
background = filters.gaussian(
    work,
    sigma=(BACKGROUND_SIGMA_Z, bg_sigma_xy, bg_sigma_xy),
    preserve_range=True
).astype(np.float32)

bg_p10 = float(np.percentile(background, 10))
bg_p90 = float(np.percentile(background, 90))
spatial_haze_fraction = float((bg_p90 - bg_p10) / (raw_dynamic + EPS))
alpha = ALPHA_HIGH_HAZE if spatial_haze_fraction >= HAZE_THRESHOLD else ALPHA_LOW_HAZE

bg_floor = float(np.percentile(background, 5))
corrected = work - (alpha * (background - bg_floor))
corrected = np.clip(corrected, 0, None).astype(np.float32)

print(
    f'Background assessment: spatial_haze_fraction={spatial_haze_fraction:.4f}, '
    f'alpha={alpha:.2f}, bg_sigma_xy={bg_sigma_xy:.2f}',
    flush=True
)

# -------------------------
# 5) Robust global normalization (whole-volume only)
# -------------------------
low = float(np.percentile(corrected, P_LOW_NORM))
high = float(np.percentile(corrected, P_HIGH_NORM))
if high <= low:
    raise ValueError(f'VERIFICATION FAILED: Invalid normalization window high<=low ({high} <= {low}).')

norm = (corrected - low) / (high - low)
norm = np.clip(norm, 0.0, 1.0).astype(np.float32)

corr_pct = percentiles_dict(norm)

# -------------------------
# 6) Save outputs
# -------------------------
tifffile.imwrite(OUTPUT_PREPROC_TIF, norm, imagej=False)

low_signal_z = int(np.argmin(slice_p50))
mid_z = int(z // 2)
high_signal_z = int(np.argmax(slice_p50))

meta = {
    'input_tif': INPUT_TIF,
    'output_preprocessed_tif': OUTPUT_PREPROC_TIF,
    'output_qc_png': OUTPUT_QC_PNG,
    'shape_zyx': [int(z), int(y), int(x)],
    'raw_dtype': str(raw.dtype),
    'output_dtype': str(norm.dtype),
    'notes': {
        'biological_marker_identity': 'unknown somatic membrane stain (user-described)',
        'calibration_note': 'Physical Z spacing unknown; preprocessing preserves voxel geometry and does not apply calibration.'
    },
    'normalization': {
        'method': 'global_whole_volume_percentile_clip_then_minmax',
        'low_percentile': P_LOW_NORM,
        'high_percentile': P_HIGH_NORM,
        'low_value_before_scaling': low,
        'high_value_before_scaling': high
    },
    'slice_drift_assessment': {
        'slice_p10_first5': [float(v) for v in slice_p10[:5]],
        'slice_p50_first5': [float(v) for v in slice_p50[:5]],
        'slice_p90_first5': [float(v) for v in slice_p90[:5]],
        'median_slope_per_slice': slope_per_slice,
        'slice_median_span_p95_minus_p5': slice_median_span,
        'drift_fraction_vs_global_dynamic': drift_fraction,
        'slice_equalization_applied': False,
        'rationale': 'Assessed only; slice-wise normalization intentionally avoided per pipeline requirement.'
    },
    'puncta_and_membrane_assessment': {
        'puncta_fraction_from_99p9_high_frequency_residual': puncta_fraction,
        'mild_denoise_used': mild_denoise_used,
        'mild_denoise_sigma_zyx': list(MILD_DENOISE_SIGMA_ZYX),
        'mild_denoise_trigger_fraction': PUNCTA_TRIGGER_FRACTION,
        'membrane_edge_strength_mean_representative_slices': membrane_edge_strength_mean,
        'membrane_edge_density_mean_representative_slices': membrane_edge_density_mean,
        'representative_slices_used_for_edge_metrics': [int(v) for v in rep_idx]
    },
    'background_correction': {
        'method': '3d_gaussian_low_frequency_subtraction_above_floor',
        'sigma_zyx': [BACKGROUND_SIGMA_Z, float(bg_sigma_xy), float(bg_sigma_xy)],
        'spatial_haze_fraction': spatial_haze_fraction,
        'haze_threshold_for_alpha_switch': HAZE_THRESHOLD,
        'alpha_used': float(alpha),
        'bg_floor_p5': bg_floor
    },
    'qc_slices': {
        'low_signal_z': low_signal_z,
        'mid_z': mid_z,
        'high_signal_z': high_signal_z
    },
    'percentiles_raw': raw_pct,
    'percentiles_corrected_normalized': corr_pct
}

with open(OUTPUT_METADATA_JSON, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

# -------------------------
# 7) QC figure: raw vs corrected with low-signal plane included
# -------------------------
rep_slices = [low_signal_z, mid_z, high_signal_z]

fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
for c, zi in enumerate(rep_slices):
    axes[0, c].imshow(raw_f[zi], cmap='gray')
    axes[0, c].set_title(f'Raw z={zi}', fontsize=12)
    axes[0, c].axis('off')

    axes[1, c].imshow(norm[zi], cmap='gray', vmin=0, vmax=1)
    axes[1, c].set_title(f'Corrected+Norm z={zi}', fontsize=12)
    axes[1, c].axis('off')

fig.suptitle('F0003 soma-stain preprocessing QC (raw vs corrected, low-signal plane included)', fontsize=14)
fig.savefig(OUTPUT_QC_PNG, dpi=300)
plt.close(fig)

# -------------------------
# 8) Fail-loud verification
# -------------------------
if not os.path.isfile(OUTPUT_PREPROC_TIF) or os.path.getsize(OUTPUT_PREPROC_TIF) == 0:
    raise ValueError('VERIFICATION FAILED: Preprocessed TIFF was not written or is empty.')

if not os.path.isfile(OUTPUT_METADATA_JSON) or os.path.getsize(OUTPUT_METADATA_JSON) == 0:
    raise ValueError('VERIFICATION FAILED: Metadata JSON was not written or is empty.')

if not os.path.isfile(OUTPUT_QC_PNG) or os.path.getsize(OUTPUT_QC_PNG) == 0:
    raise ValueError('VERIFICATION FAILED: QC PNG was not written or is empty.')

if norm.shape != raw.shape:
    raise ValueError(f'VERIFICATION FAILED: Shape changed from {raw.shape} to {norm.shape}.')

if norm.dtype != np.float32:
    raise ValueError(f'VERIFICATION FAILED: Output dtype expected float32, got {norm.dtype}.')

if np.nanmin(norm) < -1e-6 or np.nanmax(norm) > 1.000001:
    raise ValueError('VERIFICATION FAILED: Normalized output is outside [0,1].')

print(
    f'SUCCESS: Wrote micro_sam-ready soma-stain stack with preserved ZYX shape {norm.shape}. '
    f'Output={OUTPUT_PREPROC_TIF}',
    flush=True
)
