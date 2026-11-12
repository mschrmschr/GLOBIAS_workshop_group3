import json
import tifffile
from skimage import measure, segmentation, exposure

# =========================
# CONFIG
# =========================
LABELS_TIF = "/app/data/projects/globias_3d_nuclei_soma/processed_images/pilot/F0003/F0003_stain_micro_sam_3d_soma_labels_uint32.tif"
OBJECTS_CSV = "/app/data/projects/globias_3d_nuclei_soma/processed_images/pilot/F0003/F0003_stain_micro_sam_3d_soma_objects_pilot.csv"
RAW_TIF = "/data/tasks/Globias_workshop/stain/F0003_stain.tif"
CORRECTED_TIF = "/app/data/projects/globias_3d_nuclei_soma/processed_images/pilot/F0003/F0003_stain_preprocessed_for_micro_sam_3d_float32_0to1.tif"
OUT_DIR = "/app/data/projects/globias_3d_nuclei_soma/processed_images/pilot/F0003"

MIN_VOXELS = 300
TARGET_Z_PLANES = [56, 77]
QC_PLANES = [10, 26, 56, 77]  # includes one earlier representative plane and the requested planes
XY_PIXEL_SIZE_UM = 0.324512482
ASSUMED_Z_SPACING_UM = 1.0
VOLUME_NOTE = (
    "PROVISIONAL: XY=0.324512482 um and assumed Z=1.0 um; "
    "physical voxel calibration not available."
)

# =========================
# I/O paths
# =========================
revised_labels_tif = os.path.join(
    OUT_DIR, "F0003_stain_micro_sam_3d_soma_labels_uint32_revised_sizefilter.tif"
)
revised_objects_csv = os.path.join(
    OUT_DIR, "F0003_stain_micro_sam_3d_soma_objects_pilot_revised.csv"
)
removed_report_csv = os.path.join(
    OUT_DIR, "F0003_stain_micro_sam_3d_soma_removed_oversized_objects_report.csv"
)
metadata_json = os.path.join(
    OUT_DIR, "F0003_stain_micro_sam_3d_soma_upper_size_filter_metadata.json"
)
qc_planes_png = os.path.join(
    OUT_DIR, "F0003_stain_micro_sam_3d_soma_QC_planes_before_after_revised.png"
)
qc_focus_png = os.path.join(
    OUT_DIR, "F0003_stain_micro_sam_3d_soma_QC_z56_z77_before_after_revised.png"
)
qc_mip_png = os.path.join(
    OUT_DIR, "F0003_stain_micro_sam_3d_soma_QC_xy_mip_before_after_revised.png"
)


def _assert_file_nonempty(path):
    if (not os.path.isfile(path)) or os.path.getsize(path) == 0:
        raise ValueError(f"VERIFICATION FAILED: expected non-empty file not found: {path}")


def _derive_upper_cutoff(volumes):
    """
    Reproducible conservative upper-tail rule:
    1) Report Tukey-IQR upper outliers for transparency.
    2) Select cutoff from robust log-volume MAD outliers when present (z > 3.5).
       Cutoff is the largest non-outlier volume.
    3) Fallback to Tukey upper fence if no robust outliers exist.
    """
    vols = np.asarray(volumes, dtype=np.int64)
    if vols.size == 0:
        raise ValueError("VERIFICATION FAILED: cannot derive cutoff from empty volume array")

    q1, q3 = np.percentile(vols, [25, 75])
    iqr = q3 - q1
    upper_fence_iqr = q3 + 1.5 * iqr
    iqr_outlier_mask = vols > upper_fence_iqr
    iqr_outlier_count = int(np.sum(iqr_outlier_mask))
    print(
        f"IQR outlier check on voxel_volume: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
        f"upper_fence={upper_fence_iqr:.2f}, outliers_above_fence={iqr_outlier_count}",
        flush=True,
    )

    logv = np.log10(vols.astype(np.float64))
    med = np.median(logv)
    mad = np.median(np.abs(logv - med))

    robust_outlier_mask = np.zeros_like(vols, dtype=bool)
    if mad > 0:
        robust_z = 0.6745 * (logv - med) / mad
        robust_outlier_mask = robust_z > 3.5

    n_robust = int(np.sum(robust_outlier_mask))
    if n_robust > 0:
        max_non_outlier = int(np.max(vols[~robust_outlier_mask])) if np.any(~robust_outlier_mask) else MIN_VOXELS
        cutoff = max(max_non_outlier, MIN_VOXELS)
        rule = "log10-volume MAD robust outlier rule (z>3.5); cutoff=largest non-outlier"
    else:
        cutoff = max(int(np.floor(upper_fence_iqr)), MIN_VOXELS)
        rule = "fallback Tukey upper fence Q3+1.5*IQR"

    details = {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "iqr_upper_fence": float(upper_fence_iqr),
        "iqr_outlier_count": iqr_outlier_count,
        "log10_median": float(med),
        "log10_mad": float(mad),
        "robust_outlier_count": n_robust,
        "rule_used": rule,
    }
    return int(cutoff), details


def _make_overlay(base2d, labels2d):
    base = np.asarray(base2d, dtype=np.float32)
    p2, p98 = np.percentile(base, [2, 98])
    if p98 <= p2:
        p2, p98 = float(np.min(base)), float(np.max(base) + 1e-6)
    base_n = exposure.rescale_intensity(base, in_range=(p2, p98), out_range=(0, 1))
    rgb = np.dstack([base_n, base_n, base_n])
    boundaries = segmentation.find_boundaries(labels2d > 0, mode="outer")
    rgb[boundaries, 0] = 1.0
    rgb[boundaries, 1] = 0.2
    rgb[boundaries, 2] = 0.2
    return rgb


# =========================
# Load data
# =========================
labels = tifffile.imread(LABELS_TIF)
raw = tifffile.imread(RAW_TIF)
corrected = tifffile.imread(CORRECTED_TIF)
df = pd.read_csv(OBJECTS_CSV)

print(f"Loaded labels shape={labels.shape}, dtype={labels.dtype}", flush=True)
print(f"Loaded raw shape={raw.shape}, dtype={raw.dtype}", flush=True)
print(f"Loaded corrected shape={corrected.shape}, dtype={corrected.dtype}", flush=True)
print(f"Loaded object table rows={len(df)}", flush=True)

if labels.ndim != 3:
    raise ValueError(f"VERIFICATION FAILED: labels must be 3D ZYX, got ndim={labels.ndim}")
if raw.shape != labels.shape or corrected.shape != labels.shape:
    raise ValueError("VERIFICATION FAILED: raw/corrected stack shape does not match labels shape")
if labels.size == 0:
    raise ValueError("VERIFICATION FAILED: empty label volume")

required_cols = {
    "label_id", "voxel_volume", "bbox_zmin", "bbox_zmax_exclusive",
    "bbox_ymin", "bbox_ymax_exclusive", "bbox_xmin", "bbox_xmax_exclusive"
}
missing = sorted(required_cols - set(df.columns))
if missing:
    raise ValueError(f"VERIFICATION FAILED: missing expected columns in object CSV: {missing}")

# =========================
# Derive upper size cutoff
# =========================
volumes = df["voxel_volume"].to_numpy(dtype=np.int64)
upper_cutoff, cutoff_details = _derive_upper_cutoff(volumes)

# Explicit plane intersection inspection
for z in TARGET_Z_PLANES:
    if z < 0 or z >= labels.shape[0]:
        raise ValueError(f"VERIFICATION FAILED: target z-plane {z} outside volume depth {labels.shape[0]}")

for z in TARGET_Z_PLANES:
    ids_on_plane = np.unique(labels[z])
    ids_on_plane = ids_on_plane[ids_on_plane > 0]
    print(f"z={z}: objects intersecting plane = {len(ids_on_plane)}", flush=True)

# Apply size filter only (lower + upper)
df["intersects_z56"] = (df["bbox_zmin"] <= 56) & (56 < df["bbox_zmax_exclusive"])
df["intersects_z77"] = (df["bbox_zmin"] <= 77) & (77 < df["bbox_zmax_exclusive"])

df["remove_small"] = df["voxel_volume"] < MIN_VOXELS
df["remove_oversized"] = df["voxel_volume"] > upper_cutoff
df["remove_final"] = df["remove_small"] | df["remove_oversized"]

removed_df = df[df["remove_final"]].copy()
oversized_removed_df = df[df["remove_oversized"]].copy()

if len(df) > 0 and len(removed_df) == len(df):
    raise ValueError("VERIFICATION FAILED: size filtering removed all objects")

removed_target_ids = set(oversized_removed_df.loc[
    oversized_removed_df["intersects_z56"] | oversized_removed_df["intersects_z77"], "label_id"
].astype(int).tolist())
if len(removed_target_ids) == 0:
    raise ValueError(
        "VERIFICATION FAILED: no oversized object intersecting z=56/z=77 was removed; "
        "selected upper cutoff does not target user-reported artifacts."
    )

kept_ids = df.loc[~df["remove_final"], "label_id"].astype(np.int64).to_numpy()
kept_id_set = set(kept_ids.tolist())

# =========================
# Relabel sequentially
# =========================
max_old = int(labels.max())
if max_old <= 0:
    raise ValueError("VERIFICATION FAILED: input labels contain no foreground objects")

lut = np.zeros(max_old + 1, dtype=np.uint32)
for new_id, old_id in enumerate(sorted(kept_id_set), start=1):
    if old_id <= max_old:
        lut[int(old_id)] = np.uint32(new_id)

labels_clipped = labels.copy()
labels_clipped[labels_clipped > max_old] = 0  # defensive, should not happen
revised_labels = lut[labels_clipped]

# =========================
# Recompute per-object measurements
# =========================
props = measure.regionprops(revised_labels)
zdim, ydim, xdim = revised_labels.shape

rows = []
voxel_um3 = (XY_PIXEL_SIZE_UM ** 2) * ASSUMED_Z_SPACING_UM
for p in props:
    zc, yc, xc = p.centroid
    zmin, ymin, xmin, zmax, ymax, xmax = p.bbox
    vox = int(p.area)

    border_touch_xy = (ymin == 0) or (xmin == 0) or (ymax == ydim) or (xmax == xdim)
    border_touch_z = (zmin == 0) or (zmax == zdim)
    border_touch_any = border_touch_xy or border_touch_z

    rows.append({
        "label_id": int(p.label),
        "voxel_volume": vox,
        "centroid_z": float(zc),
        "centroid_y": float(yc),
        "centroid_x": float(xc),
        "centroid_zyx": f"({zc:.3f}, {yc:.3f}, {xc:.3f})",
        "bbox_zmin": int(zmin),
        "bbox_zmax_exclusive": int(zmax),
        "bbox_ymin": int(ymin),
        "bbox_ymax_exclusive": int(ymax),
        "bbox_xmin": int(xmin),
        "bbox_xmax_exclusive": int(xmax),
        "bbox_zyx_minmax_exclusive": f"[{zmin}:{zmax}, {ymin}:{ymax}, {xmin}:{xmax}]",
        "border_touch_xy": bool(border_touch_xy),
        "border_touch_z": bool(border_touch_z),
        "border_touch_any": bool(border_touch_any),
        "provisional_volume_um3_xy0p324512482_assumed_z1p0": float(vox * voxel_um3),
        "volume_calibration_note": VOLUME_NOTE,
    })

revised_df = pd.DataFrame(rows).sort_values("label_id").reset_index(drop=True)
if len(df) > 0 and len(revised_df) == 0:
    raise ValueError("VERIFICATION FAILED: revised measurement table is empty although input had objects")

# =========================
# Save core outputs
# =========================
tifffile.imwrite(revised_labels_tif, revised_labels.astype(np.uint32), compression="zlib")
revised_df.to_csv(revised_objects_csv, index=False)

removed_report_cols = [
    "label_id", "voxel_volume", "intersects_z56", "intersects_z77",
    "bbox_zmin", "bbox_zmax_exclusive", "bbox_ymin", "bbox_ymax_exclusive",
    "bbox_xmin", "bbox_xmax_exclusive", "remove_small", "remove_oversized", "remove_final"
]
removed_df.sort_values(["remove_oversized", "voxel_volume"], ascending=[False, False]).to_csv(
    removed_report_csv, index=False, columns=removed_report_cols
)

# =========================
# QC figures
# =========================
fig, axes = plt.subplots(len(QC_PLANES), 3, figsize=(15, 4 * len(QC_PLANES)), constrained_layout=True)
if len(QC_PLANES) == 1:
    axes = np.array([axes])

for i, z in enumerate(QC_PLANES):
    if z < 0 or z >= labels.shape[0]:
        raise ValueError(f"VERIFICATION FAILED: QC plane z={z} outside volume")

    base = corrected[z]
    before_rgb = _make_overlay(base, labels[z])
    after_rgb = _make_overlay(base, revised_labels[z])

    p2, p98 = np.percentile(base.astype(np.float32), [2, 98])
    axes[i, 0].imshow(base, cmap="gray", vmin=p2, vmax=p98)
    axes[i, 0].set_title(f"Corrected z={z}")
    axes[i, 1].imshow(before_rgb)
    axes[i, 1].set_title(f"Before labels z={z}")
    axes[i, 2].imshow(after_rgb)
    axes[i, 2].set_title(f"After labels z={z}")

    for j in range(3):
        axes[i, j].axis("off")

fig.suptitle("F0003 soma pilot revision: size-filter QC planes (before vs after)", fontsize=14)
fig.savefig(qc_planes_png, dpi=300, bbox_inches="tight")
plt.close(fig)

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
for r, z in enumerate([56, 77]):
    base = corrected[z]
    before_rgb = _make_overlay(base, labels[z])
    after_rgb = _make_overlay(base, revised_labels[z])

    axes2[r, 0].imshow(before_rgb)
    axes2[r, 0].set_title(f"Before overlay z={z}")
    axes2[r, 1].imshow(after_rgb)
    axes2[r, 1].set_title(f"After overlay z={z}")
    axes2[r, 0].axis("off")
    axes2[r, 1].axis("off")

fig2.suptitle("Requested verification planes: z=56 and z=77 (before vs after)", fontsize=14)
fig2.savefig(qc_focus_png, dpi=300, bbox_inches="tight")
plt.close(fig2)

raw_mip = np.max(corrected, axis=0)
before_mip = np.max(labels > 0, axis=0)
after_mip = np.max(revised_labels > 0, axis=0)

p2, p98 = np.percentile(raw_mip.astype(np.float32), [2, 98])
base_mip = exposure.rescale_intensity(raw_mip.astype(np.float32), in_range=(p2, p98), out_range=(0, 1))

mip_before_rgb = np.dstack([base_mip, base_mip, base_mip])
mip_after_rgb = np.dstack([base_mip, base_mip, base_mip])
mip_before_rgb[before_mip, 0] = 1.0
mip_before_rgb[before_mip, 1] = 0.2
mip_before_rgb[before_mip, 2] = 0.2
mip_after_rgb[after_mip, 0] = 0.2
mip_after_rgb[after_mip, 1] = 1.0
mip_after_rgb[after_mip, 2] = 0.2

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
axes3[0].imshow(base_mip, cmap="gray")
axes3[0].set_title("Corrected XY MIP")
axes3[1].imshow(mip_before_rgb)
axes3[1].set_title("Before mask projection")
axes3[2].imshow(mip_after_rgb)
axes3[2].set_title("After mask projection")
for ax in axes3:
    ax.axis("off")
fig3.suptitle("F0003 soma pilot revision: XY MIP comparison", fontsize=14)
fig3.savefig(qc_mip_png, dpi=300, bbox_inches="tight")
plt.close(fig3)

# =========================
# Metadata JSON
# =========================
meta = {
    "input_labels_tif": LABELS_TIF,
    "input_objects_csv": OBJECTS_CSV,
    "input_raw_tif": RAW_TIF,
    "input_corrected_tif": CORRECTED_TIF,
    "output_revised_labels_tif": revised_labels_tif,
    "output_revised_objects_csv": revised_objects_csv,
    "output_removed_report_csv": removed_report_csv,
    "output_qc_planes_png": qc_planes_png,
    "output_qc_focus_png": qc_focus_png,
    "output_qc_xy_mip_png": qc_mip_png,
    "size_filter": {
        "min_voxels_kept": MIN_VOXELS,
        "max_voxels_kept": int(upper_cutoff),
        "rationale": cutoff_details,
        "rule_summary": (
            "Upper cutoff derived from full voxel_volume distribution using robust "
            "log-volume MAD outlier rule with Tukey fallback; no intensity/shape/manual deletion used."
        ),
    },
    "target_planes": TARGET_Z_PLANES,
    "qc_planes": QC_PLANES,
    "plane_intersection_summary": {
        "z56_object_count": int(np.sum((df["bbox_zmin"] <= 56) & (56 < df["bbox_zmax_exclusive"]))),
        "z77_object_count": int(np.sum((df["bbox_zmin"] <= 77) & (77 < df["bbox_zmax_exclusive"]))),
        "oversized_removed_ids_intersecting_z56_or_z77": sorted(list(removed_target_ids)),
    },
    "counts": {
        "objects_before": int(len(df)),
        "objects_after": int(len(revised_df)),
        "removed_total": int(len(removed_df)),
        "removed_small_lt300": int(np.sum(df["remove_small"])),
        "removed_oversized_gt_max": int(np.sum(df["remove_oversized"])),
    },
    "calibration": {
        "xy_pixel_size_um_provisional": XY_PIXEL_SIZE_UM,
        "z_spacing_um_provisional": ASSUMED_Z_SPACING_UM,
        "note": VOLUME_NOTE,
    },
}

with open(metadata_json, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

# =========================
# Verify outputs
# =========================
for p in [
    revised_labels_tif,
    revised_objects_csv,
    removed_report_csv,
    metadata_json,
    qc_planes_png,
    qc_focus_png,
    qc_mip_png,
]:
    _assert_file_nonempty(p)

if revised_df["label_id"].duplicated().any():
    raise ValueError("VERIFICATION FAILED: revised object table has duplicate label_id entries")

if not ((revised_df["voxel_volume"] >= MIN_VOXELS).all() and (revised_df["voxel_volume"] <= upper_cutoff).all()):
    raise ValueError("VERIFICATION FAILED: revised objects violate configured [min,max] voxel-volume filter")

if int(revised_labels.max()) != int(len(revised_df)):
    raise ValueError(
        "VERIFICATION FAILED: sequential relabel mismatch: max label != number of rows in revised table"
    )

print(
    f"Completed revision. Upper voxel-volume cutoff={upper_cutoff}. "
    f"Objects before={len(df)}, after={len(revised_df)}, removed={len(removed_df)}; "
    f"oversized removed={int(np.sum(df['remove_oversized']))}.",
    flush=True,
)
print(f"Removed oversized objects intersecting z=56/z=77: {sorted(list(removed_target_ids))}", flush=True)
