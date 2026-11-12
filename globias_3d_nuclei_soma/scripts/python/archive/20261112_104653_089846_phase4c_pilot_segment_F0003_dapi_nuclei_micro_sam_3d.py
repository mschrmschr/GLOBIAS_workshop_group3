# imagentj-env: napari-mcp
"""
Pilot true-3D micro_sam AIS segmentation for F0003 DAPI nuclei.

Outputs in processed_images/pilot/F0003:
1) uint32 3D instance labels TIFF (same ZYX shape as input)
2) per-object CSV with voxel count, centroid, bbox, border-touching flag, and provisional um^3
3) QC PNG overlays (representative + low-signal planes; XY max projection)
4) JSON metadata with model/device/parameters/versions/counts/calibration caveat
"""

import json
from datetime import datetime

import tifffile
import torch
import scipy.ndimage as ndi
import micro_sam
from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
    automatic_instance_segmentation,
)

# ---- CONFIG -----------------------------------------------------------------
RAW_DAPI_PATH = "/data/tasks/Globias_workshop/DAPI/F0003_DAPI.tif"
PREPROCESSED_DAPI_PATH = "/app/data/projects/globias_3d_nuclei_soma/processed_images/pilot/F0003/F0003_DAPI_preprocessed_for_micro_sam_3d_float32_0to1.tif"
OUT_DIR = "/app/data/projects/globias_3d_nuclei_soma/processed_images/pilot/F0003"

MODEL_TYPE = "vit_b_lm"           # light-microscopy model
SEG_MODE = "ais"                  # decoder-based automatic instance segmentation
CHECKPOINT_PATH = None            # stock model
MIN_VOXELS = 25                   # remove only clearly implausible tiny components

XY_UM_PROVISIONAL = 0.324512482
Z_UM_ASSUMED = 1.0
# -----------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

label_tif_path = os.path.join(OUT_DIR, "F0003_DAPI_micro_sam_3d_nuclei_labels_uint32.tif")
objects_csv_path = os.path.join(OUT_DIR, "F0003_DAPI_micro_sam_3d_nuclei_objects_pilot.csv")
qc_planes_png_path = os.path.join(OUT_DIR, "F0003_DAPI_micro_sam_3d_nuclei_QC_planes_overlay.png")
qc_mip_png_path = os.path.join(OUT_DIR, "F0003_DAPI_micro_sam_3d_nuclei_QC_xy_mip_overlay.png")
metadata_json_path = os.path.join(OUT_DIR, "F0003_DAPI_micro_sam_3d_nuclei_segmentation_metadata.json")

if not os.path.isfile(PREPROCESSED_DAPI_PATH):
    raise FileNotFoundError(f"Missing preprocessed input: {PREPROCESSED_DAPI_PATH}")
if not os.path.isfile(RAW_DAPI_PATH):
    raise FileNotFoundError(f"Missing raw DAPI input for QC: {RAW_DAPI_PATH}")

raw_zyx = tifffile.imread(RAW_DAPI_PATH)
pre_zyx = tifffile.imread(PREPROCESSED_DAPI_PATH)
print(f"Raw stack shape={raw_zyx.shape}, dtype={raw_zyx.dtype}")
print(f"Preprocessed stack shape={pre_zyx.shape}, dtype={pre_zyx.dtype}")

if pre_zyx.ndim != 3:
    raise ValueError(f"VERIFICATION FAILED: expected 3D ZYX preprocessed volume, got ndim={pre_zyx.ndim}")
if raw_zyx.shape != pre_zyx.shape:
    raise ValueError("VERIFICATION FAILED: raw and preprocessed stacks have different shapes.")

if not torch.cuda.is_available():
    raise RuntimeError("VERIFICATION FAILED: GPU requested but torch.cuda.is_available() is False in napari-mcp env.")

device = "cuda"
print(f"micro_sam device={device}, torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
print(f"micro_sam model={MODEL_TYPE}, segmentation_mode={SEG_MODE}, checkpoint={(CHECKPOINT_PATH or 'stock')}")

predictor, segmenter = get_predictor_and_segmenter(
    model_type=MODEL_TYPE,
    checkpoint=CHECKPOINT_PATH,
    device=device,
    segmentation_mode=SEG_MODE,
)

labels = automatic_instance_segmentation(
    predictor=predictor,
    segmenter=segmenter,
    input_path=pre_zyx,
    ndim=3,
    verbose=False,
).astype(np.uint32)

if labels.shape != pre_zyx.shape:
    raise ValueError("VERIFICATION FAILED: output label volume shape does not match input volume shape.")

counts_before = np.bincount(labels.ravel())
all_labels_before = np.flatnonzero(counts_before > 0)
all_labels_before = all_labels_before[all_labels_before != 0]
n_before = int(all_labels_before.size)

remove_labels = np.flatnonzero((counts_before < MIN_VOXELS) & (np.arange(counts_before.size) != 0)).astype(np.uint32)
labels_filtered = labels.copy()
if remove_labels.size > 0:
    labels_filtered[np.isin(labels_filtered, remove_labels)] = 0

counts_after = np.bincount(labels_filtered.ravel())
all_labels_after = np.flatnonzero(counts_after > 0)
all_labels_after = all_labels_after[all_labels_after != 0]
n_after = int(all_labels_after.size)

boundary_label_ids = set()
for face in (
    labels_filtered[0, :, :], labels_filtered[-1, :, :],
    labels_filtered[:, 0, :], labels_filtered[:, -1, :],
    labels_filtered[:, :, 0], labels_filtered[:, :, -1],
):
    boundary_label_ids.update(np.unique(face).tolist())
boundary_label_ids.discard(0)

voxel_volume_um3_provisional = XY_UM_PROVISIONAL * XY_UM_PROVISIONAL * Z_UM_ASSUMED
slices_by_label = ndi.find_objects(labels_filtered)
centroids = ndi.center_of_mass(
    np.ones(labels_filtered.shape, dtype=np.uint8), labels_filtered, all_labels_after.tolist()
)

rows = []
for idx, label_id in enumerate(all_labels_after.tolist()):
    sl = slices_by_label[label_id - 1]
    if sl is None:
        continue
    z0, z1 = int(sl[0].start), int(sl[0].stop)
    y0, y1 = int(sl[1].start), int(sl[1].stop)
    x0, x1 = int(sl[2].start), int(sl[2].stop)

    cz, cy, cx = centroids[idx]
    voxel_count = int(counts_after[label_id])
    is_border = bool(label_id in boundary_label_ids)

    rows.append({
        "label": int(label_id),
        "voxel_count": voxel_count,
        "centroid_z": float(cz),
        "centroid_y": float(cy),
        "centroid_x": float(cx),
        "centroid_zyx": f"({cz:.3f}, {cy:.3f}, {cx:.3f})",
        "bbox_zmin": z0,
        "bbox_zmax_exclusive": z1,
        "bbox_ymin": y0,
        "bbox_ymax_exclusive": y1,
        "bbox_xmin": x0,
        "bbox_xmax_exclusive": x1,
        "bbox_zyx_minmax_exclusive": f"[{z0}:{z1}, {y0}:{y1}, {x0}:{x1}]",
        "border_touching": is_border,
        "provisional_volume_um3_xy0p324512482_assumed_z1p0": float(voxel_count * voxel_volume_um3_provisional),
        "volume_calibration_note": "PROVISIONAL: computed with XY=0.324512482 um and assumed Z=1.0 um due to missing physical calibration metadata",
    })

objects_df = pd.DataFrame(rows)

expected_cols = {
    "label", "voxel_count", "centroid_zyx", "bbox_zyx_minmax_exclusive", "border_touching",
    "provisional_volume_um3_xy0p324512482_assumed_z1p0"
}
if not expected_cols.issubset(set(objects_df.columns)):
    missing = sorted(expected_cols.difference(set(objects_df.columns)))
    raise ValueError(f"VERIFICATION FAILED: objects CSV is missing expected columns: {missing}")

# Keep unique instance IDs (with potential gaps) after filtering.
labels_out = labels_filtered.astype(np.uint32)
tifffile.imwrite(label_tif_path, labels_out)
objects_df.to_csv(objects_csv_path, index=False)

# --- QC figures ---------------------------------------------------------------
z_means = pre_zyx.mean(axis=(1, 2))
z_low_signal = int(np.argmin(z_means))

candidate_planes = [0, pre_zyx.shape[0] // 4, pre_zyx.shape[0] // 2, (3 * pre_zyx.shape[0]) // 4, pre_zyx.shape[0] - 1, z_low_signal]
plane_ids = []
for z in candidate_planes:
    z_int = int(max(0, min(pre_zyx.shape[0] - 1, z)))
    if z_int not in plane_ids:
        plane_ids.append(z_int)

n_rows = len(plane_ids)
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.2 * n_rows), constrained_layout=True)
if n_rows == 1:
    axes = np.expand_dims(axes, axis=0)

for r, z in enumerate(plane_ids):
    raw2d = raw_zyx[z]
    pre2d = pre_zyx[z]
    lbl2d = labels_out[z]
    mask2d = lbl2d > 0
    boundary2d = mask2d & (~ndi.binary_erosion(mask2d, structure=np.ones((3, 3), dtype=bool), border_value=0))

    ax0, ax1, ax2 = axes[r]

    ax0.imshow(raw2d, cmap="gray")
    ax0.set_title(f"Raw DAPI z={z}", fontsize=10)
    ax0.axis("off")

    ax1.imshow(pre2d, cmap="gray")
    ax1.imshow(mask2d, cmap="spring", alpha=0.25, interpolation="nearest")
    ax1.set_title(f"Preprocessed + mask overlay z={z}", fontsize=10)
    ax1.axis("off")

    ax2.imshow(pre2d, cmap="gray")
    ax2.imshow(boundary2d, cmap="autumn", alpha=0.95, interpolation="nearest")
    if z == z_low_signal:
        ax2.set_title(f"Boundary overlay z={z} (lowest mean signal)", fontsize=10)
    else:
        ax2.set_title(f"Boundary overlay z={z}", fontsize=10)
    ax2.axis("off")

fig.suptitle("F0003 DAPI pilot 3D nuclei segmentation QC (micro_sam AIS)", fontsize=12)
fig.savefig(qc_planes_png_path, dpi=300)
plt.close(fig)

raw_mip = raw_zyx.max(axis=0)
pre_mip = pre_zyx.max(axis=0)
mask_mip = (labels_out > 0).max(axis=0)
boundary_mip = mask_mip & (~ndi.binary_erosion(mask_mip, structure=np.ones((3, 3), dtype=bool), border_value=0))

fig2, axm = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
axm[0].imshow(raw_mip, cmap="gray")
axm[0].set_title("Raw DAPI XY max projection", fontsize=10)
axm[0].axis("off")

axm[1].imshow(pre_mip, cmap="gray")
axm[1].imshow(mask_mip, cmap="spring", alpha=0.25, interpolation="nearest")
axm[1].set_title("Preprocessed XY max projection + mask", fontsize=10)
axm[1].axis("off")

axm[2].imshow(pre_mip, cmap="gray")
axm[2].imshow(boundary_mip, cmap="autumn", alpha=0.95, interpolation="nearest")
axm[2].set_title("Preprocessed XY max projection + boundaries", fontsize=10)
axm[2].axis("off")

fig2.suptitle("F0003 DAPI pilot nuclei QC: XY max projections", fontsize=12)
fig2.savefig(qc_mip_png_path, dpi=300)
plt.close(fig2)

metadata = {
    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    "input_raw_dapi_path": RAW_DAPI_PATH,
    "input_preprocessed_dapi_path": PREPROCESSED_DAPI_PATH,
    "output_label_tif_path": label_tif_path,
    "output_objects_csv_path": objects_csv_path,
    "output_qc_planes_png_path": qc_planes_png_path,
    "output_qc_xy_mip_png_path": qc_mip_png_path,
    "output_metadata_json_path": metadata_json_path,
    "volume_shape_zyx": list(pre_zyx.shape),
    "volume_dtype": str(pre_zyx.dtype),
    "model_type": MODEL_TYPE,
    "segmentation_mode": SEG_MODE,
    "checkpoint_path": CHECKPOINT_PATH,
    "device": device,
    "torch_cuda_available": bool(torch.cuda.is_available()),
    "parameters": {
        "ndim": 3,
        "min_voxels_filter": MIN_VOXELS,
        "xy_um_provisional": XY_UM_PROVISIONAL,
        "z_um_assumed": Z_UM_ASSUMED,
    },
    "counts": {
        "objects_before_filtering": n_before,
        "objects_removed_as_tiny": int(remove_labels.size),
        "objects_after_filtering": n_after,
        "border_touching_after_filtering": int(objects_df["border_touching"].sum()) if len(objects_df) > 0 else 0,
    },
    "calibration_caveat": "Physical calibration unavailable from metadata; all um^3 are provisional using XY=0.324512482 um and assumed Z=1.0 um.",
    "package_versions": {
        "python": os.sys.version,
        "torch": torch.__version__,
        "micro_sam": getattr(micro_sam, "__version__", "unknown"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "tifffile": tifffile.__version__,
    },
}

with open(metadata_json_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

for out_path in [label_tif_path, objects_csv_path, qc_planes_png_path, qc_mip_png_path, metadata_json_path]:
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise ValueError(f"VERIFICATION FAILED: expected output missing or empty: {out_path}")

print("=== Pilot segmentation summary (F0003 DAPI) ===")
print(f"Objects before tiny-component filtering (<{MIN_VOXELS} voxels): {n_before}")
print(f"Objects removed as tiny components: {int(remove_labels.size)}")
print(f"Objects after filtering: {n_after}")
print(f"Lowest-mean-signal z-plane included in QC: z={z_low_signal}")
print(f"Label TIFF: {label_tif_path}")
print(f"Object CSV: {objects_csv_path}")
print(f"QC planes PNG: {qc_planes_png_path}")
print(f"QC XY-MIP PNG: {qc_mip_png_path}")
print(f"Metadata JSON: {metadata_json_path}")
