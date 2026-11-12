# imagentj-env: napari-mcp
"""
Production batch pipeline: preprocess + true-3D micro_sam AIS segmentation for 20 paired
DAPI/soma stacks, with resumable checkpoints and per-field QC outputs.

Primary size units are voxels. Physical volume_um3 is explicitly provisional
(XY=0.324512482 um, assumed Z=1.0 um).
"""

import gc
import json
import os
import traceback
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import micro_sam
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tifffile
import torch
from micro_sam.automatic_segmentation import (
    automatic_instance_segmentation,
    get_predictor_and_segmenter,
)
from skimage import filters

# ------------------------------ CONFIG ----------------------------------------
MANIFEST_CSV = r"/app/data/projects/globias_3d_nuclei_soma/data/input_integrity_manifest.csv"
PROJECT_DIR = r"/app/data/projects/globias_3d_nuclei_soma"
BATCH_DIR = r"/app/data/projects/globias_3d_nuclei_soma/processed_images/batch"

MODEL_TYPE = "vit_b_lm"
SEG_MODE = "ais"
CHECKPOINT_PATH = None
REQUIRE_CUDA = True

NUCLEI_MIN_VOXELS = 25
SOMA_MIN_VOXELS = 300
SOMA_MAX_VOXELS = 18310

# User-approved preprocessing settings (fixed)
ROBUST_NORM_LOW_PCT = 0.5
ROBUST_NORM_HIGH_PCT = 99.8
DAPI_BG_SIGMA_ZYX = (0.0, 40.0, 40.0)
DAPI_BG_ALPHA = 0.90
SOMA_DENOISE_SIGMA_ZYX = (0.0, 0.6, 0.6)
SOMA_BG_SIGMA_ZYX = (0.0, 40.0, 40.0)
SOMA_BG_ALPHA = 0.90

XY_UM_PROVISIONAL = 0.324512482
Z_UM_PROVISIONAL = 1.0
VOXEL_UM3_PROVISIONAL = XY_UM_PROVISIONAL * XY_UM_PROVISIONAL * Z_UM_PROVISIONAL

EPS = 1e-8
# -----------------------------------------------------------------------------

os.makedirs(BATCH_DIR, exist_ok=True)


def _assert_file_nonempty(path):
    if (not os.path.isfile(path)) or os.path.getsize(path) == 0:
        raise ValueError(f"VERIFICATION FAILED: missing or empty file: {path}")


def _safe_makedirs(path):
    os.makedirs(path, exist_ok=True)


def robust_normalize_0to1(vol_f32, p_low=ROBUST_NORM_LOW_PCT, p_high=ROBUST_NORM_HIGH_PCT):
    low = float(np.percentile(vol_f32, p_low))
    high = float(np.percentile(vol_f32, p_high))
    if high <= low:
        raise ValueError(f"VERIFICATION FAILED: invalid normalization window high<=low ({high} <= {low}).")
    norm = (vol_f32 - low) / (high - low)
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
    if np.nanmin(norm) < -1e-6 or np.nanmax(norm) > 1.000001:
        raise ValueError("VERIFICATION FAILED: normalized volume outside [0,1].")
    return norm, low, high


def subtract_lowfreq_background(vol_f32, sigma_zyx, alpha):
    bg = filters.gaussian(vol_f32, sigma=sigma_zyx, preserve_range=True).astype(np.float32)
    bg_floor = float(np.percentile(bg, 5))
    corrected = vol_f32 - (float(alpha) * (bg - bg_floor))
    corrected = np.clip(corrected, 0, None).astype(np.float32)
    return corrected, bg_floor


def preprocess_dapi(raw_zyx):
    raw_f = raw_zyx.astype(np.float32, copy=False)
    corrected, bg_floor = subtract_lowfreq_background(raw_f, DAPI_BG_SIGMA_ZYX, DAPI_BG_ALPHA)
    norm, low, high = robust_normalize_0to1(corrected)
    meta = {
        "pipeline": "dapi_preprocess",
        "slice_equalization_applied": False,
        "background_sigma_zyx": list(DAPI_BG_SIGMA_ZYX),
        "background_alpha": float(DAPI_BG_ALPHA),
        "background_floor_p5": float(bg_floor),
        "normalization_low_percentile": float(ROBUST_NORM_LOW_PCT),
        "normalization_high_percentile": float(ROBUST_NORM_HIGH_PCT),
        "normalization_low_value": float(low),
        "normalization_high_value": float(high),
    }
    return norm, meta


def preprocess_soma(raw_zyx):
    raw_f = raw_zyx.astype(np.float32, copy=False)
    den = filters.gaussian(raw_f, sigma=SOMA_DENOISE_SIGMA_ZYX, preserve_range=True).astype(np.float32)
    corrected, bg_floor = subtract_lowfreq_background(den, SOMA_BG_SIGMA_ZYX, SOMA_BG_ALPHA)
    norm, low, high = robust_normalize_0to1(corrected)
    meta = {
        "pipeline": "soma_preprocess",
        "slice_equalization_applied": False,
        "denoise_sigma_zyx": list(SOMA_DENOISE_SIGMA_ZYX),
        "background_sigma_zyx": list(SOMA_BG_SIGMA_ZYX),
        "background_alpha": float(SOMA_BG_ALPHA),
        "background_floor_p5": float(bg_floor),
        "normalization_low_percentile": float(ROBUST_NORM_LOW_PCT),
        "normalization_high_percentile": float(ROBUST_NORM_HIGH_PCT),
        "normalization_low_value": float(low),
        "normalization_high_value": float(high),
    }
    return norm, meta


def filter_and_relabel(labels_u32, min_voxels, max_voxels=None):
    if labels_u32.dtype != np.uint32:
        labels_u32 = labels_u32.astype(np.uint32, copy=False)

    counts = np.bincount(labels_u32.ravel())
    if counts.size <= 1:
        return np.zeros_like(labels_u32, dtype=np.uint32), 0, 0, np.array([], dtype=np.int64)

    label_ids = np.arange(counts.size, dtype=np.int64)
    keep = label_ids > 0
    keep &= counts >= int(min_voxels)
    if max_voxels is not None:
        keep &= counts <= int(max_voxels)

    kept_ids = label_ids[keep]
    mapping = np.zeros(counts.size, dtype=np.uint32)
    mapping[kept_ids] = np.arange(1, kept_ids.size + 1, dtype=np.uint32)
    relabeled = mapping[labels_u32]

    out_counts = np.bincount(relabeled.ravel())
    n_after = int(out_counts.size - 1) if out_counts.size > 1 else 0
    if n_after != int(relabeled.max()):
        raise ValueError(
            f"VERIFICATION FAILED: relabel inconsistency (max label {int(relabeled.max())} != object count {n_after})."
        )

    kept_volumes = out_counts[1:].astype(np.int64) if out_counts.size > 1 else np.array([], dtype=np.int64)
    if kept_volumes.size > 0:
        if int(kept_volumes.min()) < int(min_voxels):
            raise ValueError("VERIFICATION FAILED: retained volume below min_voxels.")
        if max_voxels is not None and int(kept_volumes.max()) > int(max_voxels):
            raise ValueError("VERIFICATION FAILED: retained volume above max_voxels.")

    n_before = int(np.count_nonzero(counts[1:] > 0))
    n_removed = int(n_before - n_after)
    return relabeled.astype(np.uint32, copy=False), n_before, n_removed, kept_volumes


def measure_objects(labels_u32, field_id, channel_name):
    shape = labels_u32.shape
    z_max, y_max, x_max = shape

    counts = np.bincount(labels_u32.ravel())
    if counts.size <= 1:
        return pd.DataFrame(columns=[
            "field_id", "channel", "label_id", "voxel_volume",
            "centroid_z", "centroid_y", "centroid_x",
            "bbox_zmin", "bbox_zmax_exclusive", "bbox_ymin", "bbox_ymax_exclusive",
            "bbox_xmin", "bbox_xmax_exclusive",
            "touches_xy_border", "touches_z_border", "touches_any_border",
            "provisional_volume_um3", "volume_um3_note"
        ])

    label_ids = np.arange(1, counts.size, dtype=np.int64)
    slices = ndi.find_objects(labels_u32)
    centroids = ndi.center_of_mass(
        np.ones(shape, dtype=np.uint8), labels_u32, label_ids.tolist()
    )

    rows = []
    for i, lid in enumerate(label_ids.tolist()):
        if lid >= counts.size or counts[lid] <= 0:
            continue
        sl = slices[lid - 1]
        if sl is None:
            continue

        z0, z1 = int(sl[0].start), int(sl[0].stop)
        y0, y1 = int(sl[1].start), int(sl[1].stop)
        x0, x1 = int(sl[2].start), int(sl[2].stop)
        cz, cy, cx = centroids[i]
        voxel_vol = int(counts[lid])

        touch_z = bool(z0 == 0 or z1 == z_max)
        touch_xy = bool(y0 == 0 or y1 == y_max or x0 == 0 or x1 == x_max)

        rows.append({
            "field_id": field_id,
            "channel": channel_name,
            "label_id": int(lid),
            "voxel_volume": voxel_vol,
            "centroid_z": float(cz),
            "centroid_y": float(cy),
            "centroid_x": float(cx),
            "bbox_zmin": z0,
            "bbox_zmax_exclusive": z1,
            "bbox_ymin": y0,
            "bbox_ymax_exclusive": y1,
            "bbox_xmin": x0,
            "bbox_xmax_exclusive": x1,
            "touches_xy_border": touch_xy,
            "touches_z_border": touch_z,
            "touches_any_border": bool(touch_xy or touch_z),
            "provisional_volume_um3": float(voxel_vol * VOXEL_UM3_PROVISIONAL),
            "volume_um3_note": "PROVISIONAL: XY=0.324512482 um and assumed Z=1.0 um; primary size unit is voxels."
        })

    df = pd.DataFrame(rows)
    if len(df) > 0 and int(df["label_id"].max()) != len(df):
        raise ValueError("VERIFICATION FAILED: measured label ids are not sequential after relabel.")
    return df


def choose_rep_planes(pre_zyx):
    z = pre_zyx.shape[0]
    z_means = pre_zyx.mean(axis=(1, 2))
    low = int(np.argmin(z_means))
    mid = int(z // 2)
    high = int(np.argmax(z_means))
    planes = [low, mid, high]
    out = []
    for p in planes:
        p = max(0, min(z - 1, int(p)))
        if p not in out:
            out.append(p)
    return out


def save_qc_overlay(raw_zyx, pre_zyx, labels_u32, out_planes_png, out_mip_png, channel_tag, field_id):
    planes = choose_rep_planes(pre_zyx)

    fig, axes = plt.subplots(len(planes), 3, figsize=(14, 4 * len(planes)), constrained_layout=True)
    if len(planes) == 1:
        axes = np.expand_dims(axes, 0)

    for r, zi in enumerate(planes):
        raw2d = raw_zyx[zi]
        pre2d = pre_zyx[zi]
        lbl2d = labels_u32[zi] > 0
        bnd2d = lbl2d & (~ndi.binary_erosion(lbl2d, structure=np.ones((3, 3), dtype=bool), border_value=0))

        axes[r, 0].imshow(raw2d, cmap="gray")
        axes[r, 0].set_title(f"Raw z={zi}", fontsize=10)
        axes[r, 0].axis("off")

        axes[r, 1].imshow(pre2d, cmap="gray", vmin=0, vmax=1)
        axes[r, 1].imshow(lbl2d, cmap="spring", alpha=0.25, interpolation="nearest")
        axes[r, 1].set_title(f"Preprocessed + mask z={zi}", fontsize=10)
        axes[r, 1].axis("off")

        axes[r, 2].imshow(pre2d, cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(bnd2d, cmap="autumn", alpha=0.95, interpolation="nearest")
        axes[r, 2].set_title(f"Boundary overlay z={zi}", fontsize=10)
        axes[r, 2].axis("off")

    fig.suptitle(f"{field_id} {channel_tag} QC representative planes", fontsize=12)
    fig.savefig(out_planes_png, dpi=300)
    plt.close(fig)

    raw_mip = raw_zyx.max(axis=0)
    pre_mip = pre_zyx.max(axis=0)
    msk_mip = (labels_u32 > 0).max(axis=0)
    bnd_mip = msk_mip & (~ndi.binary_erosion(msk_mip, structure=np.ones((3, 3), dtype=bool), border_value=0))

    fig2, ax = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    ax[0].imshow(raw_mip, cmap="gray")
    ax[0].set_title("Raw XY-MIP", fontsize=10)
    ax[0].axis("off")

    ax[1].imshow(pre_mip, cmap="gray", vmin=0, vmax=1)
    ax[1].imshow(msk_mip, cmap="spring", alpha=0.25, interpolation="nearest")
    ax[1].set_title("Preprocessed XY-MIP + mask", fontsize=10)
    ax[1].axis("off")

    ax[2].imshow(pre_mip, cmap="gray", vmin=0, vmax=1)
    ax[2].imshow(bnd_mip, cmap="autumn", alpha=0.95, interpolation="nearest")
    ax[2].set_title("Preprocessed XY-MIP + boundaries", fontsize=10)
    ax[2].axis("off")

    fig2.suptitle(f"{field_id} {channel_tag} QC XY-MIP", fontsize=12)
    fig2.savefig(out_mip_png, dpi=300)
    plt.close(fig2)


def channel_output_paths(field_dir, field_id, channel_key):
    if channel_key == "dapi":
        return {
            "pre_tif": os.path.join(field_dir, f"{field_id}_DAPI_preprocessed_for_micro_sam_3d_float32_0to1.tif"),
            "label_tif": os.path.join(field_dir, f"{field_id}_DAPI_micro_sam_3d_nuclei_labels_uint32.tif"),
            "obj_csv": os.path.join(field_dir, f"{field_id}_DAPI_micro_sam_3d_nuclei_objects.csv"),
            "meta_json": os.path.join(field_dir, f"{field_id}_DAPI_micro_sam_3d_nuclei_metadata.json"),
            "qc_planes_png": os.path.join(field_dir, f"{field_id}_DAPI_micro_sam_3d_nuclei_QC_planes_overlay.png"),
            "qc_mip_png": os.path.join(field_dir, f"{field_id}_DAPI_micro_sam_3d_nuclei_QC_xy_mip_overlay.png"),
        }
    return {
        "pre_tif": os.path.join(field_dir, f"{field_id}_stain_preprocessed_for_micro_sam_3d_float32_0to1.tif"),
        "label_tif": os.path.join(field_dir, f"{field_id}_stain_micro_sam_3d_soma_labels_uint32.tif"),
        "obj_csv": os.path.join(field_dir, f"{field_id}_stain_micro_sam_3d_soma_objects.csv"),
        "meta_json": os.path.join(field_dir, f"{field_id}_stain_micro_sam_3d_soma_metadata.json"),
        "qc_planes_png": os.path.join(field_dir, f"{field_id}_stain_micro_sam_3d_soma_QC_planes_overlay.png"),
        "qc_mip_png": os.path.join(field_dir, f"{field_id}_stain_micro_sam_3d_soma_QC_xy_mip_overlay.png"),
    }


def validate_existing_channel_outputs(paths, expected_shape, channel_key):
    try:
        for p in paths.values():
            _assert_file_nonempty(p)

        pre = tifffile.imread(paths["pre_tif"])
        lbl = tifffile.imread(paths["label_tif"])
        if tuple(pre.shape) != tuple(expected_shape) or tuple(lbl.shape) != tuple(expected_shape):
            return False, "shape mismatch"
        if pre.dtype != np.float32:
            return False, "preprocess dtype not float32"
        if lbl.dtype != np.uint32:
            return False, "labels dtype not uint32"

        obj = pd.read_csv(paths["obj_csv"])
        required_cols = {
            "field_id", "channel", "label_id", "voxel_volume",
            "centroid_z", "centroid_y", "centroid_x",
            "bbox_zmin", "bbox_zmax_exclusive", "bbox_ymin", "bbox_ymax_exclusive",
            "bbox_xmin", "bbox_xmax_exclusive",
            "touches_xy_border", "touches_z_border", "touches_any_border",
            "provisional_volume_um3"
        }
        if not required_cols.issubset(set(obj.columns)):
            return False, "object csv schema mismatch"

        max_label = int(lbl.max())
        if max_label != len(obj):
            return False, f"max label {max_label} != object rows {len(obj)}"

        if len(obj) > 0:
            vmin = int(obj["voxel_volume"].min())
            vmax = int(obj["voxel_volume"].max())
            if channel_key == "dapi":
                if vmin < NUCLEI_MIN_VOXELS:
                    return False, "nuclei min filter violated"
            else:
                if vmin < SOMA_MIN_VOXELS or vmax > SOMA_MAX_VOXELS:
                    return False, "soma range filter violated"

        with open(paths["meta_json"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        approved = meta.get("approved_settings", {})

        expected = {
            "model_type": MODEL_TYPE,
            "segmentation_mode": SEG_MODE,
            "nuclei_min_voxels": NUCLEI_MIN_VOXELS,
            "soma_min_voxels": SOMA_MIN_VOXELS,
            "soma_max_voxels": SOMA_MAX_VOXELS,
            "dapi_bg_sigma_zyx": list(DAPI_BG_SIGMA_ZYX),
            "dapi_bg_alpha": DAPI_BG_ALPHA,
            "soma_denoise_sigma_zyx": list(SOMA_DENOISE_SIGMA_ZYX),
            "soma_bg_sigma_zyx": list(SOMA_BG_SIGMA_ZYX),
            "soma_bg_alpha": SOMA_BG_ALPHA,
        }
        for k, v in expected.items():
            if approved.get(k) != v:
                return False, f"metadata setting mismatch: {k}"

        return True, "validated"
    except Exception as e:
        return False, str(e)


def process_channel(field_id, channel_key, raw_zyx, predictor, segmenter, field_dir):
    paths = channel_output_paths(field_dir, field_id, channel_key)
    expected_shape = tuple(raw_zyx.shape)

    valid_existing, reason = validate_existing_channel_outputs(paths, expected_shape, channel_key)
    if valid_existing:
        obj_df = pd.read_csv(paths["obj_csv"])
        return {
            "status": "skipped_existing_valid",
            "reason": reason,
            "paths": paths,
            "objects_df": obj_df,
            "n_objects": int(len(obj_df)),
            "n_before": int(len(obj_df)),
            "n_removed": 0,
        }

    if channel_key == "dapi":
        pre_zyx, pre_meta = preprocess_dapi(raw_zyx)
        channel_name = "DAPI"
        obj_name = "nuclei"
        min_vox, max_vox = NUCLEI_MIN_VOXELS, None
    else:
        pre_zyx, pre_meta = preprocess_soma(raw_zyx)
        channel_name = "stain"
        obj_name = "soma"
        min_vox, max_vox = SOMA_MIN_VOXELS, SOMA_MAX_VOXELS

    if tuple(pre_zyx.shape) != expected_shape:
        raise ValueError("VERIFICATION FAILED: preprocessed shape differs from input shape.")

    labels_raw = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=pre_zyx,
        ndim=3,
        verbose=False,
    ).astype(np.uint32)

    if tuple(labels_raw.shape) != expected_shape:
        raise ValueError("VERIFICATION FAILED: segmentation label shape differs from input shape.")

    labels_f, n_before, n_removed, kept_vols = filter_and_relabel(labels_raw, min_voxels=min_vox, max_voxels=max_vox)
    obj_df = measure_objects(labels_f, field_id=field_id, channel_name=obj_name)

    if int(labels_f.max()) != len(obj_df):
        raise ValueError("VERIFICATION FAILED: label max does not equal measured object count.")

    if kept_vols.size != len(obj_df):
        raise ValueError("VERIFICATION FAILED: kept volume array length != measured object rows.")

    tifffile.imwrite(paths["pre_tif"], pre_zyx, imagej=False)
    tifffile.imwrite(paths["label_tif"], labels_f, imagej=False)
    obj_df.to_csv(paths["obj_csv"], index=False)

    save_qc_overlay(
        raw_zyx=raw_zyx,
        pre_zyx=pre_zyx,
        labels_u32=labels_f,
        out_planes_png=paths["qc_planes_png"],
        out_mip_png=paths["qc_mip_png"],
        channel_tag=("DAPI nuclei" if channel_key == "dapi" else "stain soma"),
        field_id=field_id,
    )

    meta = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "field_id": field_id,
        "channel_key": channel_key,
        "raw_shape_zyx": [int(v) for v in expected_shape],
        "preprocessed_dtype": str(pre_zyx.dtype),
        "labels_dtype": str(labels_f.dtype),
        "model_type": MODEL_TYPE,
        "segmentation_mode": SEG_MODE,
        "checkpoint_path": CHECKPOINT_PATH,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "counts": {
            "objects_before_size_filter": int(n_before),
            "objects_removed_by_size_filter": int(n_removed),
            "objects_after_size_filter": int(len(obj_df)),
            "touches_xy_border": int(obj_df["touches_xy_border"].sum()) if len(obj_df) > 0 else 0,
            "touches_z_border": int(obj_df["touches_z_border"].sum()) if len(obj_df) > 0 else 0,
            "touches_any_border": int(obj_df["touches_any_border"].sum()) if len(obj_df) > 0 else 0,
        },
        "size_filter": {
            "min_voxels": int(min_vox),
            "max_voxels": (None if max_vox is None else int(max_vox))
        },
        "preprocessing": pre_meta,
        "provisional_calibration": {
            "xy_um": float(XY_UM_PROVISIONAL),
            "z_um_assumed": float(Z_UM_PROVISIONAL),
            "note": "PROVISIONAL ONLY. Primary reported sizes are voxel counts."
        },
        "approved_settings": {
            "model_type": MODEL_TYPE,
            "segmentation_mode": SEG_MODE,
            "nuclei_min_voxels": NUCLEI_MIN_VOXELS,
            "soma_min_voxels": SOMA_MIN_VOXELS,
            "soma_max_voxels": SOMA_MAX_VOXELS,
            "dapi_bg_sigma_zyx": list(DAPI_BG_SIGMA_ZYX),
            "dapi_bg_alpha": DAPI_BG_ALPHA,
            "soma_denoise_sigma_zyx": list(SOMA_DENOISE_SIGMA_ZYX),
            "soma_bg_sigma_zyx": list(SOMA_BG_SIGMA_ZYX),
            "soma_bg_alpha": SOMA_BG_ALPHA,
            "slice_equalization": "disabled",
            "normalization": "whole-volume robust percentile minmax"
        },
        "outputs": paths,
    }

    with open(paths["meta_json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for p in paths.values():
        _assert_file_nonempty(p)

    return {
        "status": "processed",
        "reason": "fresh run",
        "paths": paths,
        "objects_df": obj_df,
        "n_objects": int(len(obj_df)),
        "n_before": int(n_before),
        "n_removed": int(n_removed),
    }


def main():
    print("=== Batch micro_sam 3D segmentation start ===", flush=True)
    print(f"Manifest: {MANIFEST_CSV}", flush=True)

    if not os.path.isfile(MANIFEST_CSV):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_CSV}")

    manifest = pd.read_csv(MANIFEST_CSV)
    required_manifest_cols = {"field_id", "dapi_path", "stain_path", "pair_valid_for_downstream_3d"}
    if not required_manifest_cols.issubset(set(manifest.columns)):
        missing = sorted(required_manifest_cols.difference(set(manifest.columns)))
        raise ValueError(f"VERIFICATION FAILED: manifest missing columns: {missing}")

    manifest = manifest.loc[manifest["pair_valid_for_downstream_3d"] == True].copy()
    manifest = manifest.sort_values("field_id").reset_index(drop=True)

    if len(manifest) != 20:
        print(f"WARNING: expected 20 valid pairs, found {len(manifest)} in manifest.", flush=True)

    if manifest["field_id"].duplicated().any():
        raise ValueError("VERIFICATION FAILED: duplicate field_id entries in manifest.")

    if REQUIRE_CUDA and (not torch.cuda.is_available()):
        raise RuntimeError("VERIFICATION FAILED: REQUIRE_CUDA=True but torch.cuda.is_available() is False.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device selected: {device}; torch.cuda.is_available()={torch.cuda.is_available()}, torch={torch.__version__}", flush=True)
    print(f"micro_sam version: {getattr(micro_sam, '__version__', 'unknown')}", flush=True)

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=MODEL_TYPE,
        checkpoint=CHECKPOINT_PATH,
        device=device,
        segmentation_mode=SEG_MODE,
    )

    nuclei_rows = []
    soma_rows = []
    counts_rows = []
    failure_rows = []

    for _, row in manifest.iterrows():
        field_id = str(row["field_id"])
        dapi_path = str(row["dapi_path"])
        stain_path = str(row["stain_path"])
        field_dir = os.path.join(BATCH_DIR, field_id)
        _safe_makedirs(field_dir)

        print(f"\n--- Processing field {field_id} ---", flush=True)
        print(f"DAPI:  {dapi_path}", flush=True)
        print(f"Stain: {stain_path}", flush=True)

        dapi_status = "not_run"
        soma_status = "not_run"
        dapi_count_total = np.nan
        soma_count_total = np.nan
        dapi_nonborder = np.nan
        soma_nonborder = np.nan
        dapi_error = ""
        soma_error = ""

        try:
            raw_dapi = tifffile.imread(dapi_path)
            print(f"Loaded DAPI shape={raw_dapi.shape}, dtype={raw_dapi.dtype}", flush=True)
            if raw_dapi.ndim != 3:
                raise ValueError(f"VERIFICATION FAILED: DAPI is not 3D for {field_id}.")
        except Exception as e:
            dapi_status = "error_load"
            dapi_error = str(e)
            raw_dapi = None
            failure_rows.append({"field_id": field_id, "channel": "dapi", "error": traceback.format_exc()})

        try:
            raw_stain = tifffile.imread(stain_path)
            print(f"Loaded stain shape={raw_stain.shape}, dtype={raw_stain.dtype}", flush=True)
            if raw_stain.ndim != 3:
                raise ValueError(f"VERIFICATION FAILED: stain is not 3D for {field_id}.")
        except Exception as e:
            soma_status = "error_load"
            soma_error = str(e)
            raw_stain = None
            failure_rows.append({"field_id": field_id, "channel": "soma", "error": traceback.format_exc()})

        if raw_dapi is not None and raw_stain is not None and tuple(raw_dapi.shape) != tuple(raw_stain.shape):
            msg = f"VERIFICATION FAILED: DAPI/stain shape mismatch in {field_id}: {raw_dapi.shape} vs {raw_stain.shape}"
            dapi_status = "error_shape"
            soma_status = "error_shape"
            dapi_error = msg
            soma_error = msg
            failure_rows.append({"field_id": field_id, "channel": "pair", "error": msg})

        if raw_dapi is not None and dapi_status == "not_run":
            try:
                dapi_res = process_channel(field_id, "dapi", raw_dapi, predictor, segmenter, field_dir)
                dapi_status = dapi_res["status"]
                dapi_count_total = dapi_res["n_objects"]
                dapi_nonborder = int((~dapi_res["objects_df"]["touches_any_border"]).sum()) if len(dapi_res["objects_df"]) > 0 else 0
                nuclei_rows.append(dapi_res["objects_df"])
                print(f"DAPI nuclei objects: {int(dapi_count_total)} ({dapi_status})", flush=True)
            except Exception:
                dapi_status = "error_process"
                dapi_error = traceback.format_exc()
                failure_rows.append({"field_id": field_id, "channel": "dapi", "error": dapi_error})

        if raw_stain is not None and soma_status == "not_run":
            try:
                soma_res = process_channel(field_id, "soma", raw_stain, predictor, segmenter, field_dir)
                soma_status = soma_res["status"]
                soma_count_total = soma_res["n_objects"]
                soma_nonborder = int((~soma_res["objects_df"]["touches_any_border"]).sum()) if len(soma_res["objects_df"]) > 0 else 0
                soma_rows.append(soma_res["objects_df"])
                print(f"Soma objects: {int(soma_count_total)} ({soma_status})", flush=True)
            except Exception:
                soma_status = "error_process"
                soma_error = traceback.format_exc()
                failure_rows.append({"field_id": field_id, "channel": "soma", "error": soma_error})

        ratio_total = np.nan
        ratio_nonborder = np.nan
        if pd.notna(dapi_count_total) and pd.notna(soma_count_total):
            if float(soma_count_total) != 0.0:
                ratio_total = float(dapi_count_total) / float(soma_count_total)
            else:
                print(f"WARNING: {field_id} has zero soma count; nuclei/soma ratio set to NaN.", flush=True)

        if pd.notna(dapi_nonborder) and pd.notna(soma_nonborder):
            if float(soma_nonborder) != 0.0:
                ratio_nonborder = float(dapi_nonborder) / float(soma_nonborder)
            else:
                print(f"WARNING: {field_id} has zero non-border soma count; non-border ratio set to NaN.", flush=True)

        field_status = "ok"
        if ("error" in dapi_status) or ("error" in soma_status):
            field_status = "partial_or_failed"

        counts_rows.append({
            "field_id": field_id,
            "dapi_total_nuclei_count": dapi_count_total,
            "soma_total_object_count": soma_count_total,
            "dapi_nonborder_nuclei_count": dapi_nonborder,
            "soma_nonborder_object_count": soma_nonborder,
            "nuclei_to_soma_ratio_total": ratio_total,
            "nuclei_to_soma_ratio_nonborder": ratio_nonborder,
            "dapi_status": dapi_status,
            "soma_status": soma_status,
            "field_status": field_status,
            "dapi_error": dapi_error,
            "soma_error": soma_error,
        })

        del raw_dapi
        del raw_stain
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    nuclei_combined = pd.concat(nuclei_rows, ignore_index=True) if len(nuclei_rows) > 0 else pd.DataFrame()
    soma_combined = pd.concat(soma_rows, ignore_index=True) if len(soma_rows) > 0 else pd.DataFrame()
    counts_df = pd.DataFrame(counts_rows)
    fail_df = pd.DataFrame(failure_rows)

    # Required: every input field appears exactly once in counts output
    expected_fields = manifest["field_id"].astype(str).tolist()
    if sorted(counts_df["field_id"].astype(str).tolist()) != sorted(expected_fields):
        raise ValueError("VERIFICATION FAILED: counts output does not contain each input field exactly once.")
    if counts_df["field_id"].duplicated().any():
        raise ValueError("VERIFICATION FAILED: duplicate field rows in counts output.")

    nuclei_combined_csv = os.path.join(BATCH_DIR, "batch_nuclei_objects_combined.csv")
    soma_combined_csv = os.path.join(BATCH_DIR, "batch_soma_objects_combined.csv")
    counts_csv = os.path.join(BATCH_DIR, "batch_per_field_counts_status.csv")
    failures_csv = os.path.join(BATCH_DIR, "batch_failures.csv")
    summary_json = os.path.join(BATCH_DIR, "batch_summary.json")
    summary_txt = os.path.join(BATCH_DIR, "batch_summary.txt")

    nuclei_combined.to_csv(nuclei_combined_csv, index=False)
    soma_combined.to_csv(soma_combined_csv, index=False)
    counts_df.to_csv(counts_csv, index=False)
    fail_df.to_csv(failures_csv, index=False)

    for p in [nuclei_combined_csv, soma_combined_csv, counts_csv, failures_csv]:
        _assert_file_nonempty(p)

    summary = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "project_dir": PROJECT_DIR,
        "manifest_csv": MANIFEST_CSV,
        "batch_output_dir": BATCH_DIR,
        "n_fields_in_manifest_valid": int(len(manifest)),
        "n_fields_attempted": int(len(counts_df)),
        "n_fields_ok": int((counts_df["field_status"] == "ok").sum()),
        "n_fields_partial_or_failed": int((counts_df["field_status"] != "ok").sum()),
        "n_failures_logged": int(len(fail_df)),
        "model": {
            "model_type": MODEL_TYPE,
            "segmentation_mode": SEG_MODE,
            "checkpoint_path": CHECKPOINT_PATH,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "torch_cuda_available": bool(torch.cuda.is_available())
        },
        "approved_parameters": {
            "dapi_background_sigma_xy_px": 40.0,
            "dapi_background_alpha": DAPI_BG_ALPHA,
            "soma_denoise_sigma_zyx": list(SOMA_DENOISE_SIGMA_ZYX),
            "soma_background_sigma_xy_px": 40.0,
            "soma_background_alpha": SOMA_BG_ALPHA,
            "normalization_percentiles": [ROBUST_NORM_LOW_PCT, ROBUST_NORM_HIGH_PCT],
            "slice_equalization": "disabled",
            "nuclei_min_voxels": NUCLEI_MIN_VOXELS,
            "soma_voxel_range_inclusive": [SOMA_MIN_VOXELS, SOMA_MAX_VOXELS]
        },
        "provisional_calibration": {
            "xy_um": XY_UM_PROVISIONAL,
            "z_um_assumed": Z_UM_PROVISIONAL,
            "voxel_um3": VOXEL_UM3_PROVISIONAL,
            "note": "Primary size unit is voxel count. um^3 values are provisional."
        },
        "outputs": {
            "nuclei_combined_csv": nuclei_combined_csv,
            "soma_combined_csv": soma_combined_csv,
            "counts_csv": counts_csv,
            "failures_csv": failures_csv
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "Batch micro_sam 3D segmentation summary",
        f"Timestamp UTC: {summary['timestamp_utc']}",
        f"Valid fields in manifest: {summary['n_fields_in_manifest_valid']}",
        f"Fields attempted: {summary['n_fields_attempted']}",
        f"Fields OK: {summary['n_fields_ok']}",
        f"Fields partial/failed: {summary['n_fields_partial_or_failed']}",
        f"Failure records: {summary['n_failures_logged']}",
        f"Model: {MODEL_TYPE} ({SEG_MODE}) on {summary['model']['device']}",
        f"Nuclei combined CSV: {nuclei_combined_csv}",
        f"Soma combined CSV: {soma_combined_csv}",
        f"Counts CSV: {counts_csv}",
        f"Failures CSV: {failures_csv}",
    ]
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    _assert_file_nonempty(summary_json)
    _assert_file_nonempty(summary_txt)

    print("=== Batch run complete ===", flush=True)
    print(f"Fields attempted: {len(counts_df)}", flush=True)
    print(f"Fields OK: {(counts_df['field_status'] == 'ok').sum()}", flush=True)
    print(f"Fields partial/failed: {(counts_df['field_status'] != 'ok').sum()}", flush=True)
    print(f"Nuclei rows: {len(nuclei_combined)}", flush=True)
    print(f"Soma rows: {len(soma_combined)}", flush=True)
    print(f"Counts CSV: {counts_csv}", flush=True)
    print(f"Summary JSON: {summary_json}", flush=True)


if __name__ == "__main__":
    main()
