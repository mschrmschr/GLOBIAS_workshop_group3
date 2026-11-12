"""Validation figures for the GloBIAS group-3 slides.
Reads the project's raw/preprocessed/label TIFFs (read-only) and writes PNGs to /out.
Run inside agenticj:gpu-local (tifffile, skimage, matplotlib available)."""
import json
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

P = "/proj"
OUT = "/out"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
DARK = "#111418"


def norm(img, lo=1, hi=99.8):
    a, b = np.percentile(img, [lo, hi])
    return np.clip((img.astype(np.float32) - a) / (b - a + 1e-9), 0, 1)


def overlay(gray, labels, color=(1.0, 0.85, 0.1), thick=False):
    rgb = np.dstack([gray] * 3)
    from skimage.morphology import binary_dilation, disk
    b = find_boundaries(labels, mode="inner")
    if thick: b = binary_dilation(b, disk(1))
    rgb[b] = color
    return rgb


def panel(ax, img, title, cmap="gray"):
    ax.imshow(img, cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=13, color="white", pad=6)
    ax.axis("off")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200, facecolor=DARK, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


raw_d = tifffile.imread(f"{P}/raw_images/F0003_DAPI.tif")
raw_s = tifffile.imread(f"{P}/raw_images/F0003_stain.tif")
pre_d = tifffile.imread(f"{P}/processed_images/batch/F0003/F0003_DAPI_preprocessed_for_micro_sam_3d_float32_0to1.tif")
pre_s = tifffile.imread(f"{P}/processed_images/batch/F0003/F0003_stain_preprocessed_for_micro_sam_3d_float32_0to1.tif")
lab_d = tifffile.imread(f"{P}/processed_images/batch/F0003/F0003_DAPI_micro_sam_3d_nuclei_labels_uint32.tif")
lab_s = tifffile.imread(f"{P}/processed_images/batch/F0003/F0003_stain_micro_sam_3d_soma_labels_uint32.tif")

# ---- 1. preprocessing: raw vs corrected, same plane, per-panel p1-p99.8 display
zs = [0, 42, 83]
fig, axes = plt.subplots(2, 4, figsize=(16, 8.4), facecolor=DARK)
panel(axes[0, 0], norm(raw_d[42]), "DAPI raw  z=42")
panel(axes[0, 1], norm(pre_d[42]), "DAPI corrected  z=42")
panel(axes[0, 2], norm(raw_d[83]), "DAPI raw  z=83 (deep, hazy)")
panel(axes[0, 3], norm(pre_d[83]), "DAPI corrected  z=83")
panel(axes[1, 0], norm(raw_s[26]), "Soma stain raw  z=26")
panel(axes[1, 1], norm(pre_s[26]), "Soma stain corrected  z=26")
panel(axes[1, 2], norm(raw_s[56]), "Soma stain raw  z=56 (low signal)")
panel(axes[1, 3], norm(pre_s[56]), "Soma stain corrected  z=56")
fig.subplots_adjust(wspace=0.04, hspace=0.12)
save(fig, "val_preprocessing_F0003.png")

# per-slice profile: raw vs corrected median and p99 (normalised) - shows haze removal
def prof(stack):
    s = stack.reshape(stack.shape[0], -1)
    idx = np.random.default_rng(0).choice(s.shape[1], 200000, replace=False)
    s = s[:, idx].astype(np.float32)
    return np.median(s, 1), np.percentile(s, 99.5, 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor="white")
for ax, raw, pre, name in [(axes[0], raw_d, pre_d, "DAPI"), (axes[1], raw_s, pre_s, "Soma stain")]:
    rm, rp = prof(raw); pm, pp = prof(pre)
    ax.plot(rp / rm, label="raw  (p99.5 / median)", color="#888888", lw=2)
    ax.plot(pp / np.maximum(pm, 1e-3), label="corrected  (p99.5 / median)", color="#0072B2", lw=2)
    ax.set_xlabel("z slice"); ax.set_ylabel("signal / background")
    ax.set_title(f"{name}: per-slice contrast, F0003"); ax.legend(frameon=False, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/val_contrast_profile_F0003.png", dpi=200)
plt.close(fig)

# ---- 2. segmentation: XY crop + XZ section with label boundaries
y0, x0, w = 350, 300, 350
zc = 42
yc = y0 + w // 2
zstretch = 3  # assumed Z = 1.0 um vs XY 0.3245 um -> ~3x
def xz(stack, y):
    return np.repeat(stack[:, y, x0:x0 + w], zstretch, axis=0)

fig, axes = plt.subplots(2, 3, figsize=(16, 9.2), facecolor=DARK,
                         gridspec_kw={"width_ratios": [1, 1, 1]})
g = norm(pre_d[zc, y0:y0 + w, x0:x0 + w])
panel(axes[0, 0], g, f"DAPI corrected  z={zc}")
panel(axes[0, 1], overlay(g, lab_d[zc, y0:y0 + w, x0:x0 + w]), "+ 3D nuclei labels (micro_sam)")
gx = norm(xz(pre_d, yc))
panel(axes[0, 2], overlay(gx, xz(lab_d, yc)), "XZ section (Z ×3, assumed spacing)")
n_on = [len(np.unique(lab_s[z])) for z in range(lab_s.shape[0])]
zs_best = int(np.argmax(n_on))
from scipy.ndimage import uniform_filter
dens = uniform_filter((lab_s[zs_best] > 0).astype(np.float32), 350)
cy, cx = np.unravel_index(np.argmax(dens[175:-175, 175:-175]), (650, 650))
sy, sx = cy, cx  # top-left of 350 crop
gs = norm(pre_s[zs_best, sy:sy + w, sx:sx + w])
panel(axes[1, 0], gs, f"Soma stain corrected  z={zs_best} (zoom)")
panel(axes[1, 1], overlay(gs, lab_s[zs_best, sy:sy + w, sx:sx + w], thick=True), "+ 3D soma labels (300-18,310 vox)")
ys = sy + w // 2
panel(axes[1, 2], overlay(norm(np.repeat(pre_s[:, ys, sx:sx + w], zstretch, 0)), np.repeat(lab_s[:, ys, sx:sx + w], zstretch, 0), thick=True),
      f"XZ section (Z x3, assumed spacing)")
fig.subplots_adjust(wspace=0.04, hspace=0.12)
save(fig, "val_segmentation_F0003.png")

# ---- 3. soma size filter before/after (pilot) at z=56 and z=77
before = tifffile.imread(f"{P}/processed_images/pilot/F0003/F0003_stain_micro_sam_3d_soma_labels_uint32.tif")
after = tifffile.imread(f"{P}/processed_images/pilot/F0003/F0003_stain_micro_sam_3d_soma_labels_uint32_revised_sizefilter.tif")
pre_p = tifffile.imread(f"{P}/processed_images/pilot/F0003/F0003_stain_preprocessed_for_micro_sam_3d_float32_0to1.tif")
fig, axes = plt.subplots(1, 2, figsize=(14, 7.4), facecolor=DARK)
for ax, z in zip(axes, [56, 77]):
    gimg = norm(pre_p[z])
    rgb = np.dstack([gimg] * 3)
    removed = (before[z] > 0) & (after[z] == 0)
    from skimage.morphology import binary_dilation, disk
    rgb[binary_dilation(find_boundaries(np.where(removed, before[z], 0), mode="inner"), disk(2))] = (1.0, 0.2, 0.2)
    rgb[binary_dilation(find_boundaries(after[z], mode="inner"), disk(2))] = (1.0, 0.85, 0.1)
    panel(ax, rgb, f"z={z}:  red = removed (>18,310 vox)   yellow = kept")
fig.subplots_adjust(wspace=0.04)
save(fig, "val_soma_sizefilter_F0003.png")

print(json.dumps({"soma_best_z": zs_best, "dapi_crop": [y0, x0, w]}))
