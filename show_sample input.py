# roi_view_2x3_with_contours.py
#%%
# =========================
# CONFIG
# =========================
IMAGE_DIR       = r"C:\Users\z724360\Downloads\Hieab\Annotations\Nii"
LABEL_DIR       = r"C:\Users\z724360\Downloads\Hieab\Annotations\label"
CSO_DIR         = r"C:\Users\z724360\Downloads\Hieab\Annotations\data"   # folder with *.cso (optional)

TARGET_LABELS   = (14,)
ROI_SIZES       = (16, 16, 16)     # your asymmetric ROI function still uses these sizes
PAD_VOX         = (16, 16, 16)     # expanded crop pad (the second row of 2x3)
ROT_K           = 3                # 3 = -90° CW; 1 = +90° CCW; 2 = 180°; 0 = no rotation
TITLE_FONTSIZE  = 14               # ⬅️ global subplot title size

# Contour overlay defaults
DEFAULT_TOL                = 3     # slice hit tolerance (voxels)
DEFAULT_CONTOUR_COLOR      = "red"
DEFAULT_CONTOUR_LINEWIDTH  = 3
DEFAULT_SLICE_STRATEGY     = "contour"  # 'roi' or 'contour'
DEBUG_HITS_IN_TITLES       = True  # show how many contour points hit each panel

# =========================
# IMPORTS
# =========================
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================
# IO & BASIC UTILS
# =========================
def load_nifti(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata(), nii.affine

def _clip_pair(a, b, lim):
    a = max(0, int(np.floor(a)))
    b = min(lim, int(np.ceil(b)))
    if b <= a:
        b = min(lim, a + 1)
    return a, b

def _expand_bounds(bounds, pad_vox, shape3d):
    """Expand ROI bounds by pad_vox=(px,py,pz) and clip to image shape."""
    x0, x1, y0, y1, z0, z1 = bounds
    px, py, pz             = pad_vox
    X, Y, Z                = shape3d
    ex0, ex1 = _clip_pair(x0 - px, x1 + px, X)
    ey0, ey1 = _clip_pair(y0 - py, y1 + py, Y)
    ez0, ez1 = _clip_pair(z0 - pz, z1 + pz, Z)
    return (ex0, ex1, ey0, ey1, ez0, ez1)

def _center_from_bounds(bounds):
    x0, x1, y0, y1, z0, z1 = bounds
    return (int(round((x0 + x1) / 2)),
            int(round((y0 + y1) / 2)),
            int(round((z0 + z1) / 2)))

# =========================
# ROI PIPELINE (YOUR LOGIC)
# =========================
def find_center_of_label(label_data, target_labels):
    target_voxels = np.argwhere(np.isin(label_data, target_labels))
    if target_voxels.size == 0:
        return None
    center = np.mean(target_voxels, axis=0)
    return tuple(np.round(center).astype(int))

def extract_custom_subroi(image_data, label_data, center, roi_sizes=(48, 48, 48), exclude_labels=None):
    """
    Asymmetric logic (unchanged):
      x: symmetric
      y: biased positive (0.5/2.5)
      z: biased positive (0.5/1.5)
    """
    size_x, size_y, size_z = roi_sizes
    x, y, z = center

    x_start = max(0, x - int(size_x / 2))
    x_end   = min(image_data.shape[0], x + int(size_x / 2))

    y_start = max(0, y + int(0.5 * size_y / 2))
    y_end   = min(image_data.shape[1], y + int(2.5 * size_y / 2))

    z_start = max(0, z - int(0.5 * size_z / 2))
    z_end   = min(image_data.shape[2], z + int(1.5 * size_z / 2))

    roi_image = image_data[x_start:x_end, y_start:y_end, z_start:z_end]
    refined_bounds = (x_start, x_end, y_start, y_end, z_start, z_end)
    print("Original ROI shape:", roi_image.shape)
    return roi_image, refined_bounds

# =========================
# RECTANGLE MAPPERS (UNROTATED)
# =========================
def _rect_for_view(bounds, view):
    """
    Rectangle coords for matplotlib in (col, row, width, height) per view:
      axial    : image[:, :, z]       -> rows=x, cols=y
      coronal  : image[:, y, :]       -> rows=x, cols=z
      sagittal : image[x, :, :]       -> rows=y, cols=z
    """
    x0, x1, y0, y1, z0, z1 = bounds
    if view == 'axial':
        return (y0, x0, y1 - y0, x1 - x0)
    if view == 'coronal':
        return (z0, x0, z1 - z0, x1 - x0)
    if view == 'sagittal':
        return (z0, y0, z1 - z0, y1 - y0)
    raise ValueError("view must be 'axial' | 'coronal' | 'sagittal'.")

def _rect_for_zoom(bounds_roi, bounds_expanded, view):
    """
    Map the SAME ROI rectangle into the expanded (zoom) crop coordinates (UNROTATED).
    """
    x0, x1, y0, y1, z0, z1 = bounds_roi
    ex0, ex1, ey0, ey1, ez0, ez1 = bounds_expanded

    if view == 'axial':      # rows=x-ex0, cols=y-ey0
        return (y0 - ey0, x0 - ex0, y1 - y0, x1 - x0)
    if view == 'coronal':    # rows=x-ex0, cols=z-ez0
        return (z0 - ez0, x0 - ex0, z1 - z0, x1 - x0)
    if view == 'sagittal':   # rows=y-ey0, cols=z-ez0
        return (z0 - ez0, y0 - ey0, z1 - z0, y1 - y0)
    raise ValueError

# =========================
# ROTATION HELPERS
# =========================
def _rot90(img2d, k=ROT_K):
    """Rotate 2D array CCW by 90° k times (numpy semantics)."""
    return np.rot90(img2d, k=k)

def _rotate_rect_by_k(x, y, w, h, img_shape, k=ROT_K):
    """
    Rotate rectangle by k*90° around origin, matching np.rot90 for arrays.
    img_shape = (rows, cols) BEFORE rotation.
    """
    rows, cols = img_shape
    rc = np.array([[y, x], [y, x+w], [y+h, x], [y+h, x+w]], dtype=float)
    r, c = rc[:, 0], rc[:, 1]
    k = int(k) % 4
    if k == 0:
        r2, c2 = r, c
    elif k == 1:
        r2, c2 = cols - 1 - c, r
    elif k == 2:
        r2, c2 = rows - 1 - r, cols - 1 - c
    else:
        r2, c2 = c, rows - 1 - r
    return float(c2.min()), float(r2.min()), float(c2.max()-c2.min()), float(r2.max()-r2.min())

def _rotate_points_by_k(r, c, img_shape, k=ROT_K):
    """Rotate arrays of points (rows=r, cols=c) to match np.rot90."""
    rows, cols = img_shape
    k = int(k) % 4
    if k == 0:
        return r, c
    if k == 1:
        return cols - 1 - c, r
    if k == 2:
        return rows - 1 - r, cols - 1 - c
    return c, rows - 1 - r  # k==3

# =========================
# CROPPER
# =========================
def _crop_2d(image, bounds, view, fixed_idx):
    x0, x1, y0, y1, z0, z1 = bounds
    if view == 'axial':      # z fixed -> (rows=x, cols=y)
        return image[x0:x1, y0:y1, int(fixed_idx)]
    if view == 'coronal':    # y fixed -> (rows=x, cols=z)
        return image[x0:x1, int(fixed_idx), z0:z1]
    if view == 'sagittal':   # x fixed -> (rows=y, cols=z)
        return image[int(fixed_idx), y0:y1, z0:z1]
    raise ValueError

# =========================
# CONTOUR I/O (CSO) — OPTIONAL
# =========================
def read_cso_file_text(cso_file_path):
    """
    Read a text-based .cso file and extract contours.
    Only the "# NumPathPointLists" section is processed.
    Returns: list of (N,3) arrays in world coords (x,y,z).
    """
    if not os.path.isfile(cso_file_path):
        return []
    contours = []
    with open(cso_file_path, 'r') as f:
        lines = f.readlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line == "# NumPathPointLists":
            idx += 1
            num_path_point_lists = int(lines[idx].split()[1])
            idx += 1
            contour_points = []
            for _ in range(num_path_point_lists):
                u32_line = lines[idx].strip()
                if u32_line.startswith("U32_"):
                    parts = u32_line.split()
                    num_points = int(parts[1])
                    coords_values = parts[2:]
                    idx += 1
                    while len(coords_values) < num_points * 3:
                        coords_values.extend(lines[idx].split())
                        idx += 1
                    coords_values = list(map(float, coords_values))
                    for i in range(0, len(coords_values), 3):
                        x, y, z = coords_values[i:i+3]
                        # NOTE: Your earlier code flipped x,y signs. Keep or remove based on your data.
                        contour_points.append([-x, -y, z])
                idx += 2
            if contour_points:
                contour_points.append(contour_points[0])  # close polygon
                contours.append(np.array(contour_points, dtype=float))
                break
            idx += 1
        else:
            idx += 1
    return contours

def world_to_voxel(contours_world, inv_affine):
    """Convert list of world-space contours to voxel coords using inv_affine."""
    vox = []
    for c in contours_world:
        if c.size == 0:
            continue
        v = nib.affines.apply_affine(inv_affine, c)  # float voxel coords OK
        # Safety clamp (optional)
        vox.append(v)
    return vox

# =========================
# CONTOUR PLOTTING
# =========================
def _draw_contours_on_slice(ax,
                            contours_voxel,
                            view,
                            slice_idx,
                            img_shape_rc,
                            bounds_expanded=None,
                            tol=DEFAULT_TOL,
                            k=ROT_K,
                            color=DEFAULT_CONTOUR_COLOR,
                            lw=DEFAULT_CONTOUR_LINEWIDTH):
    """
    Draw contours on an axis for a view + slice. Returns number of points plotted ("hits").
    """
    if not contours_voxel:
        return 0

    # Per-view mapping
    if view == 'axial':      # rows=x, cols=y, fixed z
        def select_mask(pts): return np.abs(pts[:, 2] - slice_idx) < tol
        def r_before(pts):    return pts[:, 0]
        def c_before(pts):    return pts[:, 1]
        def apply_crop(r, c, ex0, ey0, ez0): return r - ex0, c - ey0
    elif view == 'coronal':  # rows=x, cols=z, fixed y
        def select_mask(pts): return np.abs(pts[:, 1] - slice_idx) < tol
        def r_before(pts):    return pts[:, 0]
        def c_before(pts):    return pts[:, 2]
        def apply_crop(r, c, ex0, ey0, ez0): return r - ex0, c - ez0
    elif view == 'sagittal': # rows=y, cols=z, fixed x
        def select_mask(pts): return np.abs(pts[:, 0] - slice_idx) < tol
        def r_before(pts):    return pts[:, 1]
        def c_before(pts):    return pts[:, 2]
        def apply_crop(r, c, ex0, ey0, ez0): return r - ey0, c - ez0
    else:
        raise ValueError("view must be 'axial' | 'coronal' | 'sagittal'.")

    rows, cols = img_shape_rc
    hits_total = 0

    for contour in contours_voxel:
        if contour.size == 0:
            continue
        pts = contour
        mask = select_mask(pts)
        if not np.any(mask):
            continue
        pts2d = pts[mask]
        hits_total += pts2d.shape[0]

        r = r_before(pts2d).astype(float)
        c = c_before(pts2d).astype(float)

        # Move into crop frame BEFORE rotation (zoom panels only)
        if bounds_expanded is not None:
            ex0, ex1, ey0, ey1, ez0, ez1 = bounds_expanded
            r, c = apply_crop(r, c, ex0, ey0, ez0)

        # Rotate to match displayed image
        r_rot, c_rot = _rotate_points_by_k(r, c, img_shape_rc, k=k)
        ax.plot(c_rot, r_rot, color=color, linewidth=lw)

    return hits_total

# =========================
# MAIN FIGURE
# =========================
def show_roi_grid_same_roi(image_data,
                           label_data,
                           target_labels=TARGET_LABELS,
                           roi_sizes=ROI_SIZES,
                           pad_vox=PAD_VOX,
                           figsize=(15, 10),
                           title_fontsize=TITLE_FONTSIZE,
                           # contours
                           contours_voxel=None,
                           tol=DEFAULT_TOL,
                           contour_color=DEFAULT_CONTOUR_COLOR,
                           contour_linewidth=DEFAULT_CONTOUR_LINEWIDTH,
                           slice_strategy=DEFAULT_SLICE_STRATEGY):
    """
    2×3 grid:
      Row 1: original axial/coronal/sagittal with ROI rectangle — rotated by ROT_K.
      Row 2: expanded crops with SAME ROI rectangle mapped — also rotated.
    If contours are provided (list of (N,3) voxel coords), overlays them on ALL six panels.
    """
    # ---- ROI selection ----
    center = find_center_of_label(label_data, target_labels)
    if center is None:
        raise RuntimeError("No voxels found for target labels; cannot locate ROI center.")
    _, roi_bounds = extract_custom_subroi(image_data, label_data, center, roi_sizes=roi_sizes)
    cx_roi, cy_roi, cz_roi = _center_from_bounds(roi_bounds)

    # Slice choice (ROI center vs contour median)
    if contours_voxel and slice_strategy == 'contour':
        try:
            all_x = np.concatenate([c[:, 0] for c in contours_voxel if c.size > 0])
            all_y = np.concatenate([c[:, 1] for c in contours_voxel if c.size > 0])
            all_z = np.concatenate([c[:, 2] for c in contours_voxel if c.size > 0])
            cx = int(np.round(np.median(all_x))) if all_x.size else cx_roi
            cy = int(np.round(np.median(all_y))) if all_y.size else cy_roi
            cz = int(np.round(np.median(all_z))) if all_z.size else cz_roi
        except Exception:
            cx, cy, cz = cx_roi, cy_roi, cz_roi
    else:
        cx, cy, cz = cx_roi, cy_roi, cz_roi

    # ---- Row 1: full slices (then rotated) ----
    axial_raw    = image_data[:, :, cz]   # rows=x, cols=y
    coronal_raw  = image_data[:, cy, :]   # rows=x, cols=z
    sagittal_raw = image_data[cx, :, :]   # rows=y, cols=z

    axial_rot    = _rot90(axial_raw,   ROT_K)
    coronal_rot  = _rot90(coronal_raw, ROT_K)
    sagittal_rot = _rot90(sagittal_raw,ROT_K)

    # Rectangles (UNROTATED) -> rotate to match displays
    ax_rect_r = _rotate_rect_by_k(*_rect_for_view(roi_bounds, 'axial'),    img_shape=axial_raw.shape,    k=ROT_K)
    co_rect_r = _rotate_rect_by_k(*_rect_for_view(roi_bounds, 'coronal'),  img_shape=coronal_raw.shape,  k=ROT_K)
    sa_rect_r = _rotate_rect_by_k(*_rect_for_view(roi_bounds, 'sagittal'), img_shape=sagittal_raw.shape, k=ROT_K)

    # ---- Row 2: expanded crops (then rotated) ----
    expanded = _expand_bounds(roi_bounds, pad_vox, image_data.shape)

    axial_zoom_raw    = _crop_2d(image_data, expanded, 'axial',    cz)  # (rows=x, cols=y)
    coronal_zoom_raw  = _crop_2d(image_data, expanded, 'coronal',  cy)  # (rows=x, cols=z)
    sagittal_zoom_raw = _crop_2d(image_data, expanded, 'sagittal', cx)  # (rows=y, cols=z)

    axial_zoom    = _rot90(axial_zoom_raw,   ROT_K)
    coronal_zoom  = _rot90(coronal_zoom_raw, ROT_K)
    sagittal_zoom = _rot90(sagittal_zoom_raw,ROT_K)

    ax_zoom_rect_r = _rotate_rect_by_k(*_rect_for_zoom(roi_bounds, expanded, 'axial'),    img_shape=axial_zoom_raw.shape,    k=ROT_K)
    co_zoom_rect_r = _rotate_rect_by_k(*_rect_for_zoom(roi_bounds, expanded, 'coronal'),  img_shape=coronal_zoom_raw.shape,  k=ROT_K)
    sa_zoom_rect_r = _rotate_rect_by_k(*_rect_for_zoom(roi_bounds, expanded, 'sagittal'), img_shape=sagittal_zoom_raw.shape, k=ROT_K)

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Helper for title with optional hit counts
    def t(label, hits):
        return f"{label} (hits: {hits})" if DEBUG_HITS_IN_TITLES else label

    # ---------- Row 1: originals ----------
    # Axial
    ax = axes[0, 0]
    ax.imshow(axial_rot, cmap='gray', origin='lower')
    rx, ry, rw, rh = ax_rect_r
    ax.add_patch(Rectangle((rx, ry), rw, rh, edgecolor='red', facecolor='none', linewidth=2))
    hits = _draw_contours_on_slice(ax, contours_voxel, 'axial', cz, axial_raw.shape,
                                   bounds_expanded=None, tol=tol, k=ROT_K,
                                   color=contour_color, lw=contour_linewidth)
    ax.set_title("Axial", fontsize=title_fontsize); ax.axis('off')

    # Coronal
    ax = axes[0, 1]
    ax.imshow(coronal_rot, cmap='gray', origin='lower')
    rx, ry, rw, rh = co_rect_r
    ax.add_patch(Rectangle((rx, ry), rw, rh, edgecolor='red', facecolor='none', linewidth=2))
    hits = _draw_contours_on_slice(ax, contours_voxel, 'coronal', cy, coronal_raw.shape,
                                   bounds_expanded=None, tol=tol, k=ROT_K,
                                   color=contour_color, lw=contour_linewidth)
    ax.set_title("Coronal", fontsize=title_fontsize); ax.axis('off')

    # Sagittal
    ax = axes[0, 2]
    ax.imshow(sagittal_rot, cmap='gray', origin='lower')
    rx, ry, rw, rh = sa_rect_r
    ax.add_patch(Rectangle((rx, ry), rw, rh, edgecolor='red', facecolor='none', linewidth=2))
    hits = _draw_contours_on_slice(ax, contours_voxel, 'sagittal', cx, sagittal_raw.shape,
                                   bounds_expanded=None, tol=tol, k=ROT_K,
                                   color=contour_color, lw=contour_linewidth)
    ax.set_title("Sagittal", fontsize=title_fontsize); ax.axis('off')

    # ---------- Row 2: zooms ----------
    # Axial zoom
    ax = axes[1, 0]
    ax.imshow(axial_zoom, cmap='gray', origin='lower')
    rx, ry, rw, rh = ax_zoom_rect_r
    ax.add_patch(Rectangle((rx, ry), rw, rh, edgecolor='red', facecolor='none', linewidth=2))
    hits = _draw_contours_on_slice(ax, contours_voxel, 'axial', cz, axial_zoom_raw.shape,
                                   bounds_expanded=expanded, tol=tol, k=ROT_K,
                                   color=contour_color, lw=contour_linewidth)
    ax.set_title("Axial", fontsize=title_fontsize); ax.axis('off')

    # Coronal zoom
    ax = axes[1, 1]
    ax.imshow(coronal_zoom, cmap='gray', origin='lower')
    rx, ry, rw, rh = co_zoom_rect_r
    ax.add_patch(Rectangle((rx, ry), rw, rh, edgecolor='red', facecolor='none', linewidth=2))
    hits = _draw_contours_on_slice(ax, contours_voxel, 'coronal', cy, coronal_zoom_raw.shape,
                                   bounds_expanded=expanded, tol=tol, k=ROT_K,
                                   color=contour_color, lw=contour_linewidth)
    ax.set_title("Coronal ", fontsize=title_fontsize); ax.axis('off')

    # Sagittal zoom
    ax = axes[1, 2]
    ax.imshow(sagittal_zoom, cmap='gray', origin='lower')
    rx, ry, rw, rh = sa_zoom_rect_r
    ax.add_patch(Rectangle((rx, ry), rw, rh, edgecolor='red', facecolor='none', linewidth=2))
    hits = _draw_contours_on_slice(ax, contours_voxel, 'sagittal', cx, sagittal_zoom_raw.shape,
                                   bounds_expanded=expanded, tol=tol, k=ROT_K,
                                   color=contour_color, lw=contour_linewidth)
    ax.set_title("Sagittal ", fontsize=title_fontsize); ax.axis('off')

    plt.tight_layout()
    plt.show()

    return roi_bounds, expanded

# =========================
# MAIN (batch example)
# =========================
if __name__ == "__main__":
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".nii", ".nii.gz"))]

    for image_name in files:
        try:
            print("\nProcessing:", image_name)
            image_path = os.path.join(IMAGE_DIR, image_name)

            # Guess label path (adjust if needed)
            label_guess = image_name.replace('_T1.nii', '.nii.gz')
            label_path  = os.path.join(LABEL_DIR, label_guess)
            if not os.path.isfile(label_path):
                base = os.path.splitext(os.path.splitext(image_name)[0])[0]
                alt  = os.path.join(LABEL_DIR, base + ".nii.gz")
                if os.path.isfile(alt):
                    label_path = alt
                else:
                    raise FileNotFoundError(f"Label not found for {image_name}")

            # Load data
            img_nii = nib.load(image_path)
            img     = img_nii.get_fdata()
            lbl     = nib.load(label_path).get_fdata()
            inv_aff = np.linalg.inv(img_nii.affine)

            # Optional contours (CSO -> world -> voxel). Skip if no CSO.
            cso_path = os.path.join(CSO_DIR, image_name.replace('T1.nii', 'MARKERS.cso'))
            contours_voxel = []
            if os.path.isfile(cso_path):
                contours_world = read_cso_file_text(cso_path)
                contours_voxel = world_to_voxel(contours_world, inv_affine=inv_aff)
                print(f"Loaded {len(contours_voxel)} contour(s) from CSO.")
            else:
                print("No CSO file found for this image; running without contours.")

            # Show grid
            roi_bounds, expanded = show_roi_grid_same_roi(
                image_data=img,
                label_data=lbl,
                target_labels=TARGET_LABELS,
                roi_sizes=ROI_SIZES,
                pad_vox=PAD_VOX,
                figsize=(15, 10),
                title_fontsize=TITLE_FONTSIZE,
                contours_voxel=contours_voxel if len(contours_voxel) > 0 else None,
                tol=DEFAULT_TOL,
                contour_color=DEFAULT_CONTOUR_COLOR,
                contour_linewidth=DEFAULT_CONTOUR_LINEWIDTH,
                slice_strategy=DEFAULT_SLICE_STRATEGY,  # 'roi' or 'contour'
            )

            print("ROI bounds:", roi_bounds, "Expanded:", expanded)

        except Exception as e:
            print("Error processing:", image_name)
            print(e)
        break
# %%    