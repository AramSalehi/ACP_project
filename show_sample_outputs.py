#%%

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.orientations import io_orientation, ornt2axcodes
from shapely.geometry import Polygon
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom
from matplotlib.widgets import Button
import os
import tkinter as tk
from tkinter import filedialog
import traceback
from skimage.draw import polygon  # For filling in contours on the mask



# ---------- Utility Functions ------

def load_nifti(file_path):
    """Load a NIfTI file and return its data and affine."""
    nii = nib.load(file_path)
    return nii.get_fdata(), nii.affine

def find_center_of_label(label_data, target_labels):
    """
    Find the approximate center of the region containing the target labels.
    """
    target_voxels = np.argwhere(np.isin(label_data, target_labels))
    if target_voxels.size == 0:
        return None
    center = np.mean(target_voxels, axis=0)
    return tuple(np.round(center).astype(int))

def extract_custom_subroi(image_data, label_data, center, roi_sizes=(48, 48, 48), exclude_labels=None):
    """
    Extract a non-cubic ROI from the 3D image based on the center.
    Custom offsets are applied to define the ROI boundaries.
    """
    size_x, size_y, size_z = roi_sizes
    x, y, z = center
    x_start = max(0, x - int(size_x / 2))
    x_end = min(image_data.shape[0], x + int(size_x / 2))
    y_start = max(0, y + int(0.5 * size_y / 2))
    y_end = min(image_data.shape[1], y + int(2.5 * size_y / 2))
    z_start = max(0, z - int(0.5 * size_z / 2))
    z_end = min(image_data.shape[2], z + int(1.5 * size_z / 2))
    roi_image = image_data[x_start:x_end, y_start:y_end, z_start:z_end]
    refined_bounds = (x_start, x_end, y_start, y_end, z_start, z_end)
    # print("Original ROI shape:", roi_image.shape)
    return roi_image, refined_bounds


import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETRSliceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin_unetr = SwinUNETR(
            img_size=(32, 64, 64),
            in_channels=1,
            out_channels=1,
            feature_size=48,  # Reduce feature size for a shallower network
            depths=(2, 2, 2,2),  # Reducing transformer depth at each stage
            num_heads=(2, 4, 8,8),  # Adjusting attention heads
            use_checkpoint=False,  # Simplify by turning off checkpointing
            dropout_path_rate=0.1,  # Adding dropout for regularization
        )

        # **NEW** Expand 1-channel output to 48 channels using 1x1x1 convolution
        self.channel_expansion = nn.Conv3d(1, 48, kernel_size=1)

        self.slice_predictor = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # Pool to (batch_size, 48, 1, 1, 1)
            nn.Flatten(),  # Flatten to (batch_size, 48)
            nn.Linear(48, 32)  # Fully connected layer for classification
        )

    def forward(self, x):
        seg_output = self.swin_unetr(x)  # Output is (batch_size, 1, D, H, W)
        
        # Expand feature dimensions from 1 to 48 channels
        expanded_features = self.channel_expansion(seg_output)  # Now (batch_size, 48, D, H, W)

        slice_pred = self.slice_predictor(expanded_features)  # Predict slice index
        
        return seg_output, slice_pred  # Return both segmentation output and slice prediction



# ---------- Other Processing Functions ----------

def process_all_images_and_labels(target_labels):
    """
    Process the current image and label files to:
      1. Find the center based on target labels.
      2. Extract the ROI.
      3. Display the ROI on the original image (using ROI boundaries only).
    Returns:
      roi_bounds and the extracted ROI (cubic_window).
    """
    img_data, label_data,_= load_and_resample_label(image_path, label_path)

    max_region_center = find_center_of_label(label_data, target_labels)
    # print(f"Center for target labels found at: {max_region_center}")
    if max_region_center:
        cubic_window, roi_bounds = extract_custom_subroi(img_data, label_data, max_region_center, roi_sizes=(16, 16, 16))
        # show_roi_on_image(img_data, roi_bounds,label_data)
    return roi_bounds, cubic_window


def get_axis_mappings(axcodes):
    axis_mapping = {}
    for idx, code in enumerate(axcodes):
        if code in ('R', 'L'):
            axis_mapping['R'] = idx
        elif code in ('A', 'P'):
            axis_mapping['A'] = idx
        elif code in ('S', 'I'):
            axis_mapping['S'] = idx
    return axis_mapping
# ---------- Main Processing Loop ----------


#%%
# File paths (update these paths as necessary)

image_paths = r"C:\Users\z724360\Desktop\acp_project_final_11-09-2026\Annotations\Nii"
# image_paths = r"C:\Users\z724360\Downloads\Hieab\Annotations\Nii"
# text_file = r"C:\Users\z724360\Downloads\Hieab\Annotations\data"

label_paths = r"C:\Users\z724360\Desktop\acp_project_final_11-09-2026\Annotations\label"
# label_paths = r"C:\Users\z724360\Downloads\Hieab\Annotations\label"
df_data = {"subject_id": [],"ca_area_pred": []}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUNETRSliceClassifier()
model.load_state_dict(torch.load("best_model_final.pth",map_location=torch.device(device)))
model.to(device)
#%%
nii_files = []
for root, _, files in os.walk(label_paths):
    for fname in files:
        if fname.endswith(".nii") or fname.endswith(".nii.gz"):
            nii_files.append(fname)
# %%
from skimage import measure
from matplotlib.path import Path
import nibabel as nib
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

low_circularity_subjects = []
nested_contour_subjects = []
df_rows = []

def compute_contour_features(contour):
    x = contour[:, 1]
    y = contour[:, 0]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    diffs = np.diff(contour, axis=0, append=[contour[0]])
    perimeter = np.linalg.norm(diffs, axis=1).sum()
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    return area/16, circularity

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
def _pad_to_shape(arr, target_h, target_w, pad_value=0):
    """Pad a 2D array to (target_h, target_w) centered; returns padded arr and (pad_top, pad_left)."""
    h, w = arr.shape
    dh = max(target_h - h, 0)
    dw = max(target_w - w, 0)
    pt = dh // 2
    pb = dh - pt
    pl = dw // 2
    pr = dw - pl
    arr_padded = np.pad(arr, ((pt, pb), (pl, pr)), mode='constant', constant_values=pad_value)
    return arr_padded, pt, pl
def show_results_editable(
    original_image_data,
    roi_bounds,
    zoom_factors,
    images,
    pred_mask,
    pred_slice_idx,
    closed_contours,
):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    from skimage import measure
    from matplotlib.patches import Rectangle
    import numpy as np          # <-- added
    import torch                # <-- added

    x_start, x_end, y_start, y_end, z_start, z_end = roi_bounds
    zf_x, zf_y, zf_z = zoom_factors
    n_contours = len(closed_contours)
    keep_mask = [True] * n_contours

    # Map predicted index (in resampled ROI) to original indices
    initial_slice_idx = x_start + int(pred_slice_idx / zf_x)   # sagittal (x)
    axial_idx = z_start + int(pred_slice_idx / zf_z)           # axial (z)

    area_choice = {"value": None}

    # ---- now 1x4 subplots (add axial as 4th) ----
    fig, axs = plt.subplots(1, 4, figsize=(20, 12))
    plt.subplots_adjust(right=0.85)

    def display_sagittal_slice(slice_idx, initial_slice_idx):
        axs[2].clear()

        # Expanded ROI with ±20 margin (in y and z for sagittal)
        y0 = max(0, y_start - 20)
        y1 = min(original_image_data.shape[1], y_end + 20)
        z0 = max(0, z_start - 20)
        z1 = min(original_image_data.shape[2], z_end + 20)
        
        # Crop sagittal slice
        sag_crop = original_image_data[slice_idx, y0:y1, z0:z1]
        H_roi, W_roi = sag_crop.shape

        # Rotate image (90° CW)
        sag_rot = np.rot90(sag_crop, 3)
        axs[2].imshow(sag_rot, cmap="gray", origin="lower", interpolation="nearest")
        axs[2].axis("off")
        axs[2].set_title(f"Sagittal (x={slice_idx})")

        # Draw original ROI box inside expanded crop
        orig_y0 = y_start - y0
        orig_y1 = y_end - y0
        orig_z0 = z_start - z0
        orig_z1 = z_end - z0

        rect_x = (H_roi - 1) - orig_y1
        rect_y = orig_z0
        rect_w = orig_y1 - orig_y0
        rect_h = orig_z1 - orig_z0

        roi_rect = Rectangle((rect_x, rect_y), rect_w, rect_h,
                             linewidth=2, edgecolor='yellow', facecolor='none')
        axs[2].add_patch(roi_rect)

        # Overlay contours if this is the predicted sagittal slice
        if slice_idx == initial_slice_idx:
            for idx, cnt in enumerate(closed_contours):
                if keep_mask[idx]:
                    y_local = cnt[:, 0] / zf_y + (y_start - y0)
                    z_local = cnt[:, 1] / zf_z + (z_start - z0)
                    x_plot = (H_roi - 1) - y_local
                    y_plot = z_local
                    axs[2].plot(x_plot, y_plot, lw=2, color="red")





    def display_axial_slice(axial_idx):
        axial_idx+=3
        y0 = max(0, y_start - 20)
        y1 = min(original_image_data.shape[1], y_end + 20)
        z0 = max(0, z_start - 20)
        z1 = min(original_image_data.shape[2], z_end + 20)
        x0 = max(0, x_start - 20)
        x1 = min(original_image_data.shape[0], x_end + 20)
        axs[3].clear()
        axial_img = original_image_data[x0:x1, y0:y1, axial_idx]
        axs[3].imshow(axial_img, cmap="gray", origin="lower")
        axs[3].set_title(f"Axial @ z={axial_idx}")
        axs[3].axis("off")

        if axial_idx < pred_mask.shape[0]:
            mask_slice = pred_mask[axial_idx].cpu().numpy()
            contours = measure.find_contours(mask_slice, 0.3)
            for contour in contours:
                axs[3].plot(contour[:, 1]-x0, contour[:, 0]-y0, color="red", lw=5)

        for idx, cnt in enumerate(closed_contours):
            if keep_mask[idx]:
                ys = y_start + cnt[:, 0] / zf_y  # image row
                zs = z_start + cnt[:, 1] / zf_z  # axial index
                # print("zs:", zs)
                xs = initial_slice_idx * np.ones_like(ys)  # fixed x for sagittal slice

                inside = np.isclose(zs, axial_idx, atol=0.3)
                if inside.any():
                    axs[3].plot(ys[inside]-y0, xs[inside]-x0, 'o', color='red', markersize=10)
        # inverted for images

        fig.canvas.draw_idle()
 


    # ---- your existing axial slice (model input) & overlay (left two panels) ----
    base_img = images[0, 0, pred_slice_idx].detach().cpu().numpy()
    pred_np = ((torch.sigmoid(seg_outputs[0, 0, pred_slice_idx]) > 0.5)
               .detach().cpu().numpy().astype(bool))
    H, W = base_img.shape
    pred_rgba = np.zeros((H, W, 4), dtype=np.float32)
    pred_rgba[pred_np] = [1.0, 0.0, 0.0, 0.6]

    base_img_v = np.flip(np.rot90(base_img, k=1), axis=1)
    pred_rgba_v = np.flip(np.rot90(pred_rgba, k=1), axis=1)

    axs[0].imshow(np.flip(np.rot90(images[0, 0, pred_slice_idx].cpu(), k=1), axis=1), cmap="gray")
    axs[0].axis("off")
    axs[0].set_title(f"Input (slice {pred_slice_idx})")

    axs[1].imshow(base_img_v, cmap='gray', zorder=1)
    axs[1].imshow(pred_rgba_v, zorder=2)
    axs[1].axis("off")
    axs[1].set_title("Overlay")

    # ---- third: sagittal (zoomed), fourth: axial (now zoomed) ----
    display_sagittal_slice(initial_slice_idx, initial_slice_idx)
    display_axial_slice(axial_idx)

    kept_contours = [c for k, c in zip(keep_mask, closed_contours) if k]
    kept_indices = [i for i, k in enumerate(keep_mask) if k]
    kept_slice_indices = [pred_slice_idx for _ in kept_indices]

    plt.tight_layout()
    plt.show()
    return kept_contours, kept_slice_indices, area_choice["value"]


import matplotlib.pyplot as plt

import SimpleITK as sitk
import numpy as np

def load_and_resample_label(image_path, label_path):
    """
    Load an image and a label NIfTI file, resample the label to image's space,
    and return both as numpy arrays plus the affine (from image).
    
    Args:
        image_path (str): Path to the reference image (e.g. T1).
        label_path (str): Path to the label image (e.g. segmentation).
        
    Returns:
        img_np: np.ndarray, image data
        lbl_np: np.ndarray, label data (aligned to image)
        affine: 4x4 np.ndarray, affine from image
    """
    # Read with SimpleITK for spatial alignment
    img = sitk.ReadImage(image_path)
    img=sitk.DICOMOrient(img, 'SAR')  # Ensure RAS orientation
    
    lbl = sitk.ReadImage(label_path)
    lbl=sitk.DICOMOrient(lbl, 'SAR')  # Ensure RAS orientation
    # Resample label onto image grid (nearest neighbor to preserve integers)
    lbl_resampled = sitk.Resample(
        lbl,
        img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        lbl.GetPixelID()

    )
    
    # Convert to numpy
    img_np = sitk.GetArrayFromImage(img)
    lbl_np = sitk.GetArrayFromImage(lbl_resampled)
    
    # Get affine (as numpy) from image
    origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())
    direction = np.array(img.GetDirection()).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = direction * spacing  # scale columns by spacing
    affine[:3, 3] = origin
    
    return img_np, lbl_np, affine

#%%
# Process only the first couple of images for this example.

df_rows = []
low_circularity_subjects = []
nested_contour_subjects = []

for lbl_name in nii_files:


    # if lbl_name.endswith('sub-0920.nii.gz'):
        print("Processing:", lbl_name)
        label_path = os.path.join(label_paths, lbl_name)
        # image_path = os.path.join(image_paths, lbl_name.replace('.nii.gz', '.nii.gz'))
        # image_path = os.path.join(image_paths, lbl_name.replace('.nii.gz', '_1_T1.nii.gz'))
        image_path = os.path.join(image_paths, lbl_name.replace('.nii.gz', '_T1.nii.gz'))

        img_data, label_data,_= load_and_resample_label(image_path, label_path)
        

        target_labels = [14]
        roi_bounds, cubic_window = process_all_images_and_labels(target_labels)
        cubic_window_resampled = zoom(cubic_window, zoom=(2, 4, 4), order=1).astype(np.float32)

        m = np.max(cubic_window_resampled)
        if m > 0:
            cubic_window_resampled /= m
        else:
            # all zeros; keep as-is (or handle however you prefer)
            pass

        with torch.no_grad():
            inp_img = torch.tensor(cubic_window_resampled).unsqueeze(0).unsqueeze(0).to(device).float()
            seg_outputs, slice_pred = model(inp_img)
            pred_slice_idx = slice_pred.argmax(dim=1)[0].item()
            
            pred_seg_pro = seg_outputs[0, 0, :, :, :]
            pred_seg = torch.sigmoid(pred_seg_pro)
            pred_seg_np = pred_seg.cpu().numpy()
            pred_slice_seg = pred_seg_np[pred_slice_idx] > 0.5

            contours = measure.find_contours(pred_slice_seg.astype(float), 0.5)
            closed_contours = [c for c in contours if np.allclose(c[0], c[-1], atol=1)]

            selected_contours,kept_slice_indices, choice  = show_results_editable(
                img_data,
                roi_bounds,
                zoom_factors=(2, 4, 4),
                images=inp_img,
                pred_mask=pred_seg,
                pred_slice_idx=pred_slice_idx,
               
                closed_contours=closed_contours,
                
            )
         
#%%
