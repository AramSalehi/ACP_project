# %%
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.orientations import io_orientation, ornt2axcodes
from shapely.geometry import Polygon
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom
from skimage.draw import polygon  # For filling in contours on the mask
#%%
# ---------- Utility Functions ----------

def check_contour_in_roi(contour, roi_bounds):
    """
    Check if all points in a contour are inside the given ROI bounds.
    roi_bounds: (x_start, x_end, y_start, y_end, z_start, z_end)
    """
    x_start, x_end, y_start, y_end, z_start, z_end = roi_bounds
    is_inside = np.all(
        (contour[:, 0] >= x_start) & (contour[:, 0] < x_end) &
        (contour[:, 1] >= y_start) & (contour[:, 1] < y_end) &
        (contour[:, 2] >= z_start) & (contour[:, 2] < z_end)
    )
    return is_inside

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
    print("Original ROI shape:", roi_image.shape)
    return roi_image, refined_bounds

def calculate_contour_area(contour_slice):
    """
    Calculate the area of a 2D contour (given as an array of (x,y) points).
    """
    if len(contour_slice) < 3:
        return 0.0
    polygon_obj = Polygon(contour_slice)
    if not polygon_obj.is_valid:
        polygon_obj = polygon_obj.buffer(0)
    return polygon_obj.area

# ---------- Display Functions ----------

def show_roi_on_image(image_data, roi_bounds):
    """
    Display the ROI boundaries on the original image (without contours).
    """
    x_start, x_end, y_start, y_end, z_start, z_end = roi_bounds
    axial_slice = image_data[:, :, z_start + (z_end - z_start) // 2]
    coronal_slice = image_data[:, y_start + (y_end - y_start) // 2, :]
    sagittal_slice = image_data[x_start + (x_end - x_start) // 2, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Axial view
    axes[0].imshow(axial_slice, cmap="gray", origin="lower")
    rect_axial = Rectangle((y_start, x_start), y_end - y_start, x_end - x_start,
                             edgecolor='red', facecolor='none', linewidth=2)
    axes[0].add_patch(rect_axial)
    axes[0].set_title("Original Axial (ROI boundaries)")
    # Coronal view
    axes[1].imshow(coronal_slice, cmap="gray", origin="lower")
    rect_coronal = Rectangle((z_start, x_start), z_end - z_start, x_end - x_start,
                             edgecolor='red', facecolor='none', linewidth=2)
    axes[1].add_patch(rect_coronal)
    axes[1].set_title("Original Coronal (ROI boundaries)")
    # Sagittal view
    axes[2].imshow(sagittal_slice, cmap="gray", origin="lower")
    rect_sagittal = Rectangle((z_start, y_start), z_end - z_start, y_end - y_start,
                              edgecolor='red', facecolor='none', linewidth=2)
    axes[2].add_patch(rect_sagittal)
    axes[2].set_title("Original Sagittal (ROI boundaries)")
    plt.tight_layout()
    plt.show()

def show_roi_on_original_with_contours(image_data, roi_bounds, contours_voxel, tol=2):
    """
    Display original image slices (axial, coronal, sagittal) with ROI boundaries and overlaid contours.
    The slice indices are chosen based on the median contour coordinates that fall within the ROI.
    """
    x_start, x_end, y_start, y_end, z_start, z_end = roi_bounds

    # Axial view: determine slice using z-coordinates.
    all_axial_z = []
    for contour in contours_voxel:
        mask = ((contour[:, 0] >= x_start) & (contour[:, 0] < x_end) &
                (contour[:, 1] >= y_start) & (contour[:, 1] < y_end) &
                (contour[:, 2] >= z_start) & (contour[:, 2] < z_end))
        if np.any(mask):
            all_axial_z.append(contour[mask, 2])
    if len(all_axial_z) > 0:
        all_axial_z = np.concatenate(all_axial_z)
        axial_slice_index = int(np.round(np.median(all_axial_z)))
    else:
        axial_slice_index = z_start + (z_end - z_start) // 2

    # Coronal view: use y-coordinates.
    all_coronal_y = []
    for contour in contours_voxel:
        mask = ((contour[:, 0] >= x_start) & (contour[:, 0] < x_end) &
                (contour[:, 1] >= y_start) & (contour[:, 1] < y_end) &
                (contour[:, 2] >= z_start) & (contour[:, 2] < z_end))
        if np.any(mask):
            all_coronal_y.append(contour[mask, 1])
    if len(all_coronal_y) > 0:
        all_coronal_y = np.concatenate(all_coronal_y)
        coronal_slice_index = int(np.round(np.median(all_coronal_y)))
    else:
        coronal_slice_index = y_start + (y_end - y_start) // 2

    # Sagittal view: use x-coordinates.
    all_sagittal_x = []
    for contour in contours_voxel:
        mask = ((contour[:, 0] >= x_start) & (contour[:, 0] < x_end) &
                (contour[:, 1] >= y_start) & (contour[:, 1] < y_end) &
                (contour[:, 2] >= z_start) & (contour[:, 2] < z_end))
        if np.any(mask):
            all_sagittal_x.append(contour[mask, 0])
    if len(all_sagittal_x) > 0:
        all_sagittal_x = np.concatenate(all_sagittal_x)
        sagittal_slice_index = int(np.round(np.median(all_sagittal_x)))
    else:
        sagittal_slice_index = x_start + (x_end - x_start) // 2

    # Extract slices from the original image.
    axial_slice = image_data[:, :, axial_slice_index]
    coronal_slice = image_data[:, coronal_slice_index, :]
    sagittal_slice = image_data[sagittal_slice_index, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Axial view.
    axes[0].imshow(axial_slice, cmap='gray', origin='lower')
    rect_axial = Rectangle((y_start, x_start), y_end - y_start, x_end - x_start,
                             edgecolor='red', facecolor='none', linewidth=2)
    # axes[0].add_patch(rect_axial)
    # for contour in contours_voxel:
    #     mask = np.abs(contour[:, 2] - axial_slice_index) < tol
    #     if np.any(mask):
    #         axes[0].plot(contour[mask, 1], contour[mask, 0], color='red')
    # axes[0].set_title(f'Original Axial (slice {axial_slice_index})')
    
    # Coronal view.
    axes[1].imshow(coronal_slice, cmap='gray', origin='lower')
    rect_coronal = Rectangle((z_start, x_start), z_end - z_start, x_end - x_start,
                             edgecolor='red', facecolor='none', linewidth=2)
    # axes[1].add_patch(rect_coronal)
    # for contour in contours_voxel:
    #     mask = np.abs(contour[:, 1] - coronal_slice_index) < tol
    #     if np.any(mask):
    #         axes[1].plot(contour[mask, 2], contour[mask, 0], color='red')
    # axes[1].set_title(f'Original Coronal (slice {coronal_slice_index})')
    
    # Sagittal view.
    axes[2].imshow(sagittal_slice, origin='lower')
    rect_sagittal = Rectangle((z_start, y_start), z_end - z_start, y_end - y_start,
                              edgecolor='red', facecolor='none', linewidth=2)
    # axes[2].add_patch(rect_sagittal)
    # for contour in contours_voxel:
    #     mask = np.abs(contour[:, 0] - sagittal_slice_index) < tol
    #     if np.any(mask):
    #         axes[2].plot(contour[mask, 2], contour[mask, 1], color='red')
    # axes[2].set_title(f'Original Sagittal (slice {sagittal_slice_index})')
    
    plt.tight_layout()
    plt.show()

def show_resampled_roi_with_contours(resampled_roi, original_roi_bounds, contours_voxel, resample_factor=(1,4,4), tol=2):
    """
    Display the resampled ROI (after upsampling by resample_factor) with overlaid contours.
    The contours are adjusted (shifted and scaled) based on the original ROI bounds.
    For each view, the slice index is determined from the median coordinate of the adjusted contours.
    Additionally, the 2D area (in voxel²) of the contour(s) on each slice is calculated and annotated.
    """
    # Get ROI offset from the original image.
    x_start, _, y_start, _, z_start, _ = original_roi_bounds

    roi_contours = []
    for contour in contours_voxel:
        adjusted = contour.copy()
        # Shift relative to ROI origin then scale by resample_factor.
        adjusted[:, 0] -= x_start
        adjusted[:, 1] -= y_start
        adjusted[:, 2] -= z_start
        adjusted = adjusted * np.array(resample_factor)
        roi_contours.append(adjusted)

    # Determine slice indices based on adjusted contour coordinates.
    all_axial_z = np.concatenate([contour[:, 2] for contour in roi_contours if len(contour) > 0])
    if all_axial_z.size > 0:
        axial_slice_index = int(np.round(np.median(all_axial_z)))
    else:
        axial_slice_index = resampled_roi.shape[2] // 2

    all_coronal_y = np.concatenate([contour[:, 1] for contour in roi_contours if len(contour) > 0])
    if all_coronal_y.size > 0:
        coronal_slice_index = int(np.round(np.median(all_coronal_y)))
    else:
        coronal_slice_index = resampled_roi.shape[1] // 2

    all_sagittal_x = np.concatenate([contour[:, 0] for contour in roi_contours if len(contour) > 0])
    if all_sagittal_x.size > 0:
        sagittal_slice_index = int(np.round(np.median(all_sagittal_x)))
    else:
        sagittal_slice_index = resampled_roi.shape[0] // 2

    axial_slice   = resampled_roi[:, :, axial_slice_index]
    coronal_slice = resampled_roi[:, coronal_slice_index, :]
    sagittal_slice= resampled_roi[sagittal_slice_index, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # For each view, we will sum the areas of all contours that have points near the chosen slice.
    total_area_axial = 0.0
    total_area_coronal = 0.0
    total_area_sagittal = 0.0

    # Axial view: (rows=x, cols=y)
    axes[0].imshow(axial_slice, cmap='gray', origin='lower')
    for contour in roi_contours:
        mask_sel = np.abs(contour[:, 2] - axial_slice_index) < tol
        if np.any(mask_sel):
            pts = np.column_stack((contour[mask_sel, 1], contour[mask_sel, 0]))
            axes[0].plot(pts[:, 0], pts[:, 1], color="red")
            total_area_axial += calculate_contour_area(pts)
    axes[0].set_title(f"Resampled ROI - Axial (slice {axial_slice_index})")
    axes[0].text(5, axial_slice.shape[0]-10, f"Area: {total_area_axial:.2f} vox²", 
                 color='yellow', bbox=dict(facecolor='black', alpha=0.5))
    
    # Coronal view: (rows=x, cols=z)
    axes[1].imshow(coronal_slice, cmap='gray', origin='lower')
    for contour in roi_contours:
        mask_sel = np.abs(contour[:, 1] - coronal_slice_index) < tol
        if np.any(mask_sel):
            pts = np.column_stack((contour[mask_sel, 2], contour[mask_sel, 0]))
            axes[1].plot(pts[:, 0], pts[:, 1], color="red")
            total_area_coronal += calculate_contour_area(pts)
    axes[1].set_title(f"Resampled ROI - Coronal (slice {coronal_slice_index})")
    axes[1].text(5, coronal_slice.shape[0]-10, f"Area: {total_area_coronal:.2f} vox²", 
                 color='yellow', bbox=dict(facecolor='black', alpha=0.5))
    
    # Sagittal view: (rows=y, cols=z)
    axes[2].imshow(sagittal_slice, cmap='gray', origin='lower')
    for contour in roi_contours:
        mask_sel = np.abs(contour[:, 0] - sagittal_slice_index) < tol
        if np.any(mask_sel):
            pts = np.column_stack((contour[mask_sel, 2], contour[mask_sel, 1]))
            axes[2].plot(pts[:, 0], pts[:, 1], color="red")
            total_area_sagittal += calculate_contour_area(pts)
    axes[2].set_title(f"Resampled ROI - Sagittal (slice {sagittal_slice_index})")
    # axes[2].text(5, sagittal_slice.shape[0]-10, f"Area: {total_area_sagittal:.2f} vox²", 
    #              color='yellow', bbox=dict(facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
# ---------- New Functions for Creating and Displaying the Contour Mask ----------

def create_contour_mask(resampled_shape, original_roi_bounds, contours_voxel, resample_factor=(2,4,4), tol=2):
    """
    Create a binary mask (with the same shape as the resampled ROI) that marks the regions
    delineated by the contours. For each contour (after shifting and scaling),
    the function determines if it lies on an axial (z nearly constant), coronal (y nearly constant),
    or sagittal (x nearly constant) plane. It then fills in the polygon on that slice.
    """
    mask = np.zeros(resampled_shape, dtype=np.uint8)
    x_start, _, y_start, _, z_start, _ = original_roi_bounds

    for contour in contours_voxel:
        adjusted = contour.copy()
        # Shift relative to ROI origin then scale by resample_factor.
        adjusted[:, 0] -= x_start
        adjusted[:, 1] -= y_start
        adjusted[:, 2] -= z_start
        adjusted = adjusted * np.array(resample_factor)
        
        # Determine which coordinate is nearly constant.
        dx = np.ptp(adjusted[:, 0])
        dy = np.ptp(adjusted[:, 1])
        dz = np.ptp(adjusted[:, 2])
        
        if dz < tol:
            # Axial contour: constant z coordinate.
            slice_z = int(np.round(np.median(adjusted[:, 2])))
            # Use x and y coordinates for the polygon.
            rr, cc = polygon(adjusted[:, 0], adjusted[:, 1], shape=mask[:, :, slice_z].shape)
            mask[rr, cc, slice_z] = 1
        elif dy < tol:
            # Coronal contour: constant y coordinate.
            slice_y = int(np.round(np.median(adjusted[:, 1])))
            # Use x and z coordinates.
            rr, cc = polygon(adjusted[:, 0], adjusted[:, 2], shape=mask[:, slice_y, :].shape)
            mask[rr, slice_y, cc] = 1
        elif dx < tol:
            # Sagittal contour: constant x coordinate.
            slice_x = int(np.round(np.median(adjusted[:, 0])))
            # Use y and z coordinates.
            rr, cc = polygon(adjusted[:, 1], adjusted[:, 2], shape=mask[slice_x, :, :].shape)
            mask[slice_x, rr, cc] = 1
        else:
            print("Contour does not lie on a principal plane. Skipping mask creation for this contour.")
    
    return mask
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

def make_sagittal_rgb_mask(mask,sagittal_idx, tol=0):
    """
    Generate (and plot) an RGB mask of the central sagittal slice:
      - interior pixels in white
      - border pixels in blue
      - background in black

    Parameters
    ----------
    mask : 3D ndarray of bool or {0,1}
        Your binary mask volume.
    tol : int
        (ignored here, but you could use it if you want a thicker border)

    Returns
    -------
    rgb_mask : HxW×3 uint8 array
    """
    # pick the central sagittal slice
 
    sag = mask[17, :, :].astype(bool)

    # erode once to find interior
    struct = np.ones((3,3), dtype=bool)
    interior = binary_erosion(sag, structure=struct)
    border   = sag & ~interior

    # build RGB image
    h, w = sag.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_mask[interior] = [255, 255, 255]   # white inside
    rgb_mask[border]   = [0,   0,   255]   # blue border

    # plot
    plt.figure(figsize=(6,6))
    plt.imshow(rgb_mask)
    plt.axis('off')
    plt.title(f"Sagittal mask (slice {sagittal_idx})")
    plt.tight_layout()
    plt.show()

    return rgb_mask

def show_mask_on_resampled_roi(resampled_roi, mask):
    """
    Display the resampled ROI with the contour mask overlaid in axial, coronal, and sagittal views.
    """
    axial_index = resampled_roi.shape[2] // 2
    coronal_index = resampled_roi.shape[1] // 2
    sagittal_index = resampled_roi.shape[0] // 2+1
    
    axial_slice = resampled_roi[:, :, axial_index]
    coronal_slice = resampled_roi[:, coronal_index, :]
    sagittal_slice = resampled_roi[sagittal_index, :, :]
    rgb = make_sagittal_rgb_mask(mask,sagittal_index)
    axial_mask = mask[:, :, axial_index]
    coronal_mask = mask[:, coronal_index, :]
    sagittal_mask = mask[sagittal_index, :, :]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(axial_slice, cmap='gray', origin='lower')
    axes[0].imshow(axial_mask, cmap='jet', alpha=0.5, origin='lower')
    axes[0].set_title(f"Axial ROI with Contour Mask (slice {axial_index})")
    
    axes[1].imshow(coronal_slice, cmap='gray', origin='lower')
    axes[1].imshow(coronal_mask, cmap='jet', alpha=0.5, origin='lower')
    axes[1].set_title(f"Coronal ROI with Contour Mask (slice {coronal_index})")
    
    axes[2].imshow(sagittal_slice, cmap='gray', origin='lower')
    axes[2].imshow(sagittal_mask, cmap='jet', alpha=0.5, origin='lower')
    axes[2].set_title(f"Sagittal ROI with Contour Mask (slice {sagittal_index})")
    
    plt.tight_layout()
    plt.show()

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
    image_data, _ = load_nifti(image_path)
    label_data, _ = load_nifti(label_path)
    max_region_center = find_center_of_label(label_data, target_labels)
    print(f"Center for target labels found at: {max_region_center}")
    if max_region_center:
        cubic_window, roi_bounds = extract_custom_subroi(image_data, label_data, max_region_center, roi_sizes=(16, 16, 16))
        show_roi_on_image(image_data, roi_bounds)
    return roi_bounds, cubic_window

def read_cso_file_text(cso_file_path):
    """
    Read a text-based .cso file and extract contours.
    Only the "# NumPathPointLists" section is processed.
    """
    contours = []
    with open(cso_file_path, 'r') as f:
        lines = f.readlines()
    idx = 0
    num_lines = len(lines)
    while idx < num_lines:
        line = lines[idx].strip()
        if line == "# NumPathPointLists":
            idx += 1
            num_path_point_lists_line = lines[idx].strip()
            num_path_point_lists = int(num_path_point_lists_line.split()[1])
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
                        coord_line = lines[idx].strip()
                        coords_values.extend(coord_line.split())
                        idx += 1
                    coords_values = list(map(float, coords_values))
                    for i in range(0, len(coords_values), 3):
                        x, y, z = coords_values[i:i + 3]
                        # Adjust signs per your convention.
                        contour_points.append([-x, -y, z])
                idx += 2
            if contour_points:
                contour_points.append(contour_points[0])
                contours.append(np.array(contour_points))
                break
            idx += 1
        else:
            idx += 1
    return contours

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

# File paths (update these paths as necessary)
image_paths = r"raw_data\Nii"
cso_paths   = r"raw_data\data"
label_paths = r"raw_data\label"
#%%
from nibabel.processing import conform
# Process only the first couple of images for this example.
for image_name in os.listdir(image_paths):
    # if image_name==
    try:
        # Define file paths for the current image.
        cso_path = os.path.join(cso_paths, image_name.replace('T1.nii', 'MARKERS.cso'))
        image_path = os.path.join(image_paths, image_name)
        label_path = os.path.join(label_paths,image_name.replace('_T1.nii', '.nii.gz'))
        print("Processing:", image_name)
        
        # Load image and label data to extract affine and shape info.
        img = nib.load(image_path)
        img_data = img.get_fdata()
        label = nib.load(label_path)
        label_data = label.get_fdata()
        affine = img.affine
        inv_affine = np.linalg.inv(affine)
        
        # Get the orientation of the image.
        img_ornt = io_orientation(affine)
        img_axcodes = ornt2axcodes(img_ornt)
        axis_mapping = get_axis_mappings(img_axcodes)
        
        # Read the .cso file and process contours.
        contours = read_cso_file_text(cso_path)
       
        if not contours or len(contours) < 1:
            print("No contours found in the .cso file.")
        else:
            all_voxel_coords = []
            for contour in contours:
                # Convert world coordinates to voxel coordinates.
                world_coords = np.array(contour)
                voxel_coords = nib.affines.apply_affine(inv_affine, world_coords)
                voxel_coords_int = np.round(voxel_coords).astype(int)
                voxel_coords_int = np.clip(voxel_coords_int, [0, 0, 0], np.array(img_data.shape) - 1)
                all_voxel_coords.append(voxel_coords)
            
            # Process ROI based on target labels.
            target_labels = [14]
            roi_bounds, cubic_window = process_all_images_and_labels(target_labels)
            
            # Display original image with ROI boundaries and contours.
            show_roi_on_original_with_contours(img_data, roi_bounds, all_voxel_coords, tol=2)
            show_roi_on_original_with_contours(label_data,roi_bounds, all_voxel_coords, tol=2)
            # Resample the extracted ROI with different factors:
            # 1 in the x-dimension and 4 in the y and z dimensions.
            cubic_window_resampled = zoom(cubic_window, zoom=(2, 4, 4), order=1)
            print("Resampled ROI shape:", cubic_window_resampled.shape)
            # For the resampled ROI, the coordinate system now runs from 0 to the new shape.
            resampled_roi_bounds = (0, cubic_window_resampled.shape[0],
                                    0, cubic_window_resampled.shape[1],
                                    0, cubic_window_resampled.shape[2])
            
            # Display the resampled ROI with contours overlaid on the slice(s) that contain contour data,
            # and annotate the area inside the contour.
            show_resampled_roi_with_contours(cubic_window_resampled, roi_bounds, all_voxel_coords, resample_factor=(2,4,4), tol=2)
            
            # Create a binary mask of the contours on the upsampled ROI.
            contour_mask = create_contour_mask(cubic_window_resampled.shape, roi_bounds, all_voxel_coords, resample_factor=(2,4,4), tol=2)
            # Display the mask overlaid on the ROI.
            show_mask_on_resampled_roi(cubic_window_resampled, contour_mask)
            
            # -------------------------------
            # Save the upsampled ROI and contour mask for segmentation tasks.
            # Save the upsampled ROI with '_ROI_resampled.nii.gz' suffix.
            img_dir = r"DATA_MODEL\img"
            lbl_dir = r"DATA_MODEL\lbl"

            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            save_roi_path = os.path.join(
                img_dir,
                image_name.replace('.nii', '.nii.gz')
            )
            nib.save(nib.Nifti1Image(cubic_window_resampled, affine), save_roi_path)

            print("Saved upsampled ROI to:", save_roi_path)
            os.makedirs(os.path.dirname(save_roi_path), exist_ok=True)
            # Save the contour mask (binary mask) with '_contour_mask.nii.gz' suffix.
            save_mask_path = os.path.join(r"DATA_MODEL\lbl",
                                          image_name.replace('.nii', '.nii.gz'))
            nib.save(nib.Nifti1Image(contour_mask, affine), save_mask_path)
            print("Saved contour mask to:", save_mask_path)
            # -------------------------------
            
            # Optionally, check which contours fall outside the original ROI.
            false_roi = []
            for idx, contour in enumerate(all_voxel_coords):
                is_inside_roi = check_contour_in_roi(contour, roi_bounds)
                print(f"Contour {idx+1}: {'Inside' if is_inside_roi else 'Outside'} the ROI")
                if not is_inside_roi:
                    print(f"Contour {idx+1} has points outside the ROI.")
                    false_roi.append(image_name)
                    
    except Exception as e:
        print("Error processing image:", image_name)
        print(e)
    
#%%