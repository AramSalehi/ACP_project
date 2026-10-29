import os
import random
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import binary_opening, binary_closing
from scipy.stats import pearsonr
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, ToTensord,
    Rand3DElasticd, EnsureTyped, EnsureChannelFirstd,
    RandFlipd, RandAdjustContrastd, RandGaussianNoised, RandCropByPosNegLabeld, SpatialPadd
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.metrics import compute_dice

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    SpatialPadd(keys=["image", "label"], spatial_size=(32, 64, 64)),  # Padding to (96,96,96)
    Rand3DElasticd(keys=["image", "label"], sigma_range=(3, 5), magnitude_range=(2, 4), prob=0.5, mode=("bilinear", "nearest")),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    SpatialPadd(keys=["image", "label"], spatial_size=(32, 64, 64)),  # Padding to (96,96,96)
    ToTensord(keys=["image", "label"]),
])

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

class DiceLoss(nn.Module):
    def __init__(self, sigmoid=True, smooth=1e-6):
        super().__init__()
        self.sigmoid = sigmoid
        self.smooth = smooth

    def forward(self, pred, target):
        if self.sigmoid:
            pred = torch.sigmoid(pred)
        # flatten the tensors
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice_score

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)
        # self.bce_seg = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))  # not used in this example
        self.ce_slice = nn.CrossEntropyLoss()

    def forward(self, seg_outputs, slice_pred, labels):
        """
        Parameters:
          - seg_outputs: segmentation predictions, assumed to be of shape [B, C, D, H, W]
                         where D is the number of slices (e.g. 32)
          - slice_pred: slice classification logits of shape [B, D] (one logit per slice)
          - labels: segmentation ground truth with shape [B, 32, 64, 64] 
                    (i.e. one volume per sample, only one slice is nonzero)
        """

        B = labels.shape[0]
        
        # If labels has an extra channel (e.g. [B, 1, 32, 64, 64]), remove it:
        if labels.dim() == 5 and labels.shape[1] == 1:
            labels = labels.squeeze(1)  # now shape becomes [B, 32, 64, 64]

        # --- Determine the ground-truth slice index (gt_slice) for each sample ---
        # For each sample and for each of the 32 slices, sum the segmentation labels over the 64x64 spatial dimensions.
        # This gives a tensor of shape [B, 32]. We assume that the slice with the maximum sum (or any nonzero sum)
        # is the ground-truth slice.
        slice_sums = labels.view(B, labels.shape[1], -1).sum(dim=2)  # shape: [B, 32]
        gt_slice = slice_sums.argmax(dim=1)  # shape: [B] -- ground-truth slice indices
        # print(gt_slice)
        # --- Compute slice classification loss ---
        # slice_pred is assumed to be raw logits with shape [B, 32]
        clf_loss = self.ce_slice(slice_pred, gt_slice)

        # --- Compute segmentation (Dice) loss ---
        # First, determine which slice the network predicted (its highest logit) for each sample:
        pred_slice = slice_pred.argmax(dim=1)  # shape: [B]
        # print(pred_slice)
        seg_loss = 0.0
        for i in range(B):
            # seg_outputs[i] has shape [C, D, H, W].
            # We select the slice along the depth dimension:
            pred_seg = seg_outputs[i, :, pred_slice[i], :, :]  # shape: [C, H, W]

            # For the ground truth, select the corresponding (nonzero) slice.
            # labels[i] has shape [32, H, W]; add a channel dimension if needed.
            gt_seg = labels[i, gt_slice[i], :, :].unsqueeze(0)  # shape: [1, H, W]

            seg_loss += self.dice(pred_seg, gt_seg)
        seg_loss /= B

        # --- Total combined loss ---
        total_loss = seg_loss + clf_loss
        return total_loss

def train_unet(model, train_loader, val_loader, device, epochs=50, lr=1e-5, save_path="best_model.pth"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = CombinedLoss()  # your custom loss combining Dice and CE losses
    
    best_val_loss = float("inf")
    print("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # -------- Training Loop --------
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].float().to(device)
            # If labels come with a channel dimension [B, 1, D, H, W], squeeze it to [B, D, H, W]
            if labels.dim() == 5 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            
            optimizer.zero_grad()
            seg_outputs, slice_pred = model(images)
            loss = criterion(seg_outputs, slice_pred, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # -------- Validation Loop --------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", leave=False):
                images = batch["image"].to(device)
                labels = batch["label"].float().to(device)
                if labels.dim() == 5 and labels.shape[1] == 1:
                    labels = labels.squeeze(1)
                
                seg_outputs, slice_pred = model(images)
                loss = criterion(seg_outputs, slice_pred, labels)
                val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        print(f"Epoch [{epoch+1}/{epochs}]  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")
        
        # Save the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print("  Best model saved!")
        
        # -------- Visualization on one validation batch --------
        # (You can choose to display these only for the final epoch by wrapping this block in an if-statement.)
        with torch.no_grad():
            # Pick a random batch index from the validation set.
            random_batch_idx = random.randint(0, len(val_loader) - 1)
            for i, batch in enumerate(val_loader):
                if i == random_batch_idx:
                    images = batch["image"].to(device)
                    labels = batch["label"].float().to(device)
                    # If labels are in shape [B, 1, D, H, W], squeeze the channel dimension.
                    if labels.dim() == 5 and labels.shape[1] == 1:
                        labels = labels.squeeze(1)
                    
                    seg_outputs, slice_pred = model(images)
                    
                    # --- Determine the ground-truth slice index ---
                    # Reshape labels to [B, D, -1] and sum over spatial dims (H, W)
                    slice_sums = labels.view(labels.shape[0], labels.shape[1], -1).sum(dim=2)  # shape: [B, D]
                    gt_slice = slice_sums.argmax(dim=1)  # ground-truth slice index for each sample
                    pred_slice = slice_pred.argmax(dim=1)  # predicted slice index from the network
                    
                    # Since the batch size is one, we always use sample index 0.
                    true_slice_idx = gt_slice[0].item()
                    pred_slice_idx = pred_slice[0].item()
                    
                    # --- Extract segmentation masks ---
                    # seg_outputs is assumed to be of shape [B, C, D, H, W].
                    # For binary segmentation, we assume C == 1.
                    pred_seg = seg_outputs[0, 0, pred_slice_idx, :, :]
                    # Apply sigmoid then threshold for a binary mask.
                    pred_mask = (torch.sigmoid(pred_seg) > 0.5).float()
                    
                    # For ground truth, select the slice with the segmentation.
                    gt_seg = labels[0, true_slice_idx, :, :]
                    gt_mask = (gt_seg > 0.5).float()
                    
                    # --- Visualization ---
                    plt.figure(figsize=(15, 5))
                    
                    # 1. Input Image (using the ground-truth slice for display)
                    plt.subplot(1, 4, 1)
                    # Assuming images have shape [B, channels, D, H, W]; using channel 0.
                    plt.imshow(images[0, 0, true_slice_idx].cpu(), cmap='gray')
                    plt.title(f"Input Image\n(True Slice: {true_slice_idx})")
                    plt.axis('off')
                    
                    # 2. Ground-Truth Segmentation (Binary Mask)
                    plt.subplot(1, 4, 2)
                    plt.imshow(gt_mask.cpu(), cmap='gray')
                    plt.title(f"Ground Truth\nSlice: {true_slice_idx}\nPixels: {int(gt_mask.sum().item())}")
                    plt.axis('off')
                    
                    # 3. Predicted Segmentation (Binary Mask)
                    plt.subplot(1, 4, 3)
                    plt.imshow(pred_mask.cpu(), cmap='gray')
                    plt.title(f"Predicted\nSlice: {pred_slice_idx}\nPixels: {int(pred_mask.sum().item())}")
                    plt.axis('off')
                    
                    # 4. Overlay: Input Image + Predicted Segmentation
                    plt.subplot(1, 4, 4)
                    plt.imshow(images[0, 0, true_slice_idx].cpu(), cmap='gray')
                    plt.imshow(pred_mask.cpu(), cmap='jet', alpha=0.5)
                    plt.title("Overlay")
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    break

def deterministic_split(data_dicts, seed=42, train_fraction=0.80):
    """
    Deterministic subject split:
        80% train
        ~10% validation
        ~10% test

    IMPORTANT:
    Using this same function, the same seed, and the same sorted subject list
    in the dual-head, single-head, and 3D U-Net scripts guarantees identical
    train/validation/test subjects across all comparison models.
    """

    n = len(data_dicts)

    indices = np.arange(n)

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train = int(n * train_fraction)
    n_remaining = n - n_train

    n_test = n_remaining // 2
    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + n_test]
    val_idx = indices[n_train + n_test:]

    train_data = [data_dicts[i] for i in train_idx]
    val_data   = [data_dicts[i] for i in val_idx]
    test_data  = [data_dicts[i] for i in test_idx]

    assert len(train_data) + len(val_data) + len(test_data) == n

    return train_data, val_data, test_data

def get_data_loaders(img_dir, lbl_dir, batch_size=2):

    image_files = sorted(
        glob.glob(os.path.join(img_dir, "*.nii.gz"))
    )

    label_files = sorted(
        glob.glob(os.path.join(lbl_dir, "*.nii.gz"))
    )

    # -------------------------------------------------
    # Safer image/label matching by filename
    # -------------------------------------------------
    image_map = {
        os.path.basename(path): path
        for path in image_files
    }

    label_map = {
        os.path.basename(path): path
        for path in label_files
    }

    common_files = sorted(
        set(image_map.keys()) & set(label_map.keys())
    )

    missing_labels = sorted(
        set(image_map.keys()) - set(label_map.keys())
    )

    missing_images = sorted(
        set(label_map.keys()) - set(image_map.keys())
    )

    if missing_labels:
        raise RuntimeError(
            f"{len(missing_labels)} image(s) have no matching label. "
            f"Examples: {missing_labels[:5]}"
        )

    if missing_images:
        raise RuntimeError(
            f"{len(missing_images)} label(s) have no matching image. "
            f"Examples: {missing_images[:5]}"
        )

    if len(common_files) == 0:
        raise RuntimeError(
            "No matching image-label pairs found."
        )

    # -------------------------------------------------
    # Create subject dictionaries
    # -------------------------------------------------
    data_dicts = [
        {
            "image": image_map[name],
            "label": label_map[name],
            "subject_id": name.replace(".nii.gz", "")
        }
        for name in common_files
    ]

    # -------------------------------------------------
    # FIXED 80 / 10 / 10 SPLIT
    # -------------------------------------------------
    train_data, val_data, test_data = deterministic_split(
        data_dicts,
        seed=42,
        train_fraction=0.80
    )

    # -------------------------------------------------
    # MONAI datasets
    # -------------------------------------------------
    train_dataset = Dataset(
        data=train_data,
        transform=train_transforms
    )

    val_dataset = Dataset(
        data=val_data,
        transform=val_transforms
    )

    test_dataset = Dataset(
        data=test_data,
        transform=val_transforms
    )

    # -------------------------------------------------
    # Reproducible training order
    # -------------------------------------------------
    loader_generator = torch.Generator().manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # -------------------------------------------------
    # Print split sizes
    # -------------------------------------------------
    print(f"Total subjects: {len(data_dicts)}")
    print(f"Training:       {len(train_data)}")
    print(f"Validation:     {len(val_data)}")
    print(f"Test:           {len(test_data)}")

    # -------------------------------------------------
    # Save split manifest
    # -------------------------------------------------
    split_rows = []

    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data)
    ]:
        for item in split_data:
            split_rows.append({
                "subject_id": item["subject_id"],
                "split": split_name
            })

    split_df = pd.DataFrame(split_rows)

    split_df = split_df.sort_values(
        ["split", "subject_id"]
    )

    split_df.to_csv(
        "split_manifest.csv",
        index=False
    )

    print("Saved split manifest to: split_manifest.csv")

    return train_loader, val_loader, test_loader

MODEL_NAME = "dual_head_swin_unetr"

BEST_MODEL_PATH = "best_model.pth"

def dice_binary(pred, target, eps=1e-8):
    """
    Dice score for binary masks.
    """
    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum()
    denom = pred.sum() + target.sum()

    if denom.item() == 0:
        return 1.0

    return float(
        (2.0 * intersection + eps) /
        (denom + eps)
    )

def gt_target_slice(labels_3d):
    """
    Ground-truth target slice:
    the slice containing the largest GT foreground area.

    labels_3d shape: [D, H, W]
    """
    slice_sums = labels_3d.reshape(
        labels_3d.shape[0], -1
    ).sum(dim=1)

    return int(slice_sums.argmax().item())

def safe_pearson_r(x, y):
    """
    Pearson correlation with safety checks.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(
        pearsonr(x, y).statistic
    )

def evaluate_dual_head(
    model,
    loader,
    device,
    split_name
):
    """
    Evaluate the dual-head Swin-UNETR.

    IMPORTANT:
    The predicted slice is obtained from the explicit
    slice-classification head, preserving the original model idea.
    """

    model.eval()
    rows = []

    with torch.no_grad():

        for batch in tqdm(
            loader,
            desc=f"Evaluating {split_name}"
        ):

            images = batch["image"].float().to(device)
            labels = batch["label"].float().to(device)
            subject_ids = batch["subject_id"]

            # -------------------------------------------------
            # Dual-head forward pass
            # -------------------------------------------------
            seg_outputs, slice_pred = model(images)

            seg_probs = torch.sigmoid(seg_outputs)
            pred_binary = (
                seg_probs > 0.5
            ).float()

            batch_size = images.shape[0]

            for b in range(batch_size):

                # ---------------------------------------------
                # Ground truth
                # ---------------------------------------------
                gt3d = labels[b, 0]

                gt_slice = gt_target_slice(
                    gt3d
                )

                # ---------------------------------------------
                # Predicted slice
                # ORIGINAL DUAL-HEAD IDEA:
                # use the explicit classification head
                # ---------------------------------------------
                pred_slice = int(
                    slice_pred[b]
                    .argmax()
                    .item()
                )

                # ---------------------------------------------
                # 3D prediction
                # ---------------------------------------------
                pred3d = pred_binary[b, 0]

                gt3d_binary = (
                    gt3d > 0.5
                ).float()

                # ---------------------------------------------
                # Selected-slice masks
                # ---------------------------------------------
                gt_slice_mask = (
                    gt3d[gt_slice] > 0.5
                ).float()

                pred_slice_mask = (
                    pred3d[pred_slice]
                )

                # ---------------------------------------------
                # Dice metrics
                # ---------------------------------------------
                dice_3d = dice_binary(
                    pred3d,
                    gt3d_binary
                )

                dice_selected_slice = dice_binary(
                    pred_slice_mask,
                    gt_slice_mask
                )

                # ---------------------------------------------
                # Area
                # ---------------------------------------------
                gt_area_pixels = int(
                    gt_slice_mask
                    .sum()
                    .item()
                )

                pred_area_pixels = int(
                    pred_slice_mask
                    .sum()
                    .item()
                )

                # ---------------------------------------------
                # Localization
                # ---------------------------------------------
                slice_abs_error = abs(
                    pred_slice - gt_slice
                )

                slice_exact = int(
                    slice_abs_error == 0
                )

                slice_within_1 = int(
                    slice_abs_error <= 1
                )

                # ---------------------------------------------
                # Store subject-level results
                # ---------------------------------------------
                rows.append({
                    "subject_id":
                        subject_ids[b],

                    "split":
                        split_name,

                    "gt_slice":
                        gt_slice,

                    "pred_slice":
                        pred_slice,

                    "slice_abs_error":
                        slice_abs_error,

                    "slice_exact":
                        slice_exact,

                    "slice_within_1":
                        slice_within_1,

                    "dice_3d":
                        dice_3d,

                    "dice_selected_slice":
                        dice_selected_slice,

                    "gt_area_pixels":
                        gt_area_pixels,

                    "pred_area_pixels":
                        pred_area_pixels
                })

    df = pd.DataFrame(rows)

    output_csv = (
        f"{MODEL_NAME}_"
        f"{split_name}_results.csv"
    )

    df.to_csv(
        output_csv,
        index=False
    )

    print(
        f"Saved: {output_csv}"
    )

    return df

def summarize_results(
    df,
    split_name
):
    """
    Produce the same summary fields as the
    single-head Swin-UNETR and 3D U-Net scripts.
    """

    summary = {

        "model":
            MODEL_NAME,

        "split":
            split_name,

        "n":
            len(df),

        "mean_dice_3d":
            df["dice_3d"].mean(),

        "std_dice_3d":
            df["dice_3d"].std(ddof=1),

        "mean_dice_selected_slice":
            df["dice_selected_slice"].mean(),

        "std_dice_selected_slice":
            df["dice_selected_slice"].std(ddof=1),

        "area_pearson_r":
            safe_pearson_r(
                df["gt_area_pixels"],
                df["pred_area_pixels"]
            ),

        "slice_mae":
            df["slice_abs_error"].mean(),

        "slice_error_std":
            df["slice_abs_error"].std(ddof=1),

        "slice_exact_accuracy":
            df["slice_exact"].mean(),

        "slice_within_1_accuracy":
            df["slice_within_1"].mean()
    }

    return summary

data_dir = r"DATA_MODEL"
img_dir = os.path.join(data_dir, "img")
lbl_dir = os.path.join(data_dir, "lbl")
train_loader, val_loader, test_loader = get_data_loaders(img_dir, lbl_dir, batch_size=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUNETRSliceClassifier().to(device)

train_unet(model, train_loader, val_loader, device, epochs=100)

model = SwinUNETRSliceClassifier().to(device)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.to(device)

all_summaries = []
for split_name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
    df_split = evaluate_dual_head(model, loader, device, split_name)
    all_summaries.append(summarize_results(df_split, split_name))

summary_df = pd.DataFrame(all_summaries)
summary_output = f"{MODEL_NAME}_summary.csv"
summary_df.to_csv(summary_output, index=False)
print(f"\nSaved summary: {summary_output}")
print("\n================ DUAL-HEAD SUMMARY ================")
print(summary_df.to_string(index=False))
print("====================================================")

test_results_file = f"{MODEL_NAME}_test_results.csv"
test_df = pd.read_csv(test_results_file)
plt.figure(figsize=(8, 6))
plt.scatter(test_df["gt_area_pixels"], test_df["pred_area_pixels"], s=15)
test_r = safe_pearson_r(test_df["gt_area_pixels"], test_df["pred_area_pixels"])
plt.title(f"Dual-head Swin-UNETR\nTest-set area correlation (r = {test_r:.3f})")
plt.xlabel("Ground-truth AC area (pixels)")
plt.ylabel("Predicted AC area (pixels)")
plt.tight_layout()
plt.show()
