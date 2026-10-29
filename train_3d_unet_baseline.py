# %%
"""
Baseline experiment: standard 3D U-Net.

Purpose
-------
This script trains a conventional MONAI 3D U-Net on exactly the same inputs
and deterministic train/validation/test split as the Swin-UNETR experiments.

The model has no explicit slice-identification head. Localization is inferred
from the segmentation itself as the depth slice with the largest summed
foreground probability.

Outputs
-------
- split_manifest.csv
- best_3d_unet_baseline.pth
- unet3d_baseline_{train,val,test}_results.csv
- unet3d_baseline_summary.csv

Reported metrics
----------------
- 3D Dice
- selected-slice Dice
- Pearson r between GT and predicted AC area
- mean absolute error (MAE) between GT and predicted slice indices
- exact slice localization accuracy
- localization accuracy within +/- 1 slice
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import pearsonr
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityd,
    ToTensord,
    EnsureChannelFirstd,
    SpatialPadd,
    Rand3DElasticd
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet


# ============================================================
# CONFIG
# ============================================================
DATA_DIR = r"DATA_MODEL"
IMG_DIR = os.path.join(DATA_DIR, "img")
LBL_DIR = os.path.join(DATA_DIR, "lbl")

SEED = 42
BATCH_SIZE = 1
EPOCHS = 100
LR = 1e-5
TRAIN_FRACTION = 0.80
CHECKPOINT = "best_3d_unet_baseline.pth"
MODEL_NAME = "unet3d_baseline"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# TRANSFORMS
# ============================================================
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    SpatialPadd(keys=["image", "label"], spatial_size=(32, 64, 64)),
    Rand3DElasticd(keys=["image", "label"], sigma_range=(3, 5), magnitude_range=(2, 4), prob=0.5, mode=("bilinear", "nearest")),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    SpatialPadd(keys=["image", "label"], spatial_size=(32, 64, 64)),
    ToTensord(keys=["image", "label"]),
])


# ============================================================
# DATA SPLIT
# ============================================================
def build_data_dicts(img_dir, lbl_dir):
    image_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(lbl_dir, "*.nii.gz")))

    image_map = {os.path.basename(p): p for p in image_files}
    label_map = {os.path.basename(p): p for p in label_files}

    common = sorted(set(image_map) & set(label_map))

    missing_labels = sorted(set(image_map) - set(label_map))
    missing_images = sorted(set(label_map) - set(image_map))

    if missing_labels:
        raise RuntimeError(
            f"{len(missing_labels)} image(s) have no matching label. "
            f"First examples: {missing_labels[:5]}"
        )
    if missing_images:
        raise RuntimeError(
            f"{len(missing_images)} label(s) have no matching image. "
            f"First examples: {missing_images[:5]}"
        )
    if not common:
        raise RuntimeError("No matched image/label pairs were found.")

    return [
        {
            "image": image_map[name],
            "label": label_map[name],
            "subject_id": name.replace(".nii.gz", ""),
        }
        for name in common
    ]


def deterministic_split(data_dicts, seed=42, train_fraction=0.80):
    """
    Identical deterministic split to the Swin-UNETR ablation script.
    """
    n = len(data_dicts)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train = int(n * train_fraction)
    n_remaining = n - n_train

    n_test = n_remaining // 2
    n_val = n_remaining - n_test

    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + n_test]
    val_idx = indices[n_train + n_test:]

    assert len(train_idx) + len(val_idx) + len(test_idx) == n

    train_data = [data_dicts[i] for i in train_idx]
    val_data = [data_dicts[i] for i in val_idx]
    test_data = [data_dicts[i] for i in test_idx]

    manifest_rows = []
    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        for item in split_data:
            manifest_rows.append(
                {"subject_id": item["subject_id"], "split": split_name}
            )

    pd.DataFrame(manifest_rows).sort_values(
        ["split", "subject_id"]
    ).to_csv("split_manifest.csv", index=False)

    print(f"Total subjects: {n}")
    print(f"Training:       {len(train_data)}")
    print(f"Validation:     {len(val_data)}")
    print(f"Test:           {len(test_data)}")

    return train_data, val_data, test_data


def get_data_loaders(img_dir, lbl_dir, batch_size=1):
    data_dicts = build_data_dicts(img_dir, lbl_dir)
    train_data, val_data, test_data = deterministic_split(
        data_dicts,
        seed=SEED,
        train_fraction=TRAIN_FRACTION,
    )

    train_dataset = Dataset(train_data, transform=train_transforms)
    val_dataset = Dataset(val_data, transform=val_transforms)
    test_dataset = Dataset(test_data, transform=val_transforms)

    loader_generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================
# MODEL: STANDARD 3D U-NET
# ============================================================
class UNet3DBaseline(nn.Module):
    """
    Conventional 3D U-Net baseline implemented with MONAI.

    This intentionally contains no auxiliary localization/classification head.
    """

    def __init__(self):
        super().__init__()

        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="INSTANCE",
        )

    def forward(self, x):
        return self.unet(x)


# ============================================================
# LOSS
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = (probs * target).sum(dim=1)
        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probs.sum(dim=1) + target.sum(dim=1) + self.smooth
        )

        return 1.0 - dice.mean()


# ============================================================
# TRAINING
# ============================================================
def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=40,
    lr=1e-5,
    save_path=CHECKPOINT,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )
    criterion = DiceLoss()

    best_val_loss = float("inf")

    print(f"\nTraining {MODEL_NAME} on {device} ...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} - train",
            leave=False,
        ):
            images = batch["image"].float().to(device)
            labels = batch["label"].float().to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{epochs} - val",
                leave=False,
            ):
                images = batch["image"].float().to(device)
                labels = batch["label"].float().to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"train loss={avg_train_loss:.5f} | "
            f"val loss={avg_val_loss:.5f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model -> {save_path}")

        # ------------------------------------------------------------
        # VISUALIZE ONE RANDOM VALIDATION EXAMPLE AFTER EACH EPOCH
        # ------------------------------------------------------------
        with torch.no_grad():
            random_batch_idx = random.randint(0, len(val_loader) - 1)

            for batch_idx, batch in enumerate(val_loader):
                if batch_idx != random_batch_idx:
                    continue

                images = batch["image"].float().to(device)
                labels = batch["label"].float().to(device)

                logits = model(images)
                probs = torch.sigmoid(logits)

                # First subject in the selected validation batch.
                gt3d = labels[0, 0]       # [D,H,W]
                pred_prob3d = probs[0, 0] # [D,H,W]

                # Ground-truth slice = slice with the largest GT foreground area.
                gt_slice = int(
                    gt3d.reshape(gt3d.shape[0], -1)
                    .sum(dim=1)
                    .argmax()
                    .item()
                )

                # No slice classifier in these models:
                # infer the predicted slice from the segmentation probability map.
                pred_slice = int(
                    pred_prob3d.reshape(pred_prob3d.shape[0], -1)
                    .sum(dim=1)
                    .argmax()
                    .item()
                )

                gt_mask = (gt3d[gt_slice] > 0.5).float()
                pred_mask = (pred_prob3d[pred_slice] > 0.5).float()

                input_true = images[0, 0, gt_slice].detach().cpu()
                input_pred = images[0, 0, pred_slice].detach().cpu()
                gt_mask_cpu = gt_mask.detach().cpu()
                pred_mask_cpu = pred_mask.detach().cpu()

                fig = plt.figure(figsize=(15, 5))

                # 1. Input image at the GT slice
                plt.subplot(1, 4, 1)
                plt.imshow(input_true, cmap="gray")
                plt.title(f"Input Image\nGT Slice: {gt_slice}")
                plt.axis("off")

                # 2. GT mask
                plt.subplot(1, 4, 2)
                plt.imshow(gt_mask_cpu, cmap="gray")
                plt.title(
                    f"Ground Truth\nSlice: {gt_slice}\n"
                    f"Pixels: {int(gt_mask_cpu.sum().item())}"
                )
                plt.axis("off")

                # 3. Predicted mask at the model-selected slice
                plt.subplot(1, 4, 3)
                plt.imshow(pred_mask_cpu, cmap="gray")
                plt.title(
                    f"Predicted\nSlice: {pred_slice}\n"
                    f"Pixels: {int(pred_mask_cpu.sum().item())}"
                )
                plt.axis("off")

                # 4. Correct overlay: predicted mask on predicted MRI slice
                plt.subplot(1, 4, 4)
                plt.imshow(input_pred, cmap="gray")
                plt.imshow(pred_mask_cpu, cmap="jet", alpha=0.5)
                plt.title(
                    f"Overlay\nGT={gt_slice}, Pred={pred_slice}, "
                    f"|Δ|={abs(pred_slice - gt_slice)}"
                )
                plt.axis("off")

                plt.tight_layout()
                plt.show()
                plt.close(fig)
                break


# ============================================================
# METRICS
# ============================================================
def dice_binary(pred, target, eps=1e-8):
    pred = pred.float()
    target = target.float()

    intersection = (pred * target).sum()
    denom = pred.sum() + target.sum()

    if denom.item() == 0:
        return 1.0

    return float((2.0 * intersection + eps) / (denom + eps))


def gt_target_slice(labels_3d):
    return int(
        labels_3d.reshape(labels_3d.shape[0], -1)
        .sum(dim=1)
        .argmax()
    )


def predicted_slice_from_segmentation(prob_3d):
    slice_scores = prob_3d.reshape(prob_3d.shape[0], -1).sum(dim=1)
    return int(slice_scores.argmax())


def evaluate(model, loader, device, split_name):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
            images = batch["image"].float().to(device)
            labels = batch["label"].float().to(device)
            subject_ids = batch["subject_id"]

            logits = model(images)
            probs = torch.sigmoid(logits)
            pred_binary = (probs > 0.5).float()

            batch_size = images.shape[0]

            for b in range(batch_size):
                gt3d = labels[b, 0]
                pred_prob3d = probs[b, 0]
                pred3d = pred_binary[b, 0]

                gt_slice = gt_target_slice(gt3d)
                pred_slice = predicted_slice_from_segmentation(pred_prob3d)

                gt_slice_mask = (gt3d[gt_slice] > 0.5).float()
                pred_slice_mask = pred3d[pred_slice]

                dice_3d = dice_binary(pred3d, (gt3d > 0.5).float())
                dice_slice = dice_binary(pred_slice_mask, gt_slice_mask)

                gt_area_px = int(gt_slice_mask.sum().item())
                pred_area_px = int(pred_slice_mask.sum().item())

                slice_error = abs(pred_slice - gt_slice)

                rows.append({
                    "subject_id": subject_ids[b],
                    "split": split_name,
                    "gt_slice": gt_slice,
                    "pred_slice": pred_slice,
                    "slice_abs_error": slice_error,
                    "slice_exact": int(slice_error == 0),
                    "slice_within_1": int(slice_error <= 1),
                    "dice_3d": dice_3d,
                    "dice_selected_slice": dice_slice,
                    "gt_area_pixels": gt_area_px,
                    "pred_area_pixels": pred_area_px,
                })

    df = pd.DataFrame(rows)
    out_csv = f"{MODEL_NAME}_{split_name}_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    return df


def safe_pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(pearsonr(x, y).statistic)


def summarize(df, split_name):
    return {
        "model": MODEL_NAME,
        "split": split_name,
        "n": len(df),
        "mean_dice_3d": df["dice_3d"].mean(),
        "std_dice_3d": df["dice_3d"].std(ddof=1),
        "mean_dice_selected_slice": df["dice_selected_slice"].mean(),
        "std_dice_selected_slice": df["dice_selected_slice"].std(ddof=1),
        "area_pearson_r": safe_pearson_r(
            df["gt_area_pixels"],
            df["pred_area_pixels"],
        ),
        "slice_mae": df["slice_abs_error"].mean(),
        "slice_error_std": df["slice_abs_error"].std(ddof=1),
        "slice_exact_accuracy": df["slice_exact"].mean(),
        "slice_within_1_accuracy": df["slice_within_1"].mean(),
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders(
        IMG_DIR,
        LBL_DIR,
        batch_size=BATCH_SIZE,
    )

    model = UNet3DBaseline().to(DEVICE)

    train_model(
        model,
        train_loader,
        val_loader,
        DEVICE,
        epochs=EPOCHS,
        lr=LR,
        save_path=CHECKPOINT,
    )

    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE)
    )
    model.to(DEVICE)

    all_summaries = []

    for split_name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        df_split = evaluate(model, loader, DEVICE, split_name)
        all_summaries.append(summarize(df_split, split_name))

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(
        f"{MODEL_NAME}_summary.csv",
        index=False,
    )

    print("\n================ SUMMARY ================")
    print(summary_df.to_string(index=False))
    print("=========================================")
