# """
# Member 3 — Y-Model (VS Code Ready)
# ====================================
# Run this directly in VS Code.
# Your CSVs already have 'fruit' and 'ripeness' columns — no preprocessing needed.

# BEFORE RUNNING:
#   1. Set BASE_DIR below to where your CommonDataset folder actually lives.
#   2. pip install torch torchvision pandas numpy scikit-learn Pillow

# USAGE:
#   Full training (Multi-Task + Focal Loss):
#       python member3_train.py

#   Ablation — no Focal Loss (for ablation table row 2):
#       python member3_train.py --no_focal

#   Ripeness-focused weights (if fruit acc >> ripeness acc):
#       python member3_train.py --alpha 0.3 --beta 0.7
# """

# import argparse
# import os
# import time
# from pathlib import Path, PureWindowsPath

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from PIL import Image
# from sklearn.metrics import classification_report, f1_score
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models, transforms


# # ══════════════════════════════════════════════════════════════════════════════
# # ✏️  SET THIS TO YOUR DATASET ROOT FOLDER
# # ══════════════════════════════════════════════════════════════════════════════
# BASE_DIR = r"C:\Users\capaw\Downloads\Final_dataset"
# # Example if you moved it:
# # BASE_DIR = r"D:\MyData\CommonDataset"
# # BASE_DIR = r"C:\Users\YourName\Desktop\CommonDataset"

# # CSV paths (put them in the same folder as this script, or give full paths)
# TRAIN_CSV = "train.csv"
# VAL_CSV   = "val.csv"
# TEST_CSV  = "test.csv"

# # ══════════════════════════════════════════════════════════════════════════════
# # CLASS MAPPINGS
# # ══════════════════════════════════════════════════════════════════════════════
# FRUIT_CLASSES    = ["apple", "banana", "mango", "orange", "tomato"]
# RIPENESS_CLASSES = ["overripe", "ripe", "unripe"]

# FRUIT2IDX    = {f: i for i, f in enumerate(FRUIT_CLASSES)}
# RIPENESS2IDX = {r: i for i, r in enumerate(RIPENESS_CLASSES)}

# NUM_FRUIT    = len(FRUIT_CLASSES)       # 5
# NUM_RIPENESS = len(RIPENESS_CLASSES)    # 3


# # ══════════════════════════════════════════════════════════════════════════════
# # 1.  DATASET
# # ══════════════════════════════════════════════════════════════════════════════

# def resolve_path(raw_path: str, base_dir: str) -> str:
#     return raw_path.replace("CommonDataset", "Final_dataset")


# class FruitDataset(Dataset):
#     def __init__(self, csv_path: str, base_dir: str, transform=None):
#         self.df        = pd.read_csv(csv_path)
#         self.base_dir  = base_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row      = self.df.iloc[idx]
#         img_path = resolve_path(row["image_path"], self.base_dir)

#         try:
#             image = Image.open(img_path).convert("RGB")
#         except FileNotFoundError:
#             raise FileNotFoundError(
#                 f"\n❌  Image not found: {img_path}"
#                 f"\n   Check that BASE_DIR is correct: {self.base_dir}"
#             )

#         if self.transform:
#             image = self.transform(image)

#         fruit_idx    = FRUIT2IDX[row["fruit"]]
#         ripeness_idx = RIPENESS2IDX[row["ripeness"]]

#         return image, fruit_idx, ripeness_idx


# def get_transforms():
#     train_tf = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
#         transforms.RandomRotation(15),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225]),
#     ])
#     val_tf = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225]),
#     ])
#     return train_tf, val_tf


# # ══════════════════════════════════════════════════════════════════════════════
# # 2.  FOCAL LOSS
# # ══════════════════════════════════════════════════════════════════════════════

# class FocalLoss(nn.Module):
#     """
#     FL(p_t) = -(1 - p_t)^γ · log(p_t)
#     γ=0 → standard CrossEntropy. γ=2 → penalises easy/frequent classes.
#     """
#     def __init__(self, gamma=2.0, weight=None, reduction="mean"):
#         super().__init__()
#         self.gamma     = gamma
#         self.weight    = weight
#         self.reduction = reduction

#     def forward(self, logits, targets):
#         log_prob   = F.log_softmax(logits, dim=1)
#         ce_loss    = F.nll_loss(log_prob, targets,
#                                 weight=self.weight, reduction="none")
#         p_t        = torch.exp(-ce_loss)
#         focal_loss = ((1.0 - p_t) ** self.gamma) * ce_loss

#         if self.reduction == "mean":
#             return focal_loss.mean()
#         elif self.reduction == "sum":
#             return focal_loss.sum()
#         return focal_loss


# # ══════════════════════════════════════════════════════════════════════════════
# # 3.  Y-MODEL
# # ══════════════════════════════════════════════════════════════════════════════

# class YModel(nn.Module):
#     """
#     Shared ResNet50 backbone → two classification heads:
#       Branch A: Fruit     (5 classes)
#       Branch B: Ripeness  (3 classes)
#     """
#     def __init__(self, dropout=0.5, freeze_backbone=True):
#         super().__init__()

#         resnet         = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#         self.backbone  = nn.Sequential(*list(resnet.children())[:-2])
#         # Output: (N, 2048, 7, 7)

#         if freeze_backbone:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False
#             # Fine-tune only the last residual block
#             for p in list(self.backbone.parameters())[-20:]:
#                 p.requires_grad = True

#         self.gap = nn.AdaptiveAvgPool2d(1)   # → (N, 2048)

#         # Branch A — Fruit
#         self.fruit_head = nn.Sequential(
#             nn.Linear(2048, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(256, NUM_FRUIT),
#         )

#         # Branch B — Ripeness
#         self.ripeness_head = nn.Sequential(
#             nn.Linear(2048, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(256, NUM_RIPENESS),
#         )

#         self._init_weights()

#     def _init_weights(self):
#         for head in [self.fruit_head, self.ripeness_head]:
#             for layer in head:
#                 if isinstance(layer, nn.Linear):
#                     nn.init.kaiming_normal_(layer.weight)
#                     nn.init.zeros_(layer.bias)

#     def forward(self, x):
#         feat  = self.backbone(x)                        # (N, 2048, 7, 7)
#         feat  = self.gap(feat).view(x.size(0), -1)     # (N, 2048)
#         return self.fruit_head(feat), self.ripeness_head(feat)

#     def unfreeze_all(self):
#         for p in self.parameters():
#             p.requires_grad = True
#         print("🔓  Full backbone unfrozen for fine-tuning.")


# # ══════════════════════════════════════════════════════════════════════════════
# # 4.  COMBINED LOSS
# # ══════════════════════════════════════════════════════════════════════════════

# class MultiTaskLoss(nn.Module):
#     """L_total = α·L_fruit + β·L_ripeness"""
#     def __init__(self, alpha=0.5, beta=0.5,
#                  focal_gamma=2.0, ripe_weights=None):
#         super().__init__()
#         self.alpha          = alpha
#         self.beta           = beta
#         self.fruit_loss_fn  = nn.CrossEntropyLoss()
#         self.ripe_loss_fn   = FocalLoss(gamma=focal_gamma, weight=ripe_weights)

#     def forward(self, f_logits, r_logits, f_targets, r_targets):
#         Lf = self.fruit_loss_fn(f_logits, f_targets)
#         Lr = self.ripe_loss_fn(r_logits, r_targets)
#         return self.alpha * Lf + self.beta * Lr, Lf.detach(), Lr.detach()


# # ══════════════════════════════════════════════════════════════════════════════
# # 5.  METRICS
# # ══════════════════════════════════════════════════════════════════════════════

# def evaluate(model, loader, criterion, device):
#     model.eval()
#     total_loss = 0.0
#     all_ft, all_fp, all_rt, all_rp = [], [], [], []

#     with torch.no_grad():
#         for imgs, ft, rt in loader:
#             imgs = imgs.to(device)
#             ft   = ft.to(device)
#             rt   = rt.to(device)

#             fo, ro          = model(imgs)
#             loss, _, _      = criterion(fo, ro, ft, rt)
#             total_loss     += loss.item()

#             all_ft.extend(ft.cpu().numpy())
#             all_fp.extend(fo.argmax(1).cpu().numpy())
#             all_rt.extend(rt.cpu().numpy())
#             all_rp.extend(ro.argmax(1).cpu().numpy())

#     # Combined 15-class accuracy
#     combined_true = [f * 3 + r for f, r in zip(all_ft, all_rt)]
#     combined_pred = [f * 3 + r for f, r in zip(all_fp, all_rp)]
#     acc = np.mean(np.array(combined_true) == np.array(combined_pred))

#     # F1 for "Ripe" class — index 1 in RIPENESS_CLASSES = ["overripe","ripe","unripe"]
#     f1_ripe = f1_score(all_rt, all_rp, labels=[1], average="macro", zero_division=0)

#     return {
#         "loss":   total_loss / len(loader),
#         "acc":    acc,
#         "f1_ripe": f1_ripe,
#         "all_ft": all_ft, "all_fp": all_fp,
#         "all_rt": all_rt, "all_rp": all_rp,
#     }


# # ══════════════════════════════════════════════════════════════════════════════
# # 6.  MAIN
# # ══════════════════════════════════════════════════════════════════════════════

# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print(f"\n{'='*60}")
#     print(f"  Member 3 — Y-Model")
#     print(f"  Device    : {device}")
#     print(f"  Focal Loss: {'ON (γ=2.0)' if not args.no_focal else 'OFF (ablation)'}")
#     print(f"  α={args.alpha} (fruit)   β={args.beta} (ripeness)")
#     print(f"  BASE_DIR  : {BASE_DIR}")
#     print(f"{'='*60}\n")

#     # ── Data ──────────────────────────────────────────────────────────────────
#     train_tf, val_tf = get_transforms()

#     train_ds = FruitDataset(TRAIN_CSV, BASE_DIR, train_tf)
#     val_ds   = FruitDataset(VAL_CSV,   BASE_DIR, val_tf)
#     test_ds  = FruitDataset(TEST_CSV,  BASE_DIR, val_tf)

#     train_loader = DataLoader(train_ds, batch_size=args.batch_size,
#                               shuffle=True,  num_workers=2, pin_memory=True)
#     val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
#                               shuffle=False, num_workers=2, pin_memory=True)
#     test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
#                               shuffle=False, num_workers=2, pin_memory=True)

#     print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

#     # ── Ripeness class weights (inverse frequency) ────────────────────────────
#     train_df    = pd.read_csv(TRAIN_CSV)
#     counts      = train_df["ripeness"].map(RIPENESS2IDX).value_counts().sort_index()
#     ripe_w      = (1.0 / counts.values.astype(float))
#     ripe_w      = ripe_w / ripe_w.sum() * NUM_RIPENESS
#     ripe_tensor = torch.tensor(ripe_w, dtype=torch.float32).to(device)
#     print(f"Ripeness weights {RIPENESS_CLASSES}: {ripe_w.round(3)}\n")

#     # ── Model + Loss + Optimiser ──────────────────────────────────────────────
#     model     = YModel(freeze_backbone=True).to(device)
#     criterion = MultiTaskLoss(
#         alpha=args.alpha,
#         beta=args.beta,
#         focal_gamma=0.0 if args.no_focal else 2.0,
#         ripe_weights=ripe_tensor if not args.no_focal else None,
#     )
#     optimizer = optim.AdamW(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=args.lr, weight_decay=1e-4,
#     )
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

#     # ── Training loop ─────────────────────────────────────────────────────────
#     best_f1  = 0.0
#     best_acc = 0.0
#     history  = []

#     for epoch in range(1, args.epochs + 1):

#         # Unfreeze backbone at warm-up epoch
#         if epoch == args.unfreeze_epoch:
#             model.unfreeze_all()
#             optimizer = optim.AdamW(model.parameters(),
#                                     lr=args.lr * 0.1, weight_decay=1e-4)
#             scheduler = optim.lr_scheduler.CosineAnnealingLR(
#                 optimizer, T_max=args.epochs - epoch)

#         # ── Train one epoch ──
#         model.train()
#         train_loss = 0.0
#         t0 = time.time()
#         for imgs, ft, rt in train_loader:
#             imgs, ft, rt = imgs.to(device), ft.to(device), rt.to(device)
#             fo, ro       = model(imgs)
#             loss, _, _   = criterion(fo, ro, ft, rt)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#         scheduler.step()

#         # ── Validate ──
#         val_metrics = evaluate(model, val_loader, criterion, device)

#         elapsed = time.time() - t0
#         print(
#             f"Epoch [{epoch:02d}/{args.epochs}]  "
#             f"TrainLoss: {train_loss/len(train_loader):.4f}  "
#             f"ValLoss: {val_metrics['loss']:.4f}  "
#             f"ValAcc: {val_metrics['acc']*100:.1f}%  "
#             f"F1(Ripe): {val_metrics['f1_ripe']:.3f}  "
#             f"({elapsed:.1f}s)"
#         )

#         history.append({
#             "epoch":    epoch,
#             "val_acc":  val_metrics["acc"],
#             "f1_ripe":  val_metrics["f1_ripe"],
#             "val_loss": val_metrics["loss"],
#         })

#         # Save best checkpoint
#         if val_metrics["f1_ripe"] > best_f1:
#             best_f1  = val_metrics["f1_ripe"]
#             best_acc = val_metrics["acc"]
#             torch.save(model.state_dict(), "best_ymodel.pth")
#             print(f"  ✅  New best saved (F1_ripe={best_f1:.3f})")

#     # ── Final test evaluation ─────────────────────────────────────────────────
#     print(f"\n{'='*60}")
#     print(f"  BEST VAL → Acc: {best_acc*100:.2f}%  F1(Ripe): {best_f1:.4f}")
#     print(f"{'='*60}")

#     model.load_state_dict(torch.load("best_ymodel.pth", map_location=device))
#     test_metrics = evaluate(model, test_loader, criterion, device)

#     print(f"\n  TEST → Acc: {test_metrics['acc']*100:.2f}%  F1(Ripe): {test_metrics['f1_ripe']:.4f}")
#     print("\n── Ripeness Report (Test) ──────────────────────────────")
#     print(classification_report(test_metrics["all_rt"], test_metrics["all_rp"],
#                                 target_names=RIPENESS_CLASSES, zero_division=0))
#     print("\n── Fruit Report (Test) ─────────────────────────────────")
#     print(classification_report(test_metrics["all_ft"], test_metrics["all_fp"],
#                                 target_names=FRUIT_CLASSES, zero_division=0))

#     pd.DataFrame(history).to_csv("training_history.csv", index=False)
#     print("\n✅  Done. Files saved: best_ymodel.pth, training_history.csv")

#     # ── Ablation table row ────────────────────────────────────────────────────
#     mode = "Multi-Task + Focal Loss" if not args.no_focal else "Multi-Task (no Focal)"
#     print(f"\n── Ablation Table Row ──────────────────────────────────")
#     print(f"  {mode:<35} | Acc: {test_metrics['acc']*100:.1f}%"
#           f" | F1(Ripe): {test_metrics['f1_ripe']:.2f}")


# # ══════════════════════════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--epochs",         type=int,   default=30)
#     parser.add_argument("--batch_size",     type=int,   default=32)
#     parser.add_argument("--lr",             type=float, default=1e-4)
#     parser.add_argument("--alpha",          type=float, default=0.5)
#     parser.add_argument("--beta",           type=float, default=0.5)
#     parser.add_argument("--unfreeze_epoch", type=int,   default=10)
#     parser.add_argument("--no_focal",       action="store_true")
#     args = parser.parse_args()
#     main(args)


# ============================================================
# EVALUATION SCRIPT (RESNET Y-MODEL)
# ============================================================

# import torch
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from PIL import Image
# from sklearn.metrics import confusion_matrix, classification_report
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models, transforms
# import torch.nn as nn


# # ============================================================
# # PATHS
# # ============================================================
# BASE_DIR = r"C:\Users\capaw\Downloads\Final_dataset"
# TEST_CSV = "test.csv"


# # ============================================================
# # CLASS MAPPINGS
# # ============================================================
# FRUIT_CLASSES    = ["apple", "banana", "mango", "orange", "tomato"]
# RIPENESS_CLASSES = ["overripe", "ripe", "unripe"]

# FRUIT2IDX    = {f: i for i, f in enumerate(FRUIT_CLASSES)}
# RIPENESS2IDX = {r: i for i, r in enumerate(RIPENESS_CLASSES)}

# NUM_FRUIT    = len(FRUIT_CLASSES)
# NUM_RIPENESS = len(RIPENESS_CLASSES)


# # ============================================================
# # DATASET
# # ============================================================
# def resolve_path(raw_path: str, base_dir: str) -> str:
#     return raw_path.replace("CommonDataset", "Final_dataset")


# class FruitDataset(Dataset):
#     def __init__(self, csv_path: str, base_dir: str, transform=None):
#         self.df        = pd.read_csv(csv_path)
#         self.base_dir  = base_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row      = self.df.iloc[idx]
#         img_path = resolve_path(row["image_path"], self.base_dir)

#         image = Image.open(img_path).convert("RGB")

#         if self.transform:
#             image = self.transform(image)

#         fruit_idx    = FRUIT2IDX[row["fruit"]]
#         ripeness_idx = RIPENESS2IDX[row["ripeness"]]

#         return image, fruit_idx, ripeness_idx


# # ============================================================
# # TRANSFORMS (match validation)
# # ============================================================
# def get_transforms():
#     return transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225]),
#     ])


# # ============================================================
# # MODEL (RESNET50 VERSION)
# # ============================================================
# class YModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])

#         self.gap = nn.AdaptiveAvgPool2d(1)

#         self.fruit_head = nn.Sequential(
#             nn.Linear(2048, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, NUM_FRUIT),
#         )

#         self.ripeness_head = nn.Sequential(
#             nn.Linear(2048, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, NUM_RIPENESS),
#         )

#     def forward(self, x):
#         feat = self.backbone(x)
#         feat = self.gap(feat).view(x.size(0), -1)
#         return self.fruit_head(feat), self.ripeness_head(feat)


# # ============================================================
# # MAIN
# # ============================================================
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Data
#     val_tf = get_transforms()
#     test_ds = FruitDataset(TEST_CSV, BASE_DIR, val_tf)
#     test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

#     # Model
#     model = YModel().to(device)
#     model.load_state_dict(torch.load("best_ymodel.pth", map_location=device))
#     model.eval()

#     all_ft, all_fp = [], []
#     all_rt, all_rp = [], []

#     print("Running inference...")

#     with torch.no_grad():
#         for imgs, ft, rt in test_loader:
#             imgs = imgs.to(device)

#             fo, ro = model(imgs)

#             all_fp.extend(fo.argmax(1).cpu().numpy())
#             all_ft.extend(ft.numpy())

#             all_rp.extend(ro.argmax(1).cpu().numpy())
#             all_rt.extend(rt.numpy())

#     # ========================================================
#     # CONFUSION MATRIX
#     # ========================================================
#     def plot_cm(y_true, y_pred, class_names, title, cmap):
#         cm = confusion_matrix(y_true, y_pred)

#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
#                     xticklabels=class_names,
#                     yticklabels=class_names)

#         plt.title(title)
#         plt.xlabel("Predicted")
#         plt.ylabel("Actual")
#         plt.tight_layout()
#         plt.show()

#     plot_cm(all_ft, all_fp, FRUIT_CLASSES,
#             "Fruit Confusion Matrix", "Blues")

#     plot_cm(all_rt, all_rp, RIPENESS_CLASSES,
#             "Ripeness Confusion Matrix", "Oranges")

#     # ========================================================
#     # REPORTS
#     # ========================================================
#     print("\n=== Fruit Classification Report ===")
#     print(classification_report(all_ft, all_fp,
#                                 target_names=FRUIT_CLASSES))

#     print("\n=== Ripeness Classification Report ===")
#     print(classification_report(all_rt, all_rp,
#                                 target_names=RIPENESS_CLASSES))


# # ============================================================
# if __name__ == "__main__":
#     main() 



# ============================================================
# TRAINING HISTORY PLOT (NO TRAINING REQUIRED)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD DATA
# ============================================================
history_path = "training_history.csv"

try:
    history_df = pd.read_csv(history_path)
except FileNotFoundError:
    raise FileNotFoundError(
        f"\n❌ Could not find {history_path}"
        "\nMake sure training_history.csv is in the same folder."
    )


# ============================================================
# PLOT
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


# -------------------------
# Plot 1: Validation Loss
# -------------------------
ax1.plot(history_df['epoch'], history_df['val_loss'],
         label='Validation Loss', linewidth=2)

ax1.set_title('Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()


# -------------------------
# Plot 2: Accuracy + F1
# -------------------------
ax2.plot(history_df['epoch'], history_df['val_acc'],
         label='Validation Accuracy', linewidth=2)

ax2.plot(history_df['epoch'], history_df['f1_ripe'],
         label='F1 Score (Ripe)', linewidth=2, linestyle='--')

ax2.set_title('Model Performance')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Score')
ax2.legend()


# ============================================================
# SAVE + SHOW
# ============================================================
plt.tight_layout()

# Save for report (VERY useful)
plt.savefig("training_plot.png", dpi=300)

plt.show()

print("✅ Plot saved as training_plot.png")