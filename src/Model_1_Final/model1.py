# ============================================================
# MEMBER 1 — THE BASELINE SPECIALIST  (v2 — High-Accuracy)
# Transfer Learning via Bottleneck Feature Extraction
# Models : VGG16 | ResNet50 | MobileNetV2
# Dataset: 5 fruits × 3 ripeness = 15 classes
# Optimised for Kaggle (P100/T4 GPU)
#
# Split source: pre-defined train.xls / test.xls / val.xls CSVs
# ============================================================

# ── 0. MIXED PRECISION — must be set BEFORE any Keras imports ──
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications import 
    vgg16        as vgg16_mod,
    resnet50     as resnet50_mod,
    mobilenet_v2 as mobilenetv2_mod,
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

tf.keras.mixed_precision.set_global_policy("mixed_float16")

print("TensorFlow :", tf.__version__)
print("GPU        :", tf.config.list_physical_devices("GPU"))
print("Precision  :", tf.keras.mixed_precision.global_policy().name)


# ── 1. PATHS & DIRECTORIES ────────────────────────────────────────────────
# 1. Update this to the Kaggle folder that contains 'base', 'data_scrapped', and 'tomatos'
IMAGE_DATASET_ROOT = "/kaggle/input/datasets/shivamm08/final-dataset1/Final_dataset/Final_dataset"

# 2. Update these to the exact Kaggle paths for your uploaded CSVs
TRAIN_CSV_PATH = "/kaggle/input/datasets/shivamm08/final-dataset1/train.csv"
VAL_CSV_PATH   = "/kaggle/input/datasets/shivamm08/final-dataset1/val.csv"
TEST_CSV_PATH  = "/kaggle/input/datasets/shivamm08/final-dataset1/test.csv"

# 3. Cleaner output directory names
WORK_DIR    = "/kaggle/working/Fruit_Classifier_Results"
FEAT_DIR    = os.path.join(WORK_DIR, "bottleneck_features")
RESULTS_DIR = os.path.join(WORK_DIR, "training_metrics")

for d in [FEAT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 2. LOAD & REMAP SPLITS FROM CSV ─────────────────────────
def remap_path(windows_path: str, kaggle_root: str) -> str:
    """Extracts the relative path after 'CommonDataset/' and appends to Kaggle root."""
    p = str(windows_path).replace("\\", "/")
    marker = "CommonDataset/"
    idx = p.find(marker)
    
    if idx == -1:
        # Fallback just in case a path is weird
        return os.path.join(kaggle_root, os.path.basename(p))
        
    relative = p[idx + len(marker):]
    return os.path.join(kaggle_root, relative)

print("\n[STEP 1] Loading pre-defined splits from CSV files …")

train_df = pd.read_csv(TRAIN_CSV_PATH)
val_df   = pd.read_csv(VAL_CSV_PATH)
test_df  = pd.read_csv(TEST_CSV_PATH)

for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
    # Assuming the first column containing the path is named 'image_path'
    # If your CSV has no header, you might need to use df.iloc[:, 0] instead
    df["kaggle_path"] = df["image_path"].apply(
        lambda p: remap_path(p, IMAGE_DATASET_ROOT)
    )
    missing = (~df["kaggle_path"].apply(os.path.exists)).sum()
    print(f"  {name:5s}: {len(df):>5} rows  |  missing files: {missing}")

# Build sorted class list from training labels
ALL_CLASSES  = sorted(train_df["label"].unique())
NUM_CLASSES  = len(ALL_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}

print(f"\n  Classes ({NUM_CLASSES}): {ALL_CLASSES}")

# ── 3. CONFIGURATION ────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 256
EPOCHS_HEAD  = 60
DROPOUT_1    = 0.5
DROPOUT_2    = 0.3
DENSE_UNITS  = [512, 256]
L2_REG       = 1e-4
LABEL_SMOOTH = 0.10
WARMUP_FRAC  = 0.10

# ── 4. MODEL CONFIGURATIONS ─────────────────────────────────
MODEL_CONFIGS = {
    "VGG16": {
        "base_fn":       VGG16,
        "preprocess_fn": vgg16_mod.preprocess_input,
        "input_shape":   (224, 224, 3),
    },
    "ResNet50": {
        "base_fn":       ResNet50,
        "preprocess_fn": resnet50_mod.preprocess_input,
        "input_shape":   (224, 224, 3),
    },
    "MobileNetV2": {
        "base_fn":       MobileNetV2,
        "preprocess_fn": mobilenetv2_mod.preprocess_input,
        "input_shape":   (224, 224, 3),
    },
}

# ── 5. DUAL-POOL FEATURE EXTRACTOR ──────────────────────────
def build_feature_extractor(model_name: str):
    cfg  = MODEL_CONFIGS[model_name]
    base = cfg["base_fn"](
        weights="imagenet",
        include_top=False,
        input_shape=cfg["input_shape"],
    )
    base.trainable = False

    inp = tf.keras.Input(shape=cfg["input_shape"])
    x   = base(inp, training=False)
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    out = layers.Concatenate()([gap, gmp])

    return models.Model(inp, out, name=f"{model_name}_extractor")


# ── 6. tf.data PIPELINE (driven by DataFrame) ───────────────
AUTO = tf.data.AUTOTUNE

def build_tf_dataset_from_df(df: pd.DataFrame, preprocess_fn):
    """Build a tf.data pipeline from a split DataFrame.
    Rows whose kaggle_path does not exist on disk are silently skipped.
    Returns (dataset, label_array).
    """
    mask   = df["kaggle_path"].apply(os.path.exists)
    df_ok  = df[mask].copy()
    if (~mask).any():
        print(f"    ⚠  Skipped {(~mask).sum()} missing files")

    paths  = df_ok["kaggle_path"].values
    labels = df_ok["label"].map(CLASS_TO_IDX).values.astype(np.int32)

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32)
        img = preprocess_fn(img)
        return img, label

    ds = (
        tf.data.Dataset.from_tensor_slices((paths, labels))
        .map(load_and_preprocess, num_parallel_calls=AUTO)
        .batch(64)
        .prefetch(AUTO)
    )
    return ds, labels


# ── 7. FEATURE EXTRACTION (with caching) ────────────────────
def extract_bottleneck_features(model_name: str, force_recompute=False):
    feat_train  = os.path.join(FEAT_DIR, f"{model_name}_X_train.npy")
    label_train = os.path.join(FEAT_DIR, f"{model_name}_y_train.npy")
    feat_val    = os.path.join(FEAT_DIR, f"{model_name}_X_val.npy")
    label_val   = os.path.join(FEAT_DIR, f"{model_name}_y_val.npy")
    feat_test   = os.path.join(FEAT_DIR, f"{model_name}_X_test.npy")
    label_test  = os.path.join(FEAT_DIR, f"{model_name}_y_test.npy")
    classes_f   = os.path.join(FEAT_DIR, f"{model_name}_classes.npy")

    cache_files = [feat_train, label_train, feat_val, label_val,
                   feat_test,  label_test,  classes_f]

    if not force_recompute and all(os.path.exists(p) for p in cache_files):
        print(f"  [{model_name}] Loading cached features …")
        return (
            np.load(feat_train), np.load(label_train),
            np.load(feat_val),   np.load(label_val),
            np.load(feat_test),  np.load(label_test),
            list(np.load(classes_f, allow_pickle=True)),
        )

    print(f"\n  [{model_name}] Building dual-pool extractor …")
    extractor     = build_feature_extractor(model_name)
    preprocess_fn = MODEL_CONFIGS[model_name]["preprocess_fn"]

    print(f"  [{model_name}] Extracting TRAIN features …")
    train_ds, y_tr = build_tf_dataset_from_df(train_df, preprocess_fn)
    X_tr = extractor.predict(train_ds, verbose=1)

    print(f"  [{model_name}] Extracting VAL features …")
    val_ds, y_va = build_tf_dataset_from_df(val_df, preprocess_fn)
    X_va = extractor.predict(val_ds, verbose=1)

    print(f"  [{model_name}] Extracting TEST features …")
    test_ds, y_te = build_tf_dataset_from_df(test_df, preprocess_fn)
    X_te = extractor.predict(test_ds, verbose=1)

    print(f"  [{model_name}] Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")

    np.save(feat_train,  X_tr);  np.save(label_train, y_tr)
    np.save(feat_val,    X_va);  np.save(label_val,   y_va)
    np.save(feat_test,   X_te);  np.save(label_test,  y_te)
    np.save(classes_f,   np.array(ALL_CLASSES, dtype=object))

    del extractor
    tf.keras.backend.clear_session()

    return X_tr, y_tr, X_va, y_va, X_te, y_te, ALL_CLASSES


# ── 8. COSINE DECAY WITH LINEAR WARM-UP ─────────────────────
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, total_steps, warmup_frac=0.10, min_lr=1e-7):
        super().__init__()
        self.peak_lr      = peak_lr
        self.total_steps  = float(total_steps)
        self.warmup_steps = int(total_steps * warmup_frac)
        self.min_lr       = min_lr

    def __call__(self, step):
        step  = tf.cast(step, tf.float32)
        ws    = tf.cast(self.warmup_steps, tf.float32)
        total = self.total_steps

        warmup_lr = self.peak_lr * (step / tf.maximum(ws, 1.0))
        cos_arg   = tf.cast(np.pi, tf.float32) * (step - ws) / tf.maximum(total - ws, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + tf.cos(cos_arg))

        return tf.where(step < ws, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr":      self.peak_lr,
            "total_steps":  self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr":       self.min_lr,
        }


# ── 9. DEEP REGULARISED DENSE HEAD ──────────────────────────
def build_dense_head(input_dim: int, num_classes: int,
                     model_name: str, total_steps: int):
    reg = regularizers.l2(L2_REG)
    inp = layers.Input(shape=(input_dim,), name="features", dtype="float32")

    x = layers.Dense(DENSE_UNITS[0], activation="relu",
                     kernel_regularizer=reg, name="dense_512")(inp)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(DROPOUT_1, name="dropout_1")(x)

    x = layers.Dense(DENSE_UNITS[1], activation="relu",
                     kernel_regularizer=reg, name="dense_256")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(DROPOUT_2, name="dropout_2")(x)

    x   = layers.Lambda(lambda t: tf.cast(t, tf.float32), name="cast_fp32")(x)
    out = layers.Dense(num_classes, activation="softmax",
                       dtype="float32", name="predictions")(x)

    lr_schedule = WarmupCosineDecay(
        peak_lr=1e-3, total_steps=total_steps, warmup_frac=WARMUP_FRAC
    )
    head = models.Model(inp, out, name=f"{model_name}_dense_head")
    head.compile(
        optimizer=optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=["accuracy"],
    )
    return head


# ── 10. CLASS WEIGHTS ────────────────────────────────────────
def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(zip(classes.astype(int), weights))


# ── 11. MAIN TRAINING FUNCTION ───────────────────────────────
def train_model(model_name: str):
    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*60}")

    # Extract features
    X_train, y_train, X_val, y_val, X_test, y_test, class_names = \
        extract_bottleneck_features(model_name)

    y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_oh   = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
    y_test_oh  = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    input_dim = X_train.shape[1]
    n_train   = X_train.shape[0]
    total_steps = (n_train // BATCH_SIZE + 1) * EPOCHS_HEAD

    # Build and train head
    head = build_dense_head(input_dim, NUM_CLASSES, model_name, total_steps)

    history = head.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=EPOCHS_HEAD,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Evaluate
    val_loss, val_acc = head.evaluate(X_val, y_val_oh, verbose=0)
    tst_loss, tst_acc = head.evaluate(X_test, y_test_oh, verbose=0)

    print(f"\n{model_name} → Val Acc: {val_acc:.4f}, Test Acc: {tst_acc:.4f}")

    # =========================
    # 🔥 BUILD FULL MODEL
    # =========================
    extractor = build_feature_extractor(model_name)

    inp = tf.keras.Input(shape=(224, 224, 3))
    features = extractor(inp, training=False)
    output = head(features)

    full_model = tf.keras.Model(inp, output, name=f"{model_name}_FULL_MODEL")

    # =========================
    # 💾 SAVE FULL MODEL
    # =========================
    save_path = os.path.join(RESULTS_DIR, f"{model_name}_FULL_MODEL.keras")

    full_model.save(save_path)

    print(f"✅ Saved model → {save_path}")

    return {
        "model": model_name,
        "val_accuracy": val_acc,
        "tst_accuracy": tst_acc,
        "feature_dim": input_dim,
        "history": history.history,
    }
    
# ── 12. RUN ALL THREE MODELS ─────────────────────────────────
print("\n" + "="*60)
print("  STARTING BASELINE SPECIALIST v2 PIPELINE (CSV splits)")
print("  Dual-Pool Frozen Feature Extraction")
print("="*60)

all_results = {}
for model_name in ["VGG16", "ResNet50", "MobileNetV2"]:
    result = train_model(model_name)
    all_results[model_name] = result
    tf.keras.backend.clear_session()


# ── 13. FINAL COMPARISON TABLE + CHART ──────────────────────
print("\n" + "="*60)
print("  *** RESULTS SUMMARY — MEMBER 1 BASELINE SPECIALIST v2 ***")
print("="*60)
print(f"  {'Model':<15} {'Val Acc':>10} {'Test Acc':>10} {'Feature Dim':>14}")
print("  " + "-" * 55)
for name, res in all_results.items():
    print(f"  {name:<15} {res['val_accuracy']*100:>9.2f}%"
          f" {res['tst_accuracy']*100:>9.2f}%"
          f" {res['feature_dim']:>14,}")

names    = list(all_results.keys())
val_accs = [all_results[m]["val_accuracy"] * 100 for m in names]
tst_accs = [all_results[m]["tst_accuracy"] * 100 for m in names]

x, width = np.arange(len(names)), 0.35
fig, ax  = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, val_accs, width, label="Val",  color="#4C72B0", edgecolor="black")
bars2 = ax.bar(x + width/2, tst_accs, width, label="Test", color="#DD8452", edgecolor="black")
for bar, a in zip(list(bars1) + list(bars2), val_accs + tst_accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{a:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_title("Val & Test Accuracy — Frozen Dual-Pool Feature Extractor\n"
             "(Member 1 — Baseline Specialist v2)", fontsize=12)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("Accuracy (%)"); ax.legend()
ax.set_ylim(0, min(100, max(val_accs + tst_accs) + 8))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "member1_v2_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.show()

summary_df = pd.DataFrame([
    {
        "Model":           n,
        "Val_Accuracy_%":  round(all_results[n]["val_accuracy"]  * 100, 2),
        "Test_Accuracy_%": round(all_results[n]["tst_accuracy"]  * 100, 2),
        "Val_Loss":        round(all_results[n]["val_loss"],  4),
        "Test_Loss":       round(all_results[n]["tst_loss"],  4),
        "Feature_Dim":     all_results[n]["feature_dim"],
        "Method":          "Frozen Dual-Pool Extractor",
        "Head":            "Dense(512→BN)+Dense(256→BN)",
        "L2_Reg":          L2_REG,
        "Label_Smooth":    LABEL_SMOOTH,
        "Epochs_Run":      len(all_results[n]["history"]["val_accuracy"]),
    }
    for n in names
])
csv_path = os.path.join(RESULTS_DIR, "member1_v2_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\n  Summary CSV → {csv_path}")
print("  All done! ✓")