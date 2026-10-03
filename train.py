"""
Cats vs Dogs - MobileNetV2 transfer learning trainer.

Usage:
    # Option A: you downloaded + unzipped the Kaggle dataset into ./dataset
    python train.py --data_dir dataset

    # Option B: download automatically with kagglehub (pip install kagglehub)
    python train.py --download

Any image whose folder or file name contains "cat" or "dog" is used, so any
train/test/validation or flat layout works.

Outputs:
    models/cats_dogs_model.h5        trained model (rescaling layer included)
    models/cats_dogs_model.keras     same model, backup format
    models/class_names.json          ["cat", "dog"]
    models/metrics.json              accuracy, precision, recall, F1, sizes ...
    models/training_history.png      accuracy / loss curves
    models/confusion_matrix.png      test-set confusion matrix
    sample_images/                   a few held-out images for the app
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
IMG_SIZE = (224, 224)
CLASS_NAMES = ["cat", "dog"]  # label 0 = cat, label 1 = dog (sigmoid outputs P(dog))
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODELS_DIR = Path("models")
SAMPLES_DIR = Path("sample_images")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def download_dataset() -> Path:
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("Run `pip install kagglehub` first, or download the dataset manually.")
    path = kagglehub.dataset_download("aleemaparakatta/cats-and-dogs-mini-dataset")
    print("Dataset downloaded to:", path)
    return Path(path)


def collect_images(data_dir: Path):
    """Return image paths and labels (0 = cat, 1 = dog) from any folder layout."""
    paths, labels = [], []
    for p in sorted(data_dir.rglob("*")):
        if p.suffix.lower() not in VALID_EXT:
            continue
        hint = f"{p.parent.name} {p.stem}".lower()
        if "cat" in hint:
            paths.append(str(p))
            labels.append(0)
        elif "dog" in hint:
            paths.append(str(p))
            labels.append(1)
    return paths, labels


def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img.set_shape((*IMG_SIZE, 3))
    return img, tf.cast(label, tf.float32)


def make_dataset(paths, labels, batch_size, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
        # small dataset: show every image many times per epoch, each time augmented differently
        ds = ds.repeat(max(1, 200 // len(paths)))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        augment = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.1),
        ])
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model():
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)  # pixels -> [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="cats_dogs_mobilenetv2"), base


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#0B1020")
    for ax, (tr, va, title) in zip(
        axes, [("accuracy", "val_accuracy", "Accuracy"), ("loss", "val_loss", "Loss")]):
        ax.set_facecolor("#131A2E")
        ax.plot(history[tr], color="#8B5CF6", lw=2.2, label="Train")
        ax.plot(history[va], color="#22D3EE", lw=2.2, label="Validation")
        ax.set_title(title, color="white", fontweight="bold")
        ax.set_xlabel("Epoch", color="#9AA3C7")
        ax.tick_params(colors="#9AA3C7")
        ax.grid(alpha=0.15)
        for s in ax.spines.values():
            s.set_color("#2A3354")
        ax.legend(facecolor="#131A2E", edgecolor="#2A3354", labelcolor="white")
    plt.tight_layout()
    fig.savefig(MODELS_DIR / "training_history.png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(4.4, 4))
    fig.patch.set_facecolor("#0B1020")
    ax.set_facecolor("#131A2E")
    ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(2), [c.title() for c in CLASS_NAMES], color="white")
    ax.set_yticks(range(2), [c.title() for c in CLASS_NAMES], color="white")
    ax.set_xlabel("Predicted", color="#9AA3C7")
    ax.set_ylabel("Actual", color="#9AA3C7")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center", fontsize=16,
                    fontweight="bold", color="white" if cm[i][j] < cm.max() * 0.6 else "#0B1020")
    plt.tight_layout()
    fig.savefig(MODELS_DIR / "confusion_matrix.png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--download", action="store_true", help="download via kagglehub")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fine_tune_epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args(argv)

    data_dir = download_dataset() if args.download else Path(args.data_dir)
    paths, labels = collect_images(data_dir)
    if not paths:
        raise SystemExit(
            f"No cat/dog images found in '{data_dir}'. Unzip the Kaggle dataset there "
            "or run `python train.py --download`.")
    print(f"Found {len(paths)} images | cats={labels.count(0)} dogs={labels.count(1)}")

    # ~70 / 15 / 15 stratified split (counts are forced to >= 2 per split so
    # stratification also works on very small datasets)
    n = len(paths)
    n_hold = max(int(round(n * 0.30)), 4)
    tr_p, tmp_p, tr_l, tmp_l = train_test_split(
        paths, labels, test_size=n_hold, stratify=labels, random_state=SEED)
    va_p, te_p, va_l, te_l = train_test_split(
        tmp_p, tmp_l, test_size=0.50, stratify=tmp_l, random_state=SEED)
    print(f"Train={len(tr_p)} Val={len(va_p)} Test={len(te_p)}")

    train_ds = make_dataset(tr_p, tr_l, args.batch_size, True)
    val_ds = make_dataset(va_p, va_l, args.batch_size, False)
    test_ds = make_dataset(te_p, te_l, args.batch_size, False)

    model, base = build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7),
    ]

    # Phase 1: train the classification head
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    history = {k: [float(x) for x in v] for k, v in h1.history.items()}

    # Phase 2: fine-tune the top 30 layers of MobileNetV2
    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(1e-5),
                      loss="binary_crossentropy", metrics=["accuracy"])
        h2 = model.fit(train_ds, validation_data=val_ds,
                       epochs=args.fine_tune_epochs, callbacks=callbacks)
        for k, v in h2.history.items():
            history.setdefault(k, []).extend(float(x) for x in v)

    # Evaluate on the held-out test set
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    probs = model.predict(test_ds, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(te_l, preds, labels=[0, 1])
    report = classification_report(te_l, preds, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    print(classification_report(te_l, preds, target_names=CLASS_NAMES, zero_division=0))

    # Save everything
    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / "cats_dogs_model.h5")
    model.save(MODELS_DIR / "cats_dogs_model.keras")  # backup format
    with open(MODELS_DIR / "class_names.json", "w") as f:
        json.dump(CLASS_NAMES, f)
    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "val_accuracy": float(max(history["val_accuracy"])),
        "total_images": len(paths),
        "train_images": len(tr_p),
        "val_images": len(va_p),
        "test_images": len(te_p),
        "parameters": int(model.count_params()),
        "precision": {c: float(report[c]["precision"]) for c in CLASS_NAMES},
        "recall": {c: float(report[c]["recall"]) for c in CLASS_NAMES},
        "f1": {c: float(report[c]["f1-score"]) for c in CLASS_NAMES},
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    plot_history(history)
    plot_confusion(cm)

    # Copy a few unseen test images as app samples
    SAMPLES_DIR.mkdir(exist_ok=True)
    for old in SAMPLES_DIR.glob("sample_*"):
        old.unlink()
    for label, name in enumerate(CLASS_NAMES):
        pool = [p for p, l in zip(te_p, te_l) if l == label][:4]
        for i, p in enumerate(pool, 1):
            shutil.copy(p, SAMPLES_DIR / f"sample_{name}_{i}{Path(p).suffix.lower()}")

    print(f"\nDone. Test accuracy: {test_acc:.4f}")
    print("Saved: models/cats_dogs_model.h5, metrics.json, class_names.json, charts, sample_images/")


if __name__ == "__main__":
    main()
