"""
quick_train.py  –  Fine-tune YOLOv8-nano on ambulance + fire truck dataset.
Uses only 400 images at 320 px for speed.  On a low-spec CPU this runs in
roughly 15-20 minutes and produces best_model.pt ready for camera.py.
"""

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
TRAIN_IMGS = BASE_DIR / "fire truck" / "train" / "images"
TRAIN_LBLS = BASE_DIR / "fire truck" / "train" / "labels"
TEMP_DIR   = BASE_DIR / "temp_dataset"

# ─── Config ───────────────────────────────────────────────────────────────────
MAX_IMAGES = 400     # keep low so CPU training stays fast (~15-20 min)
EPOCHS     = 5
IMG_SIZE   = 320     # smaller image = much faster

# ─── Prepare temp dataset with 80/20 train-val split ─────────────────────────
def prepare_dataset():
    print("Preparing dataset from 'fire truck/train' ...")
    for split in ("train", "val"):
        (TEMP_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (TEMP_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(TRAIN_IMGS.glob("*.jpg"))  +
        list(TRAIN_IMGS.glob("*.jpeg")) +
        list(TRAIN_IMGS.glob("*.png"))
    )
    if not images:
        raise FileNotFoundError(f"No images found in {TRAIN_IMGS}")

    random.seed(42)
    images = images[:MAX_IMAGES]
    random.shuffle(images)

    cut = int(len(images) * 0.8)
    splits = {"train": images[:cut], "val": images[cut:]}

    for split_name, split_imgs in splits.items():
        for img_path in split_imgs:
            dst_img = TEMP_DIR / split_name / "images" / img_path.name
            dst_lbl = TEMP_DIR / split_name / "labels" / (img_path.stem + ".txt")
            shutil.copy2(img_path, dst_img)
            lbl_src = TRAIN_LBLS / (img_path.stem + ".txt")
            if lbl_src.exists():
                shutil.copy2(lbl_src, dst_lbl)

    print(f"  Train: {len(splits['train'])} images  |  Val: {len(splits['val'])} images")


# ─── Write data.yaml ──────────────────────────────────────────────────────────
def write_yaml() -> str:
    yaml_path = TEMP_DIR / "data.yaml"
    train_path = (TEMP_DIR / "train" / "images").as_posix()
    val_path   = (TEMP_DIR / "val"   / "images").as_posix()
    yaml_path.write_text(
        f"train: {train_path}\n"
        f"val:   {val_path}\n"
        f"nc: 3\n"
        f"names: ['Ambulance', 'Fire truck', 'Police car']\n"
    )
    return str(yaml_path)


# ─── Train ────────────────────────────────────────────────────────────────────
def train():
    prepare_dataset()
    yaml_path = write_yaml()

    model = YOLO("yolov8n.pt")          # start from COCO pretrained nano weights

    print(f"\nStarting training: {EPOCHS} epochs  |  {IMG_SIZE}px  |  {MAX_IMAGES} images")
    print("Estimated time on a low-spec PC: 15-25 minutes.\n")

    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=4,           # small batch = low RAM usage
        workers=0,         # avoids multiprocessing issues on Windows
        device="cpu",
        project=str(BASE_DIR),
        name="emergency_detect",
        exist_ok=True,
        patience=5,
        verbose=True,
    )

    # Copy the best weights to a fixed filename that camera.py picks up
    best_pt = BASE_DIR / "emergency_detect" / "weights" / "best.pt"
    dest_pt = BASE_DIR / "best_model.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, dest_pt)
        print("\n✅  Training complete!")
        print(f"    Model saved → best_model.pt")
        print("    Now run:  python camera.py")
    else:
        print("Training finished but best.pt not found – check emergency_detect/weights/")


if __name__ == "__main__":
    train()
