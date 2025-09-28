import os
import random
import shutil
from pathlib import Path

def split_dataset(src_dir, train_dir, test_dir, split_ratio=0.8):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    files = [f for f in Path(src_dir).glob("*") if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    random.shuffle(files)

    split_idx = int(len(files) * split_ratio)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    for f in train_files:
        shutil.copy(f, Path(train_dir) / f.name)
    for f in test_files:
        shutil.copy(f, Path(test_dir) / f.name)

    print(f"Done: {len(train_files)} train, {len(test_files)} test images.")

if __name__ == "__main__":
    # adjust paths
    split_dataset("gathered_files", "datasets/renaissance/trainA", "datasets/renaissance/testA")
    split_dataset("images", "datasets/renaissance/trainB", "datasets/renaissance/testB")
