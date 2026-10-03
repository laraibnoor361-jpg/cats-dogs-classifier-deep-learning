"""
Copy a few demo images from dataset/ into sample_images/ so they appear in the
app sidebar ("Try a sample").

Usage:  python make_samples.py --data_dir dataset --per_class 4
(train.py also refreshes sample_images/ with unseen test images automatically.)
"""
import argparse
import random
import shutil
from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--per_class", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path("sample_images")
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("sample_*"):
        old.unlink()
    random.seed(42)
    for name in ("cat", "dog"):
        pool = [p for p in sorted(Path(args.data_dir).rglob("*"))
                if p.suffix.lower() in VALID_EXT and name in f"{p.parent.name} {p.stem}".lower()]
        if not pool:
            print(f"No {name} images found in {args.data_dir}")
            continue
        for i, p in enumerate(random.sample(pool, min(args.per_class, len(pool))), 1):
            shutil.copy(p, out_dir / f"sample_{name}_{i}{p.suffix.lower()}")
    print("Sample images saved to", out_dir)


if __name__ == "__main__":
    main()
