"""
Объединение всех badge-датасетов + hard negatives
в единый YOLO dataset.

Структура:
dataset/
├── badges/
│   ├── dataset1/
│   ├── dataset2/
│   └── ...
└── hard_negatives/
    ├── train/
    └── valid/

Результат:
dataset_merged/
├── images/
├── labels/
└── data.yaml
"""

import shutil
from pathlib import Path
from collections import Counter


BADGES_DIR = Path("./badges")
NEGATIVES_DIR = Path("./hard_negatives")

OUTPUT_DIR = Path("./dataset_merged")


# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def copy_image_and_label(
    img_path: Path,
    lbl_path: Path | None,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    prefix: str,
):
    """
    Копирует изображение и label.
    Все классы переводятся в class_id=0 (badge).
    """

    new_name = f"{prefix}__{img_path.stem}"

    dst_img = dst_img_dir / f"{new_name}{img_path.suffix.lower()}"
    dst_lbl = dst_lbl_dir / f"{new_name}.txt"

    shutil.copy2(img_path, dst_img)

    # negative image
    if lbl_path is None or not lbl_path.exists():
        dst_lbl.write_text("")
        return

    lines = lbl_path.read_text().splitlines()

    converted = []

    for line in lines:
        parts = line.strip().split()

        # YOLO format: class x y w h
        if len(parts) < 5:
            continue

        converted.append("0 " + " ".join(parts[1:]))

    dst_lbl.write_text("\n".join(converted))


# ─────────────────────────────────────────────────────────────
# Merge badge datasets
# ─────────────────────────────────────────────────────────────

def merge_badge_datasets():

    stats = Counter()

    badge_datasets = [
        p for p in BADGES_DIR.iterdir()
        if p.is_dir()
    ]

    print(f"\nНайдено badge-датасетов: {len(badge_datasets)}")

    for dataset_dir in badge_datasets:

        print(f"\n[{dataset_dir.name}]")

        for split in ("train", "valid", "test"):

            img_dir = dataset_dir / split / "images"
            lbl_dir = dataset_dir / split / "labels"

            if not img_dir.exists():
                continue

            out_img_dir = OUTPUT_DIR / "images" / split
            out_lbl_dir = OUTPUT_DIR / "labels" / split

            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            images = [p for p in img_dir.iterdir() if is_image(p)]

            print(f"  {split}: {len(images)}")

            for img_path in images:

                lbl_path = lbl_dir / f"{img_path.stem}.txt"

                copy_image_and_label(
                    img_path=img_path,
                    lbl_path=lbl_path,
                    dst_img_dir=out_img_dir,
                    dst_lbl_dir=out_lbl_dir,
                    prefix=dataset_dir.name,
                )

                stats[split] += 1

    return stats


# ─────────────────────────────────────────────────────────────
# Add hard negatives
# ─────────────────────────────────────────────────────────────

def add_hard_negatives():

    stats = Counter()

    print("\nДобавляем hard negatives...")

    for split in ("train", "valid", "test"):

        img_dir = NEGATIVES_DIR / split / "images"
        lbl_dir = NEGATIVES_DIR / split / "labels"

        if not img_dir.exists():
            continue

        out_img_dir = OUTPUT_DIR / "images" / split
        out_lbl_dir = OUTPUT_DIR / "labels" / split

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = [p for p in img_dir.iterdir() if is_image(p)]

        print(f"  {split}: {len(images)} negatives")

        for img_path in images:

            lbl_path = lbl_dir / f"{img_path.stem}.txt"

            copy_image_and_label(
                img_path=img_path,
                lbl_path=lbl_path,
                dst_img_dir=out_img_dir,
                dst_lbl_dir=out_lbl_dir,
                prefix="hardneg",
            )

            stats[split] += 1

    return stats


# ─────────────────────────────────────────────────────────────
# Create YAML
# ─────────────────────────────────────────────────────────────

def create_yaml():

    abs_out = OUTPUT_DIR.resolve()

    train_path = str(abs_out / "images" / "train")
    val_path   = str(abs_out / "images" / "valid")
    test_path  = str(abs_out / "images" / "test")

    yaml_content = (
        f"train: {train_path}\n"
        f"val: {val_path}\n"
        f"test: {test_path}\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['badge']\n"
    )

    yaml_path = OUTPUT_DIR / "data.yaml"

    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"\nСоздан data.yaml:")
    print(yaml_path.resolve())

    return yaml_path


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():

    if OUTPUT_DIR.exists():
        print(f"Удаляю старый dataset: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    badge_stats = merge_badge_datasets()
    neg_stats   = add_hard_negatives()

    yaml_path = create_yaml()

    print("\n" + "=" * 60)
    print("DATASET MERGED")
    print("=" * 60)

    for split in ("train", "valid", "test"):

        total = badge_stats[split] + neg_stats[split]

        print(
            f"{split}: "
            f"{total} images "
            f"(badges={badge_stats[split]}, negatives={neg_stats[split]})"
        )

    print("\nYOLO config:")
    print(yaml_path.resolve())


if __name__ == "__main__":
    main()