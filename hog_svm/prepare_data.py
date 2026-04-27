"""
Подготовка данных для HOG+SVM из датасета в формате YOLO.
Вырезает позитивные примеры (бейджи) и генерирует негативные.
"""
import os
import cv2
import numpy as np
from pathlib import Path
import random

# Целевой размер патча для HOG
PATCH_SIZE = (64, 128)  # (ширина, высота)


def load_yolo_annotations(label_path: str, img_w: int, img_h: int) -> list[tuple]:
    """Парсит YOLO-разметку и возвращает список bbox в пикселях."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            # Клипуем по границам изображения
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def extract_patches(dataset_root: str, output_dir: str, neg_per_image: int = 5):
    """
    Обходит датасет в YOLO-формате, вырезает позитивы и негативы.
    Ожидаемая структура:
        dataset_root/
            images/train/*.jpg
            labels/train/*.txt
    """
    pos_dir = Path(output_dir) / "positives"
    neg_dir = Path(output_dir) / "negatives"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(dataset_root)
    pos_count = neg_count = 0

    for split in ("train", "valid", "test"):
        # Roboflow кладёт: dataset/train/images/*.jpg и dataset/train/labels/*.txt
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.exists():
            continue

        for img_path in img_dir.glob("*.[jp][pn]g"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            label_path = lbl_dir / (img_path.stem + ".txt")
            boxes = load_yolo_annotations(str(label_path), w, h)

            # Позитивы
            for x1, y1, x2, y2 in boxes:
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, PATCH_SIZE)
                out_path = pos_dir / f"pos_{pos_count:06d}.jpg"
                cv2.imwrite(str(out_path), crop)
                pos_count += 1

            # Негативы — случайные патчи без пересечения с bbox
            attempts = 0
            saved = 0
            while saved < neg_per_image and attempts < 50:
                attempts += 1
                pw, ph = PATCH_SIZE
                if w <= pw or h <= ph:
                    break
                rx = random.randint(0, w - pw)
                ry = random.randint(0, h - ph)
                # Проверяем, что не пересекаемся с позитивными bbox
                overlap = any(
                    rx < x2 and rx + pw > x1 and ry < y2 and ry + ph > y1
                    for x1, y1, x2, y2 in boxes
                )
                if overlap:
                    continue
                crop = img[ry:ry + ph, rx:rx + pw]
                out_path = neg_dir / f"neg_{neg_count:06d}.jpg"
                cv2.imwrite(str(out_path), crop)
                neg_count += 1
                saved += 1

    print(f"Готово: {pos_count} позитивов, {neg_count} негативов")
    print(f"Сохранено в: {output_dir}")
    return pos_count, neg_count


if __name__ == "__main__":
    import sys
    # Укажи путь к одному из скачанных датасетов
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "./dataset/id-card-l8shn"
    extract_patches(dataset_path, output_dir="./hog_svm/data")