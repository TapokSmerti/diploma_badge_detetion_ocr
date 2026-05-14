"""
Переразметка негативных датасетов для обучения YOLOv8.

Два режима:

1. PURE NEGATIVE (--mode pure)
   Просто копирует изображения с пустыми label-файлами.
   "На этих фото бейджей нет" — модель учится не срабатывать.

2. HARD NEGATIVE MINING (--mode hard)
   Прогоняет текущую модель на негативах, отбирает только те
   изображения где модель ошиблась (ложные срабатывания).
   Это самые ценные примеры — именно то что модель путает с бейджами.

Использование:
    # Режим 1: все негативы
    python relabel_negatives.py --mode pure

    # Режим 2: только hard negatives (рекомендуется)
    python relabel_negatives.py --mode hard --weights yolo/runs/badge_detector/weights/best.pt

    # После — добавить в датасет и переобучить
    python yolo/train.py
"""

import argparse
import shutil
import cv2
from pathlib import Path


NEGATIVES_SOURCE = Path("./negatives")
OUTPUT_DIR       = Path("./dataset/hard_negatives")
SPLITS           = {"train": 0.85, "valid": 0.15}  # пропорции разбивки


def collect_images(source_dir: Path) -> list:
    """Рекурсивно собирает все изображения из папки."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in source_dir.rglob("*") if p.suffix.lower() in exts]
    return imgs


def split_files(files: list, ratios: dict) -> dict:
    """Делит список файлов на train/valid по заданным пропорциям."""
    import random
    files = files.copy()
    random.shuffle(files)
    result = {}
    idx = 0
    items = list(ratios.items())
    for i, (split, ratio) in enumerate(items):
        if i == len(items) - 1:
            result[split] = files[idx:]
        else:
            n = int(len(files) * ratio)
            result[split] = files[idx:idx + n]
            idx += n
    return result


def save_negative(img_path: Path, dest_img: Path, dest_lbl: Path,
                  resize: tuple = None):
    """Копирует изображение и создаёт пустой label-файл."""
    if resize:
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        img = cv2.resize(img, resize)
        cv2.imwrite(str(dest_img), img)
    else:
        shutil.copy2(img_path, dest_img)
    dest_lbl.write_text("")  # пустой файл = нет объектов
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Режим 1: Pure Negative
# ──────────────────────────────────────────────────────────────────────────────

def mode_pure(source_dir: Path, out_dir: Path, max_images: int, resize):
    """Добавляет все изображения из source как негативы."""
    all_images = collect_images(source_dir)
    print(f"Найдено изображений: {len(all_images)}")

    if max_images and len(all_images) > max_images:
        import random
        all_images = random.sample(all_images, max_images)
        print(f"Ограничено до: {max_images}")

    splits = split_files(all_images, SPLITS)
    counts = {}

    for split, files in splits.items():
        img_dir = out_dir / split / "images"
        lbl_dir = out_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i, src in enumerate(files):
            stem = f"pureneg_{split}_{i:06d}"
            ext  = src.suffix.lower()
            dst_img = img_dir / (stem + ext)
            dst_lbl = lbl_dir / (stem + ".txt")
            if save_negative(src, dst_img, dst_lbl, resize):
                saved += 1

        counts[split] = saved
        print(f"  {split}: {saved} изображений")

    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Режим 2: Hard Negative Mining
# ──────────────────────────────────────────────────────────────────────────────

def mode_hard(source_dir: Path, out_dir: Path, weights: str,
              conf: float, max_images: int, resize):
    """
    Прогоняет модель на изображениях без бейджей.
    Сохраняет только те кадры, где модель ошибочно нашла бейдж.
    Это самые полезные негативы для переобучения.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    print(f"Загружаю модель: {weights}")
    model = YOLO(weights)

    all_images = collect_images(source_dir)
    print(f"Изображений для проверки: {len(all_images)}")

    if max_images and len(all_images) > max_images:
        import random
        all_images = random.sample(all_images, max_images)
        print(f"Ограничено до: {max_images} (используй --max-images для изменения)")

    # Фильтруем — оставляем только ложные срабатывания
    false_positives = []
    print("Прогоняю инференс...")
    for i, img_path in enumerate(all_images):
        if i % 200 == 0:
            print(f"  {i}/{len(all_images)}: найдено FP={len(false_positives)}")
        try:
            results = model(str(img_path), conf=conf, verbose=False)
            if len(results[0].boxes) > 0:
                false_positives.append(img_path)
        except Exception:
            continue

    print(f"\nЛожных срабатываний: {len(false_positives)} / {len(all_images)}")
    if not false_positives:
        print("⚠️  Модель не ошиблась ни разу — хорошая новость!")
        print("   Попробуй снизить --conf (например 0.3) чтобы найти слабые срабатывания")
        return {}

    # Сохраняем FP как hard negatives
    splits = split_files(false_positives, SPLITS)
    counts = {}

    for split, files in splits.items():
        img_dir = out_dir / split / "images"
        lbl_dir = out_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i, src in enumerate(files):
            stem = f"hardneg_{split}_{i:06d}"
            ext  = src.suffix.lower()
            dst_img = img_dir / (stem + ext)
            dst_lbl = lbl_dir / (stem + ".txt")
            if save_negative(src, dst_img, dst_lbl, resize):
                saved += 1

        counts[split] = saved
        print(f"  {split}: {saved} hard negatives")

    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Визуализация — проверить что насобирали
# ──────────────────────────────────────────────────────────────────────────────

def visualize_sample(out_dir: Path, n: int = 9):
    """Показывает сетку из n сохранённых негативных примеров."""
    import numpy as np
    imgs = list((out_dir / "train" / "images").glob("*.[jp][pn]g"))
    if not imgs:
        return
    import random
    sample = random.sample(imgs, min(n, len(imgs)))

    tiles = []
    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (200, 150))
        tiles.append(img)

    if not tiles:
        return

    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    grid_rows = []
    for r in range(rows):
        row_tiles = tiles[r*cols:(r+1)*cols]
        while len(row_tiles) < cols:
            row_tiles.append(np.zeros((150, 200, 3), dtype=np.uint8))
        grid_rows.append(np.hstack(row_tiles))
    grid = np.vstack(grid_rows)

    out_path = out_dir / "sample_negatives.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"\n🖼  Пример негативов сохранён: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["pure", "hard"], default="hard",
                        help="pure = все изображения; hard = только ложные срабатывания")
    parser.add_argument("--source",     default=str(NEGATIVES_SOURCE),
                        help="Папка с негативными изображениями")
    parser.add_argument("--output",     default=str(OUTPUT_DIR))
    parser.add_argument("--weights",    default="./yolo/runs/badge_detector/weights/best.pt",
                        help="Путь к best.pt (только для --mode hard)")
    parser.add_argument("--conf",       type=float, default=0.3,
                        help="Порог инференса (низкий = ловим слабые FP)")
    parser.add_argument("--max-images", type=int,   default=5000,
                        help="Максимум изображений для обработки")
    parser.add_argument("--resize",     type=int,   default=None,
                        help="Ресайз до N×N (например 640)")
    args = parser.parse_args()

    source = Path(args.source)
    out    = Path(args.output)
    resize = (args.resize, args.resize) if args.resize else None

    if not source.exists():
        print(f"❌ Папка не найдена: {source}")
        print("   Сначала запусти: python download_negative_datasets.py")
        return

    print(f"Режим:   {args.mode.upper()}")
    print(f"Источник: {source}")
    print(f"Выход:   {out}")
    print()

    if args.mode == "pure":
        counts = mode_pure(source, out, args.max_images, resize)
    else:
        counts = mode_hard(source, out, args.weights, args.conf, args.max_images, resize)

    if not counts:
        return

    total = sum(counts.values())
    print(f"\n✅ Итого сохранено: {total} негативных примеров")
    print(f"   Папка: {out.resolve()}")

    # Сетка-превью
    visualize_sample(out)

    print("\n" + "="*55)
    print("Следующий шаг — добавь папку в датасет и переобучи.")
    print("В yolo/train.py добавь путь в dataset_dirs:")
    print(f'  dataset_dirs.append("{out.resolve()}")')
    print("Или добавь в datasets.txt и прогони merge заново.")
    print("="*55)


if __name__ == "__main__":
    main()