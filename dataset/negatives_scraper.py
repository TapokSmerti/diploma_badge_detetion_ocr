"""
Скачивание негативных датасетов для Hard Negative Mining.
Рассчитан на запуск в Google Colab или локально.

Скачивает:
  - COCO val2017        (~1 GB,  5 000 фото)
  - INRIA Person        (~200 MB, 1 800 фото)
  - Open Images subset  (~1 GB, ~5 000 фото, категории: Phone/Book/Laptop/Sign)

После скачивания запускай: python relabel_negatives.py
"""

import os
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path


OUTPUT_DIR = Path("./negatives")
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = ""):
    """Скачивает файл с прогресс-баром."""
    if dest.exists():
        print(f"  уже скачан: {dest.name}")
        return
    print(f"  Скачиваю {desc or dest.name} ...")
    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, count * block_size * 100 // total_size)
                print(f"\r    {pct}%", end="", flush=True)
        urllib.request.urlretrieve(url, dest, reporthook)
        print(f"\r    ✅ {dest.name}")
    except Exception as e:
        print(f"\r    ❌ Ошибка: {e}")
        if dest.exists():
            dest.unlink()
        raise


def extract_zip(archive: Path, out_dir: Path):
    print(f"  Распаковываю {archive.name} → {out_dir.name}/")
    with zipfile.ZipFile(archive) as z:
        z.extractall(out_dir)


def extract_tar(archive: Path, out_dir: Path):
    print(f"  Распаковываю {archive.name} → {out_dir.name}/")
    with tarfile.open(archive) as t:
        t.extractall(out_dir)


# ──────────────────────────────────────────────────────────────────────────────
# 1. COCO val2017 — 5 000 разнообразных фото
# ──────────────────────────────────────────────────────────────────────────────

def download_coco_val(out_dir: Path = OUTPUT_DIR / "coco"):
    """
    Скачивает COCO val2017 (~1 GB, 5 000 изображений).
    Это идеальный источник для hard negative mining:
    офисы, люди, еда, транспорт — всё кроме бейджей.
    """
    print("\n[1/3] COCO val2017")

    img_dir = out_dir / "images"
    if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) > 4000:
        print(f"  ✅ Уже скачан ({len(list(img_dir.glob('*.jpg')))} фото)")
        return img_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / "val2017.zip"
    download_file(
        "http://images.cocodataset.org/zips/val2017.zip",
        archive,
        "COCO val2017 (5000 фото, ~1 GB)"
    )

    print("  Распаковываю...")
    with zipfile.ZipFile(archive) as z:
        z.extractall(out_dir)

    # COCO распаковывается в val2017/ — переименуем в images/
    raw = out_dir / "val2017"
    if raw.exists():
        raw.rename(img_dir)

    print(f"  ✅ COCO: {len(list(img_dir.glob('*.jpg')))} изображений → {img_dir}")
    archive.unlink()  # удаляем архив чтобы сэкономить место
    return img_dir


# ──────────────────────────────────────────────────────────────────────────────
# 2. INRIA Person Dataset — люди без бейджей
# ──────────────────────────────────────────────────────────────────────────────

def download_inria(out_dir: Path = OUTPUT_DIR / "inria"):
    """
    INRIA Person Dataset — 1 832 фото людей.
    Очень релевантно: люди в рамке, но без бейджей.
    Размер ~200 MB.
    """
    print("\n[2/3] INRIA Person Dataset")

    img_dir = out_dir / "images"
    if img_dir.exists() and len(list(img_dir.rglob("*.png"))) > 100:
        n = len(list(img_dir.rglob("*.png")))
        print(f"  ✅ Уже скачан ({n} фото)")
        return img_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / "INRIAPerson.tar"
    download_file(
        "ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar",
        archive,
        "INRIA Person (~200 MB)"
    )

    extract_tar(archive, out_dir)
    archive.unlink()

    # Собираем все изображения из train и test в одну папку
    img_dir.mkdir(exist_ok=True)
    count = 0
    raw_root = out_dir / "INRIAPerson"
    for img_path in raw_root.rglob("*.png"):
        if "annotations" in str(img_path).lower():
            continue
        dest = img_dir / f"inria_{count:05d}{img_path.suffix}"
        shutil.copy2(img_path, dest)
        count += 1

    print(f"  ✅ INRIA: {count} изображений → {img_dir}")
    return img_dir


# ──────────────────────────────────────────────────────────────────────────────
# 3. Open Images v7 — выборка по категориям
# ──────────────────────────────────────────────────────────────────────────────

def download_open_images_subset(
    out_dir: Path = OUTPUT_DIR / "open_images",
    categories: list = None,
    max_images: int = 1000,
):
    """
    Скачивает подмножество Open Images v7 по нужным категориям.
    Используем официальный инструмент openimages (pip install openimages).

    Категории по умолчанию — прямоугольные объекты, похожие на бейджи:
      Mobile phone, Book, Laptop, Whiteboard, Poster, Sign, Tablet computer,
      Envelope, Passport, Ticket

    max_images — сколько фото на категорию (Open Images огромный, берём срез).
    """
    print("\n[3/3] Open Images v7 subset")

    if categories is None:
        categories = [
            "Mobile phone",
            "Book",
            "Laptop",
            "Whiteboard",
            "Poster",
            "Tablet computer",
            "Envelope",
            "Passport",
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    # Пробуем через openimages пакет
    try:
        import openimages  # noqa
    except ImportError:
        print("  Устанавливаю openimages...")
        os.system("pip install openimages -q")

    try:
        from openimages.download import download_images
        for cat in categories:
            cat_dir = img_dir / cat.replace(" ", "_").lower()
            if cat_dir.exists() and len(list(cat_dir.glob("*.jpg"))) > 10:
                print(f"  {cat}: уже скачан")
                continue
            print(f"  Скачиваю категорию: {cat}")
            try:
                download_images(
                    str(cat_dir),
                    [cat],
                    limit=max_images,
                    exclude_depictions=True,
                )
                n = len(list(cat_dir.glob("*.jpg")))
                print(f"  ✅ {cat}: {n} фото")
            except Exception as e:
                print(f"  ⚠️  {cat}: {e}")

        # Плоская структура — собираем всё в images/
        total = 0
        for cat_dir in img_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            for f in cat_dir.glob("*.jpg"):
                dest = img_dir / f"{cat_dir.name}_{f.name}"
                if not dest.exists():
                    shutil.copy2(f, dest)
                    total += 1

        print(f"  ✅ Open Images: {total} изображений → {img_dir}")
        return img_dir

    except Exception as e:
        print(f"  ⚠️  openimages не сработал: {e}")
        print("  Пробуем альтернативу через fiftyone...")
        return _download_open_images_fiftyone(out_dir, categories, max_images)


def _download_open_images_fiftyone(out_dir, categories, max_images):
    """Fallback: скачивает Open Images через fiftyone."""
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        os.system("pip install fiftyone -q")
        import fiftyone as fo
        import fiftyone.zoo as foz

    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=categories,
        max_samples=max_images,
        dataset_name="oi_negatives",
        overwrite=True,
    )

    count = 0
    for sample in dataset:
        src = Path(sample.filepath)
        if src.exists():
            dest = img_dir / f"oi_{count:05d}{src.suffix}"
            shutil.copy2(src, dest)
            count += 1

    print(f"  ✅ Open Images (fiftyone): {count} изображений → {img_dir}")
    return img_dir


# ──────────────────────────────────────────────────────────────────────────────
# Главная функция
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-coco",  action="store_true")
    parser.add_argument("--skip-inria", action="store_true")
    parser.add_argument("--skip-oi",    action="store_true",
                        help="Пропустить Open Images (самый долгий)")
    parser.add_argument("--oi-limit",   type=int, default=300,
                        help="Фото на категорию Open Images (default: 300)")
    args = parser.parse_args()

    downloaded = []

    if not args.skip_coco:
        try:
            p = download_coco_val()
            downloaded.append(("coco", p))
        except Exception as e:
            print(f"  ❌ COCO пропущен: {e}")

    if not args.skip_inria:
        try:
            p = download_inria()
            downloaded.append(("inria", p))
        except Exception as e:
            print(f"  ❌ INRIA пропущен: {e}")

    if not args.skip_oi:
        try:
            p = download_open_images_subset(max_images=args.oi_limit)
            downloaded.append(("open_images", p))
        except Exception as e:
            print(f"  ❌ Open Images пропущен: {e}")

    print("\n" + "="*50)
    print("ИТОГО скачано:")
    total = 0
    for name, img_dir in downloaded:
        n = len(list(Path(img_dir).rglob("*.[jp][pn]g")))
        print(f"  {name:15s}: {n:>5} изображений  ({img_dir})")
        total += n
    print(f"  {'ВСЕГО':15s}: {total:>5} изображений")
    print(f"\nПапка: {OUTPUT_DIR.resolve()}")
    print("\nСледующий шаг:")
    print("  python relabel_negatives.py")


if __name__ == "__main__":
    main()