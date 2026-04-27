"""
Обучение YOLOv8 на датасете бейджей.
"""
import os
import shutil
import yaml
from pathlib import Path


def merge_datasets(dataset_dirs: list, merged_dir: str = "./yolo/dataset_merged"):
    merged = Path(merged_dir)
    for split in ("train", "valid", "test"):
        (merged / "images" / split).mkdir(parents=True, exist_ok=True)
        (merged / "labels" / split).mkdir(parents=True, exist_ok=True)

    img_count = 0
    for ds_path in dataset_dirs:
        ds = Path(ds_path)
        if not ds.exists():
            print(f"  Пропуск (не найден): {ds_path}")
            continue

        for split in ("train", "valid", "test"):
            img_src = ds / split / "images"
            lbl_src = ds / split / "labels"
            if not img_src.exists():
                continue

            for img_file in img_src.glob("*.[jp][pn]g"):
                new_name = f"{ds.name}__{img_file.name}"
                shutil.copy2(img_file, merged / "images" / split / new_name)

                lbl_file = lbl_src / (img_file.stem + ".txt")
                dst_lbl = merged / "labels" / split / f"{ds.name}__{img_file.stem}.txt"
                if lbl_file.exists():
                    lines = lbl_file.read_text().splitlines()
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            new_lines.append("0 " + " ".join(parts[1:]))
                    dst_lbl.write_text("\n".join(new_lines))
                else:
                    dst_lbl.write_text("")
                img_count += 1

    for split in ("train", "valid", "test"):
        n = len(list((merged / "images" / split).glob("*.[jp][pn]g")))
        if n:
            print(f"  {split}: {n} изображений")

    # ВАЖНО: ultralytics не умеет в кириллические пути через поле path.
    # Решение: используем прямые абсолютные пути в train/val/test,
    # поле path НЕ указываем — тогда ultralytics не пытается его резолвить.
    abs_merged = merged.resolve()
    train_path = str(abs_merged / "images" / "train")
    val_path   = str(abs_merged / "images" / "valid")
    test_path  = str(abs_merged / "images" / "test")

    # Если valid пустой — используем train для валидации
    if not any((abs_merged / "images" / "valid").glob("*.[jp][pn]g")):
        print("  Папка valid пустая, используем train для валидации")
        val_path = train_path

    yaml_content = (
        f"train: {train_path}\n"
        f"val: {val_path}\n"
        f"test: {test_path}\n"
        f"nc: 1\n"
        f"names: ['badge']\n"
    )

    yaml_path = merged / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"Объединено {img_count} изображений → {merged_dir}")
    print(f"data.yaml: {yaml_path}")
    print(f"  train: {train_path}")
    print(f"  val:   {val_path}")
    return str(yaml_path.resolve())


def train(
    data_yaml: str,
    model_size: str = "n",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "./yolo/runs",
    name: str = "badge_detector",
):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Установи: pip install ultralytics")

    base_model = f"yolov8{model_size}.pt"
    print(f"Загружаем базовую модель: {base_model}")
    model = YOLO(base_model)

    print(f"\nОбучение: {epochs} эпох, imgsz={imgsz}, batch={batch}")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        pretrained=True,
        patience=15,
        save=True,
        plots=True,
        val=True,
        verbose=True,
        exist_ok=True,
    )

    best_weights = Path(project) / name / "weights" / "best.pt"
    print(f"\nОбучение завершено!")
    print(f"Лучшие веса: {best_weights}")
    try:
        print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    except Exception:
        pass
    return str(best_weights)


if __name__ == "__main__":
    import glob

    dataset_dirs = [p.rstrip("/\\") for p in glob.glob("./dataset/*/")]
    print(f"Найдено датасетов: {len(dataset_dirs)}")
    if dataset_dirs:
        print("  " + "\n  ".join(dataset_dirs[:5]) + ("..." if len(dataset_dirs) > 5 else ""))

    if not dataset_dirs:
        print("Датасеты не найдены. Сначала запусти roboflow_scraper.py")
        exit(1)

    data_yaml = merge_datasets(dataset_dirs)

    best_weights = train(
        data_yaml=data_yaml,
        model_size="n",
        epochs=10,
        imgsz=480,
        batch=4,
    )
    print(f"\nДетекция: python yolo/detect.py {best_weights} <image>")