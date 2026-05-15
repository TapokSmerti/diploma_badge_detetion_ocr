"""
Обучение YOLOv8 на датасете бейджей.
"""
import os
import shutil
import yaml
from pathlib import Path




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
        workers=16,
        degrees=90,       
        fliplr=0.5,
        flipud=0.5,       
        scale=0.5,        
        shear=5.0,
        mixup=0.5,
        copy_paste=0.5,
        perspective=0.001 
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
    ROOT = Path(__file__).resolve().parent.parent
    data_yaml = ROOT / "dataset" / "dataset_merged" / "data.yaml"

    if not Path(data_yaml).exists():
        print(f"❌ data.yaml не найден: {data_yaml}")
        print("Сначала запусти merge_dataset.py")
        exit(1)

    best_weights = train(
        data_yaml=data_yaml,
        model_size="n",
        epochs=100,
        imgsz=640,
        batch=64,
    )

    print(f"\nДетекция:")
    print(f"python yolo/detect.py {best_weights} <image>")
