#!/usr/bin/env python3
"""
Массовый экспорт детекций RT-DETR для всех видео из папки в формат CVAT.
"""
from __future__ import annotations

import json
import cv2
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import torch
import sys


def export_detections_to_cvat(
    model_path: str,
    video_path: str,
    output_json: str,
    conf_threshold: float = 0.5,
) -> str:
    """Экспорт детекций одного видео в COCO JSON"""
    
    print(f"  Загрузка модели...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    
    # Информация о видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print(f"  Видео: {Path(video_path).name} ({width}x{height}, {total_frames} кадров, {fps:.2f} FPS)")
    
    # Структура COCO JSON
    coco_data = {
        "info": {
            "description": f"Detections from {Path(model_path).name} on {Path(video_path).name}",
            "date_created": datetime.now().isoformat(),
            "version": "1.0"
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "badge"}]
    }
    
    # Инференс
    results = model(
        video_path,
        stream=True,
        conf=conf_threshold,
        verbose=False,
        device=device
    )
    
    ann_id = 0
    frame_count = 0
    total_detections = 0
    
    for frame_idx, result in enumerate(results):
        # Информация о кадре
        coco_data["images"].append({
            "id": frame_idx,
            "file_name": f"frame_{frame_idx:06d}.jpg",
            "width": width,
            "height": height,
            "frame": frame_idx
        })
        
        # Детекции
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": frame_idx,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "area": float((x2-x1) * (y2-y1)),
                    "score": float(conf),
                    "iscrowd": 0
                })
                ann_id += 1
                total_detections += 1
        
        frame_count += 1
        if frame_count % 100 == 0 or frame_count == total_frames:
            pct = 100 * frame_count / total_frames
            print(f"    {pct:.1f}% ({frame_count}/{total_frames} кадров) — {total_detections} детекций", end="\r")
    
    print()  # перевод строки после прогресс-бара
    
    # Сохраняем JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"    ✅ {total_detections} детекций сохранено в {Path(output_json).name}")
    return output_json


def find_all_videos(video_dir: str) -> list[Path]:
    """Находит все видеофайлы в папке рекурсивно."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4", ".AVI", ".MOV", ".MKV", ".WEBM"}
    video_dir_path = Path(video_dir)
    
    if not video_dir_path.exists():
        return []
    
    videos = []
    for ext in video_extensions:
        videos.extend(video_dir_path.rglob(f"*{ext}"))
    
    return sorted(videos)


def main():
    """Обрабатывает все видео из папки и сохраняет детекции для CVAT"""
    
    # ========== КОНФИГУРАЦИЯ ==========
    # Пути относительно папки inference/
    PROJECT_ROOT = Path(__file__).parent.parent  # поднимаемся на уровень выше inference/
    
    MODEL_PATH = PROJECT_ROOT / "yolo/runs/detect/yolo/runs/rtdetr_l/weights/best.pt"
    VIDEO_DIR = PROJECT_ROOT / "dataset/downloaded_videos/test_vids"
    OUTPUT_DIR = Path("./cvat_exports")  # сохраняем в текущей папке (inference/cvat_exports)
    CONF_THRESHOLD = 0.5
    # ==================================
    
    print("=" * 60)
    print("Экспорт RT-DETR детекций в CVAT формат")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Модель: {MODEL_PATH}")
    print(f"Папка с видео: {VIDEO_DIR}")
    print(f"Папка для результатов: {OUTPUT_DIR}")
    
    # Проверяем существование модели
    if not MODEL_PATH.exists():
        print(f"\n❌ Модель не найдена: {MODEL_PATH}")
        print("Поиск других моделей...")
        other_models = list(PROJECT_ROOT.rglob("best.pt"))
        if other_models:
            print("Найдены другие модели:")
            for m in other_models[:5]:
                print(f"   - {m}")
        sys.exit(1)
    else:
        print(f"✅ Модель найдена: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Проверяем существование папки с видео
    if not VIDEO_DIR.exists():
        print(f"\n❌ Папка с видео не найдена: {VIDEO_DIR}")
        sys.exit(1)
    
    # Находим все видео
    video_paths = find_all_videos(str(VIDEO_DIR))
    
    if not video_paths:
        print(f"\n❌ Видео не найдены в {VIDEO_DIR}")
        print("Поддерживаемые форматы: .mp4, .avi, .mov, .mkv, .webm")
        sys.exit(1)
    
    print(f"\n📹 Найдено видео: {len(video_paths)}")
    for v in video_paths:
        file_size = v.stat().st_size / 1024 / 1024
        print(f"   - {v.name} ({file_size:.1f} MB)")
    
    # Создаём папку для результатов
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Проверяем CUDA
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"\n🖥️ GPU: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GB)")
    else:
        print("\n⚠️ CUDA недоступна, используется CPU (будет медленно)")
    
    # Обрабатываем каждое видео
    print(f"\n🚀 Запуск обработки {len(video_paths)} видео...\n")
    
    saved_files = []
    for i, video_path in enumerate(video_paths, 1):
        print(f"\n[{i}/{len(video_paths)}] 📹 {video_path.name}")
        
        output_json = OUTPUT_DIR / f"rtdetr_{video_path.stem}.json"
        
        try:
            json_path = export_detections_to_cvat(
                model_path=str(MODEL_PATH),
                video_path=str(video_path),
                output_json=str(output_json),
                conf_threshold=CONF_THRESHOLD
            )
            saved_files.append(json_path)
        except Exception as e:
            print(f"    ❌ Ошибка: {e}")
    
    # Итог
    print(f"\n{'='*60}")
    print(f"✅ Готово! Обработано {len(saved_files)} из {len(video_paths)} видео")
    print(f"📁 Результаты сохранены в: {OUTPUT_DIR.absolute()}")
    
    if saved_files:
        print(f"\n📋 Файлы для импорта в CVAT:")
        for f in saved_files:
            print(f"   - {Path(f).name}")
        
        print(f"\n📋 Дальнейшие действия:")
        print(f"   1. В CVAT создайте Task и загрузите видео")
        print(f"   2. Импортируйте соответствующий JSON (формат COCO JSON 1.1)")
        print(f"   3. Исправьте детекции вручную (добавьте/удалите/поправьте bbox)")
        print(f"   4. Экспортируйте Ground Truth из CVAT")
        print(f"   5. Используйте GT для сравнения всех моделей")


if __name__ == "__main__":
    main()