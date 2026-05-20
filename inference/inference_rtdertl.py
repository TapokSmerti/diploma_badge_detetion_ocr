#!/usr/bin/env python3
"""
Экспорт детекций YOLO/RT-DETR в формат CVAT (COCO JSON).
Затем можно импортировать в CVAT, вручную поправить, и использовать как GT.
"""

import json
import cv2
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from typing import Optional
import torch

def export_detections_to_cvat(
    model_path: str,
    video_path: str,
    output_json: Optional[str] = None,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    class_names: dict = None,
    max_frames: Optional[int] = None,  # для теста: обработать только N кадров
) -> str:
    """
    Запускает модель на видео и сохраняет детекции в формате COCO JSON для CVAT.
    
    Args:
        model_path: путь к модели (.pt, .onnx, .torchscript)
        video_path: путь к видео
        output_json: путь для сохранения JSON (если None — генерируется автоматически)
        conf_threshold: порог уверенности
        iou_threshold: порог NMS
        class_names: словарь {id: name} для классов, если None — использует COCO классы
        max_frames: ограничить количество кадров (для быстрого теста)
    
    Returns:
        путь к сохранённому JSON файлу
    """
    
    # Загружаем модель
    print(f"Загрузка модели: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    
    # Получаем информацию о видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if max_frames and max_frames < total_frames:
        total_frames = max_frames
    
    print(f"Видео: {Path(video_path).name}")
    print(f"  Размер: {width}x{height}, {total_frames} кадров, {fps:.2f} FPS")
    print(f"Устройство: {device}")
    
    # Определяем классы
    if class_names is None:
        # Стандартные классы COCO (person = 0, но для бейджей лучше указать свой)
        # Для детекции бейджей: class_id=0 (badge)
        class_names = {0: "badge"}
    
    # Подготавливаем структуру COCO JSON
    coco_data = {
        "info": {
            "description": f"Detections from {Path(model_path).name} on {Path(video_path).name}",
            "date_created": datetime.now().isoformat(),
            "version": "1.0"
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cid, "name": name, "supercategory": "none"}
            for cid, name in class_names.items()
        ]
    }
    
    # Запускаем инференс
    print("\nЗапуск инференса...")
    results = model(
        video_path,
        stream=True,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
        device=device
    )
    
    annotation_id = 0
    frame_count = 0
    detections_count = 0
    last_report = 0
    
    for frame_idx, result in enumerate(results):
        if max_frames and frame_idx >= max_frames:
            break
            
        frame_count += 1
        
        # Добавляем информацию о кадре
        coco_data["images"].append({
            "id": frame_idx,
            "file_name": f"frame_{frame_idx:06d}.jpg",
            "width": width,
            "height": height,
            "frame": frame_idx
        })
        
        # Обрабатываем детекции на этом кадре
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confs = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confs, cls):
                x1, y1, x2, y2 = box
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                # Добавляем аннотацию
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": frame_idx,
                    "category_id": int(class_id),
                    "bbox": [float(x1), float(y1), float(bbox_width), float(bbox_height)],
                    "area": float(bbox_width * bbox_height),
                    "score": float(conf),
                    "iscrowd": 0
                })
                annotation_id += 1
                detections_count += 1
        
        # Прогресс
        if frame_count % 100 == 0 or frame_count == total_frames:
            pct = 100 * frame_count / total_frames
            print(f"  {pct:.1f}% ({frame_count}/{total_frames} кадров) — {detections_count} детекций", end="\r")
    
    print(f"\n\n✅ Обработано {frame_count} кадров")
    print(f"📊 Найдено {detections_count} детекций")
    
    # Сохраняем JSON
    if output_json is None:
        model_name = Path(model_path).stem
        video_name = Path(video_path).stem
        output_json = f"cvat_import_{model_name}_{video_name}.json"
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Сохранено: {output_json}")
    print(f"\n📋 Инструкция по импорту в CVAT:")
    print(f"   1. Создайте новый Task в CVAT")
    print(f"   2. Загрузите видео {Path(video_path).name}")
    print(f"   3. В разделе 'Load annotations' выберите формат 'COCO JSON 1.1'")
    print(f"   4. Загрузите файл {output_json}")
    print(f"   5. Нажмите 'Submit & Continue'")
    
    return output_json


def export_all_models_to_cvat(
    models_dir: str,
    video_path: str,
    output_dir: str = "./cvat_exports",
    conf_threshold: float = 0.5,
    max_frames: Optional[int] = None
) -> list[str]:
    """
    Прогоняет все модели из папки и сохраняет их детекции в отдельные JSON файлы.
    
    Args:
        models_dir: папка с моделями (ищет все .pt файлы)
        video_path: путь к видео
        output_dir: папка для сохранения JSON файлов
        conf_threshold: порог уверенности
        max_frames: ограничить количество кадров
    
    Returns:
        список сохранённых JSON файлов
    """
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ищем все .pt файлы
    model_paths = list(models_dir.rglob("*.pt"))
    # Также можно искать best.pt
    best_paths = list(models_dir.rglob("best.pt"))
    model_paths = list(set(model_paths + best_paths))
    
    if not model_paths:
        print(f"❌ Не найдено моделей в {models_dir}")
        return []
    
    print(f"🔍 Найдено моделей: {len(model_paths)}")
    
    saved_files = []
    for model_path in model_paths:
        print(f"\n{'='*60}")
        print(f"Обработка модели: {model_path.name}")
        print(f"{'='*60}")
        
        output_json = output_dir / f"{model_path.stem}_{Path(video_path).stem}.json"
        
        try:
            json_path = export_detections_to_cvat(
                model_path=str(model_path),
                video_path=video_path,
                output_json=str(output_json),
                conf_threshold=conf_threshold,
                max_frames=max_frames
            )
            saved_files.append(json_path)
        except Exception as e:
            print(f"❌ Ошибка при обработке {model_path.name}: {e}")
    
    return saved_files


def compare_model_detections(
    gt_json_path: str,
    model_json_path: str,
    iou_threshold: float = 0.5
) -> dict:
    """
    Сравнивает детекции модели с Ground Truth (после ручной правки в CVAT).
    
    Args:
        gt_json_path: путь к JSON с размеченным GT (экспорт из CVAT)
        model_json_path: путь к JSON с детекциями модели
        iou_threshold: порог IoU для совпадения
    
    Returns:
        словарь с метриками: precision, recall, f1, tp, fp, fn
    """
    with open(gt_json_path, "r") as f:
        gt_data = json.load(f)
    with open(model_json_path, "r") as f:
        model_data = json.load(f)
    
    # Группируем аннотации по кадрам
    gt_by_frame = {}
    for ann in gt_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in gt_by_frame:
            gt_by_frame[img_id] = []
        gt_by_frame[img_id].append(ann["bbox"])
    
    model_by_frame = {}
    for ann in model_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in model_by_frame:
            model_by_frame[img_id] = []
        model_by_frame[img_id].append(ann["bbox"])
    
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0
    
    tp = fp = fn = 0
    
    for frame_id in set(gt_by_frame.keys()) | set(model_by_frame.keys()):
        gt_boxes = gt_by_frame.get(frame_id, [])
        pred_boxes = model_by_frame.get(frame_id, [])
        
        matched_gt = set()
        matched_pred = set()
        
        for pi, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gi = -1
            for gi, gt_box in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gi)
                matched_pred.add(pi)
            else:
                fp += 1
        
        for gi, _ in enumerate(gt_boxes):
            if gi not in matched_gt:
                fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3)
    }


#!/usr/bin/env python3
"""
Экспорт детекций YOLO/RT-DETR в формат CVAT (COCO JSON).
Затем можно импортировать в CVAT, вручную поправить, и использовать как GT.
"""

import json
import cv2
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from typing import Optional
import torch

def export_detections_to_cvat(
    model_path: str,
    video_path: str,
    output_json: Optional[str] = None,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    class_names: dict = None,
    max_frames: Optional[int] = None,  # для теста: обработать только N кадров
) -> str:
    """
    Запускает модель на видео и сохраняет детекции в формате COCO JSON для CVAT.
    
    Args:
        model_path: путь к модели (.pt, .onnx, .torchscript)
        video_path: путь к видео
        output_json: путь для сохранения JSON (если None — генерируется автоматически)
        conf_threshold: порог уверенности
        iou_threshold: порог NMS
        class_names: словарь {id: name} для классов, если None — использует COCO классы
        max_frames: ограничить количество кадров (для быстрого теста)
    
    Returns:
        путь к сохранённому JSON файлу
    """
    
    # Загружаем модель
    print(f"Загрузка модели: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    
    # Получаем информацию о видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if max_frames and max_frames < total_frames:
        total_frames = max_frames
    
    print(f"Видео: {Path(video_path).name}")
    print(f"  Размер: {width}x{height}, {total_frames} кадров, {fps:.2f} FPS")
    print(f"Устройство: {device}")
    
    # Определяем классы
    if class_names is None:
        # Стандартные классы COCO (person = 0, но для бейджей лучше указать свой)
        # Для детекции бейджей: class_id=0 (badge)
        class_names = {0: "badge"}
    
    # Подготавливаем структуру COCO JSON
    coco_data = {
        "info": {
            "description": f"Detections from {Path(model_path).name} on {Path(video_path).name}",
            "date_created": datetime.now().isoformat(),
            "version": "1.0"
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cid, "name": name, "supercategory": "none"}
            for cid, name in class_names.items()
        ]
    }
    
    # Запускаем инференс
    print("\nЗапуск инференса...")
    results = model(
        video_path,
        stream=True,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
        device=device
    )
    
    annotation_id = 0
    frame_count = 0
    detections_count = 0
    last_report = 0
    
    for frame_idx, result in enumerate(results):
        if max_frames and frame_idx >= max_frames:
            break
            
        frame_count += 1
        
        # Добавляем информацию о кадре
        coco_data["images"].append({
            "id": frame_idx,
            "file_name": f"frame_{frame_idx:06d}.jpg",
            "width": width,
            "height": height,
            "frame": frame_idx
        })
        
        # Обрабатываем детекции на этом кадре
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confs = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confs, cls):
                x1, y1, x2, y2 = box
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                # Добавляем аннотацию
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": frame_idx,
                    "category_id": int(class_id),
                    "bbox": [float(x1), float(y1), float(bbox_width), float(bbox_height)],
                    "area": float(bbox_width * bbox_height),
                    "score": float(conf),
                    "iscrowd": 0
                })
                annotation_id += 1
                detections_count += 1
        
        # Прогресс
        if frame_count % 100 == 0 or frame_count == total_frames:
            pct = 100 * frame_count / total_frames
            print(f"  {pct:.1f}% ({frame_count}/{total_frames} кадров) — {detections_count} детекций", end="\r")
    
    print(f"\n\n✅ Обработано {frame_count} кадров")
    print(f"📊 Найдено {detections_count} детекций")
    
    # Сохраняем JSON
    if output_json is None:
        model_name = Path(model_path).stem
        video_name = Path(video_path).stem
        output_json = f"cvat_import_{model_name}_{video_name}.json"
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Сохранено: {output_json}")
    print(f"\n📋 Инструкция по импорту в CVAT:")
    print(f"   1. Создайте новый Task в CVAT")
    print(f"   2. Загрузите видео {Path(video_path).name}")
    print(f"   3. В разделе 'Load annotations' выберите формат 'COCO JSON 1.1'")
    print(f"   4. Загрузите файл {output_json}")
    print(f"   5. Нажмите 'Submit & Continue'")
    
    return output_json


def export_all_models_to_cvat(
    models_dir: str,
    video_path: str,
    output_dir: str = "./cvat_exports",
    conf_threshold: float = 0.5,
    max_frames: Optional[int] = None
) -> list[str]:
    """
    Прогоняет все модели из папки и сохраняет их детекции в отдельные JSON файлы.
    
    Args:
        models_dir: папка с моделями (ищет все .pt файлы)
        video_path: путь к видео
        output_dir: папка для сохранения JSON файлов
        conf_threshold: порог уверенности
        max_frames: ограничить количество кадров
    
    Returns:
        список сохранённых JSON файлов
    """
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ищем все .pt файлы
    model_paths = list(models_dir.rglob("*.pt"))
    # Также можно искать best.pt
    best_paths = list(models_dir.rglob("best.pt"))
    model_paths = list(set(model_paths + best_paths))
    
    if not model_paths:
        print(f"❌ Не найдено моделей в {models_dir}")
        return []
    
    print(f"🔍 Найдено моделей: {len(model_paths)}")
    
    saved_files = []
    for model_path in model_paths:
        print(f"\n{'='*60}")
        print(f"Обработка модели: {model_path.name}")
        print(f"{'='*60}")
        
        output_json = output_dir / f"{model_path.stem}_{Path(video_path).stem}.json"
        
        try:
            json_path = export_detections_to_cvat(
                model_path=str(model_path),
                video_path=video_path,
                output_json=str(output_json),
                conf_threshold=conf_threshold,
                max_frames=max_frames
            )
            saved_files.append(json_path)
        except Exception as e:
            print(f"❌ Ошибка при обработке {model_path.name}: {e}")
    
    return saved_files


def compare_model_detections(
    gt_json_path: str,
    model_json_path: str,
    iou_threshold: float = 0.5
) -> dict:
    """
    Сравнивает детекции модели с Ground Truth (после ручной правки в CVAT).
    
    Args:
        gt_json_path: путь к JSON с размеченным GT (экспорт из CVAT)
        model_json_path: путь к JSON с детекциями модели
        iou_threshold: порог IoU для совпадения
    
    Returns:
        словарь с метриками: precision, recall, f1, tp, fp, fn
    """
    with open(gt_json_path, "r") as f:
        gt_data = json.load(f)
    with open(model_json_path, "r") as f:
        model_data = json.load(f)
    
    # Группируем аннотации по кадрам
    gt_by_frame = {}
    for ann in gt_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in gt_by_frame:
            gt_by_frame[img_id] = []
        gt_by_frame[img_id].append(ann["bbox"])
    
    model_by_frame = {}
    for ann in model_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in model_by_frame:
            model_by_frame[img_id] = []
        model_by_frame[img_id].append(ann["bbox"])
    
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0
    
    tp = fp = fn = 0
    
    for frame_id in set(gt_by_frame.keys()) | set(model_by_frame.keys()):
        gt_boxes = gt_by_frame.get(frame_id, [])
        pred_boxes = model_by_frame.get(frame_id, [])
        
        matched_gt = set()
        matched_pred = set()
        
        for pi, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gi = -1
            for gi, gt_box in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gi)
                matched_pred.add(pi)
            else:
                fp += 1
        
        for gi, _ in enumerate(gt_boxes):
            if gi not in matched_gt:
                fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3)
    }


def main():
    """Пример использования"""
    
    # ========== КОНФИГУРАЦИЯ ==========
    MODEL_PATH = "../yolo/runs/detect/yolo/runs/rtdetr_l/weights/best.pt"
    VIDEO_PATH = "../dataset/downloaded_videos/test_vids/sample_video.mp4"
    OUTPUT_JSON = "./cvat_export_rtdetr.json"
    CONF_THRESHOLD = 0.5
    MAX_FRAMES = None  # None = всё видео, или число для теста (например 500)
    # ==================================
    
    print("=" * 60)
    print("Экспорт детекций RT-DETR в CVAT формат")
    print("=" * 60)
    
    # 1. Экспортируем детекции RT-DETR
    json_path = export_detections_to_cvat(
        model_path=MODEL_PATH,
        video_path=VIDEO_PATH,
        output_json=OUTPUT_JSON,
        conf_threshold=CONF_THRESHOLD,
        max_frames=MAX_FRAMES
    )
    
    print("\n" + "=" * 60)
    print("Дальнейшие действия:")
    print("=" * 60)
    print("""
    1. Загрузите видео и JSON в CVAT
    2. Вручную исправьте bbox'ы (добавьте пропущенные, удалите лишние)
    3. Экспортируйте из CVAT готовый GT (выберите формат COCO JSON 1.1)
    4. Запустите этот скрипт для сравнения всех моделей с GT:
    
    ```python
    from pathlib import Path
    from export_to_cvat import compare_model_detections
    
    gt_json = "path/to/ground_truth.json"
    
    for model_pt in Path("../yolo/runs/detect/yolo/runs").rglob("best.pt"):
        model_json = f"detections_{model_pt.parent.parent.name}.json"
        export_detections_to_cvat(str(model_pt), VIDEO_PATH, model_json)
        
        metrics = compare_model_detections(gt_json, model_json)
        print(f"{model_pt.parent.parent.name}: {metrics}")
""")