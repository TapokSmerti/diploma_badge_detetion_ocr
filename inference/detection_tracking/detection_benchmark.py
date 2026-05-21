#!/usr/bin/env python3
"""
Детекция людей + экспорт в COCO JSON для CVAT.
По аналогии с экспортом бейджиков.

Использует претренированную yolo12m.pt (COCO weights).
Детектирует только людей (class 0).

Структура вывода:
    cvat_exports_persons/
        persons_vid1.json
        persons_vid2.json
        ...
"""
from __future__ import annotations

import json
import sys
import cv2
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

# yolo12m — хороший баланс точности и скорости.
# Если хочешь максимум точности — замени на "yolo12l.pt" или "yolov9m.pt"
MODEL_NAME   = "yolo12m.pt"

VIDEO_DIR    = "../../dataset/downloaded_videos/test_vids"
OUTPUT_DIR   = "./cvat_exports_persons"

CONF         = 0.5
PERSON_CLASS = 0    # class 0 = person в COCO

# ============================================================


def find_all_videos(video_dir: str) -> list[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sorted([
        p for p in Path(video_dir).rglob("*")
        if p.suffix.lower() in exts
    ])


def export_persons_to_cvat(
    model: YOLO,
    video_path: Path,
    output_json: Path,
) -> int:
    """
    Прогоняет модель на одном видео, сохраняет детекции людей в COCO JSON.
    Возвращает кол-во детекций.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"  {video_path.name}  {width}x{height}  {total_frames} frames  {fps:.1f} FPS")

    # COCO JSON структура — category_id начинается с 1
    coco = {
        "info": {
            "description": f"Person detections from {MODEL_NAME} on {video_path.name}",
            "date_created": datetime.now().isoformat(),
            "version": "1.0",
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"}],
    }

    ann_id         = 0
    total_detected = 0

    for frame_idx, result in enumerate(model(
        str(video_path),
        stream=True,
        conf=CONF,
        classes=[PERSON_CLASS],
        verbose=False,
    )):
        coco["images"].append({
            "id":        frame_idx,
            "file_name": f"frame_{frame_idx:06d}.jpg",
            "width":     width,
            "height":    height,
            "frame":     frame_idx,
        })

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box
                bw = float(x2 - x1)
                bh = float(y2 - y1)

                coco["annotations"].append({
                    "id":          ann_id,
                    "image_id":    frame_idx,
                    "category_id": 1,          # 1, не 0 — требование COCO/CVAT
                    "bbox":        [float(x1), float(y1), bw, bh],
                    "area":        bw * bh,
                    "score":       float(conf),
                    "iscrowd":     0,
                })
                ann_id         += 1
                total_detected += 1

        # Прогресс каждые 500 кадров
        if (frame_idx + 1) % 500 == 0 or (frame_idx + 1) == total_frames:
            pct = 100 * (frame_idx + 1) / total_frames
            print(f"    {pct:.1f}%  ({frame_idx + 1}/{total_frames})  "
                  f"{total_detected} detections", end="\r")

    print()  # перенос строки после \r

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    print(f"  ✅ {total_detected} detections → {output_json.name}")
    return total_detected


def main() -> None:
    print("=" * 60)
    print("Person Detection → CVAT Export")
    print("=" * 60)
    print(f"Model:    {MODEL_NAME}")
    print(f"Videos:   {VIDEO_DIR}")
    print(f"Output:   {OUTPUT_DIR}")
    print(f"Conf:     {CONF}")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU:      {gpu.name}  {gpu.total_memory / 1024**3:.1f} GB")
    else:
        print("GPU:      not available — using CPU")

    if not Path(VIDEO_DIR).exists():
        print(f"\nERROR: VIDEO DIR not found: {VIDEO_DIR}")
        sys.exit(1)

    videos = find_all_videos(VIDEO_DIR)
    if not videos:
        print(f"ERROR: no videos found in {VIDEO_DIR}")
        sys.exit(1)

    print(f"\nVideos found: {len(videos)}")
    for v in videos:
        print(f"  {v.name}  ({v.stat().st_size / 1024 / 1024:.1f} MB)")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Загружаем модель один раз для всех видео
    print(f"\nLoading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    print("Model loaded")

    saved = []
    errors = []

    print(f"\nProcessing {len(videos)} videos...\n")

    for i, video_path in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}]")
        output_json = Path(OUTPUT_DIR) / f"persons_{video_path.stem}.json"

        try:
            n = export_persons_to_cvat(model, video_path, output_json)
            saved.append((output_json, n))
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((video_path.name, str(e)))

    print(f"\n{'='*60}")
    print(f"Done: {len(saved)}/{len(videos)} videos processed")
    print(f"Output: {Path(OUTPUT_DIR).absolute()}")

    if saved:
        print("\nFiles for CVAT import:")
        for path, n in saved:
            print(f"  {path.name}  ({n} detections)")

        print("\nNext steps:")
        print("  1. Create Task in CVAT, upload video")
        print("  2. Import JSON (format: COCO JSON 1.1)")
        print("  3. Fix annotations manually")
        print("  4. Export GT from CVAT")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()