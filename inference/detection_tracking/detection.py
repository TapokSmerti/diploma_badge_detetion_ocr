#!/usr/bin/env python3
"""
Шаг 1 — Бенчмарк детекции людей (без трекера).

Использует претренированные COCO веса (ultralytics скачает автоматически).
Детектирует только людей (class 0), собирает метрики:
- FPS, latency avg/p50/p95/p99
- VRAM, RAM
- среднее/макс кол-во людей на кадр
- % кадров где найден хотя бы один человек
- сохраняет видео с bbox

Используется для ответа на вопрос:
"Стоит ли использовать детекцию людей + ROI для инференса бейджей?"
"""
from __future__ import annotations

import gc
import csv
import json
import time
import sys
from pathlib import Path
from datetime import datetime

import cv2
import torch
import psutil
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

VIDEO_DIR        = "../dataset/downloaded_videos/test_vids"
OUTPUT_DIR       = "./person_detection_results"
OUTPUT_VIDEO_DIR = "./person_detection_videos"

# Претренированные COCO модели.
# Ultralytics скачает их автоматически при первом запуске.
PERSON_MODELS = [
    # YOLOv5
    "yolov5nu.pt",
    "yolov5su.pt",
    "yolov5mu.pt",
    # YOLOv8
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    # YOLOv9
    "yolov9s.pt",
    "yolov9m.pt",
    # YOLOv10
    "yolov10n.pt",
    "yolov10s.pt",
    "yolov10m.pt",
    # YOLO11
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    # YOLO12
    "yolo12n.pt",
    "yolo12s.pt",
    "yolo12m.pt",
    # RT-DETR
    "rtdetr-l.pt",
]

CONF_THRESHOLD = 0.5
IOU_THRESHOLD  = 0.45
PERSON_CLASS   = 0       # class 0 = person в COCO
WARMUP_FRAMES  = 5
SAVE_VIDEO     = True    # False — быстрее, без записи

# ============================================================


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def find_all_videos(video_dir: str) -> list[str]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sorted([
        str(p) for p in Path(video_dir).rglob("*")
        if p.suffix.lower() in exts
    ])


def make_video_writer(path: str, w: int, h: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {path}")
    return writer


def run_person_detector(model_name: str, model_label: str, video_path: str) -> dict:
    """
    Запускает один детектор на одном видео, детектирует только людей.
    model_name  — имя/путь для YOLO() напр. "yolov8n.pt"
    model_label — читаемое имя для CSV/папок напр. "yolov8n"
    """
    video_name = Path(video_path).name
    video_stem = Path(video_path).stem

    print("\n" + "=" * 60)
    print(f"{model_label} -> {video_name}")

    # ── Путь для выходного видео ──────────────────────────────
    output_video_path: str | None = None
    if SAVE_VIDEO:
        out_dir = Path(OUTPUT_VIDEO_DIR) / model_label
        out_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = str(out_dir / f"{video_stem}.mp4")

    writer: cv2.VideoWriter | None = None

    try:
        # ── Загрузка ──────────────────────────────────────────
        torch.cuda.empty_cache()
        gc.collect()

        ram_before = psutil.Process().memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_load  = time.perf_counter()
        model   = YOLO(model_name)
        load_ms = (time.perf_counter() - t_load) * 1000
        print(f"Loaded: {load_ms:.0f} ms")

        # ── Параметры видео ───────────────────────────────────
        cap          = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        # ── Прогрев ───────────────────────────────────────────
        print(f"Warmup ({WARMUP_FRAMES} frames)...", end=" ", flush=True)
        done = 0
        for _ in model(
            video_path, stream=True,
            conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            classes=[PERSON_CLASS], verbose=False,
        ):
            done += 1
            if done >= WARMUP_FRAMES:
                break
        print("done")

        # ── Инференс ──────────────────────────────────────────
        print(f"Inference ({total_frames} frames)...")

        frame_times_ms:    list[float] = []
        persons_per_frame: list[int]   = []
        frame_count = 0
        last_report = time.time()
        t0          = time.time()

        for result in model(
            video_path, stream=True,
            conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            classes=[PERSON_CLASS], verbose=False,
        ):
            spd = result.speed
            frame_ms = (
                spd.get("preprocess",  0)
                + spd.get("inference",  0)
                + spd.get("postprocess", 0)
            )
            frame_times_ms.append(frame_ms)

            n_persons = len(result.boxes) if result.boxes is not None else 0
            persons_per_frame.append(n_persons)

            frame_count += 1

            # Видео с bbox
            if output_video_path is not None:
                annotated = result.plot()
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = make_video_writer(output_video_path, w, h, video_fps)
                writer.write(annotated)

            # Прогресс каждые 5 секунд
            now = time.time()
            if now - last_report >= 5.0:
                elapsed    = now - t0
                fps_so_far = frame_count / elapsed if elapsed > 0 else 0
                eta        = (total_frames - frame_count) / fps_so_far if fps_so_far > 0 else 0
                print(
                    f"  {100 * frame_count / total_frames:.1f}% | "
                    f"{frame_count}/{total_frames} | "
                    f"{fps_so_far:.1f} FPS | "
                    f"ETA {format_time(eta)}"
                )
                last_report = now

        if writer is not None:
            writer.release()
            writer = None
            print(f"Video saved: {output_video_path}")

        total_time = time.time() - t0
        fps        = frame_count / total_time if total_time > 0 else 0
        vram_peak  = (
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if torch.cuda.is_available() else 0
        )
        ram_delta = psutil.Process().memory_info().rss / 1024 / 1024 - ram_before

        arr = np.array(frame_times_ms) if frame_times_ms else np.array([0.0])
        ppa = np.array(persons_per_frame) if persons_per_frame else np.array([0])

        metrics = {
            "model":                  model_label,
            "video_name":             video_name,
            "fps":                    round(fps, 2),
            "load_time_ms":           round(load_ms, 1),
            "frame_time_avg_ms":      round(float(np.mean(arr)),           2),
            "frame_time_p50_ms":      round(float(np.percentile(arr, 50)), 2),
            "frame_time_p95_ms":      round(float(np.percentile(arr, 95)), 2),
            "frame_time_p99_ms":      round(float(np.percentile(arr, 99)), 2),
            "vram_peak_mb":           round(vram_peak, 1),
            "ram_delta_mb":           round(ram_delta, 1),
            "total_frames":           frame_count,
            "persons_avg":            round(float(np.mean(ppa)),           2),
            "persons_max":            int(np.max(ppa)),
            "frames_with_person_pct": round(100 * float(np.mean(ppa > 0)), 1),
            "output_video":           output_video_path,
            "timestamp":              datetime.now().isoformat(),
        }

        print(
            f"✓ FPS: {fps:.1f} | "
            f"avg: {metrics['frame_time_avg_ms']}ms | "
            f"p95: {metrics['frame_time_p95_ms']}ms | "
            f"VRAM: {vram_peak:.0f}MB | "
            f"persons avg: {metrics['persons_avg']:.1f} | "
            f"w/person: {metrics['frames_with_person_pct']}%"
        )
        return metrics

    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "model":      model_label,
            "video_name": video_name,
            "error":      str(e),
            "timestamp":  datetime.now().isoformat(),
        }

    finally:
        if writer is not None:
            writer.release()
        try:
            del model
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()


def save_results(results: list[dict]) -> None:
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out / f"person_detection_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {json_path}")

    successful = [r for r in results if "fps" in r]
    if not successful:
        print("No successful results for CSV")
        return

    csv_path = out / f"person_detection_{ts}.csv"
    fields = [
        "model", "video_name",
        "fps", "load_time_ms",
        "frame_time_avg_ms", "frame_time_p50_ms",
        "frame_time_p95_ms", "frame_time_p99_ms",
        "vram_peak_mb", "ram_delta_mb",
        "total_frames",
        "persons_avg", "persons_max", "frames_with_person_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(successful)
    print(f"CSV:  {csv_path}")

    W = 105
    print(f"\n{'='*W}")
    print("RESULTS — sorted by FPS")
    print(f"{'='*W}")
    print(
        f"{'Model':<20} {'Video':<16} {'FPS':>7} "
        f"{'avg ms':>8} {'p95 ms':>8} {'VRAM MB':>8} "
        f"{'pers avg':>9} {'pers max':>9} {'w/person%':>10}"
    )
    print("-" * W)

    for r in sorted(successful, key=lambda x: x["fps"], reverse=True):
        print(
            f"{r['model'][:20]:<20} {r['video_name'][:16]:<16} {r['fps']:>7.1f} "
            f"{r['frame_time_avg_ms']:>8.1f} {r['frame_time_p95_ms']:>8.1f} "
            f"{r['vram_peak_mb']:>8.0f} "
            f"{r['persons_avg']:>9.1f} {r['persons_max']:>9} "
            f"{r['frames_with_person_pct']:>9.1f}%"
        )

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\nErrors: {len(errors)}")
        for r in errors:
            print(f"  {r['model']} / {r['video_name']}: {r['error']}")


def main() -> None:
    print("=" * 60)
    print("Step 1 — Person Detection Benchmark (no tracker)")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU:  {gpu.name}")
        print(f"VRAM: {gpu.total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available — using CPU")

    print(f"Save video: {'YES' if SAVE_VIDEO else 'NO'}")
    print(f"Person class: {PERSON_CLASS} (COCO)")
    print(f"Conf: {CONF_THRESHOLD}  IoU: {IOU_THRESHOLD}")

    if not Path(VIDEO_DIR).exists():
        print(f"VIDEO DIR NOT FOUND: {VIDEO_DIR}")
        sys.exit(1)

    videos = find_all_videos(VIDEO_DIR)
    if not videos:
        print(f"No videos found in {VIDEO_DIR}")
        sys.exit(1)

    print(f"\nVideos: {len(videos)}")
    for v in videos:
        print(f"  {Path(v).name}")

    print(f"\nModels: {len(PERSON_MODELS)}")
    for m in PERSON_MODELS:
        print(f"  {m}")

    total_runs = len(PERSON_MODELS) * len(videos)
    print(f"\nTotal runs: {len(PERSON_MODELS)} models × {len(videos)} videos = {total_runs}")
    print("\nContinue? (y/n): ", end="")
    if input().strip().lower() != "y":
        sys.exit(0)

    results:  list[dict] = []
    t_global = time.time()
    run_idx  = 0

    for model_name in PERSON_MODELS:
        model_label = Path(model_name).stem   # "yolov8n.pt" → "yolov8n"
        for video_path in videos:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}]")

            result = run_person_detector(model_name, model_label, video_path)
            results.append(result)

            elapsed   = time.time() - t_global
            remaining = total_runs - run_idx
            if remaining > 0:
                eta = elapsed / run_idx * remaining
                print(f"Progress: {run_idx}/{total_runs} | ETA {format_time(eta)}")

    save_results(results)
    print("\nDONE")


if __name__ == "__main__":
    main()