#!/usr/bin/env python3
"""
Шаг 1 — Бенчмарк детекции людей (без трекера) + GT evaluation.

GT берётся из cvat_exports_persons/<persons_<video_stem>.json> (COCO JSON).
Если GT нет — считаются только метрики производительности.

Метрики:
- FPS, latency avg/p50/p95/p99
- VRAM, RAM
- persons_avg, persons_max, frames_with_person_pct
- Precision, Recall, F1 (если есть GT)
- TP, FP, FN (если есть GT)
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
GT_DIR           = "./cvat_exports_persons"   # папка с persons_<stem>.json
OUTPUT_DIR       = "./person_detection_results"
OUTPUT_VIDEO_DIR = "./person_detection_videos"

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
IOU_THRESHOLD  = 0.5
PERSON_CLASS   = 0
WARMUP_FRAMES  = 5
SAVE_VIDEO     = True

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


def load_gt(gt_dir: str, video_stem: str) -> dict | None:
    """
    Ищет GT файл: <gt_dir>/persons_<video_stem>.json
    Возвращает dict {frame_id: [[x,y,w,h], ...]} или None если не найден.
    """
    gt_path = Path(gt_dir) / f"persons_{video_stem}.json"
    if not gt_path.exists():
        return None

    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_by_frame: dict[int, list] = {}
    for ann in data.get("annotations", []):
        fid = ann["image_id"]
        gt_by_frame.setdefault(fid, []).append(ann["bbox"])  # [x, y, w, h]

    return gt_by_frame


def calculate_iou(box1: list, box2: list) -> float:
    """IoU для боксов в формате [x, y, w, h]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return inter / union if union > 0 else 0.0


def evaluate(predictions: dict, gt_data: dict, iou_thr: float) -> dict:
    """
    predictions: {frame_id: [[x,y,w,h], ...]}
    gt_data:     {frame_id: [[x,y,w,h], ...]}
    """
    tp = fp = fn = 0

    all_frames = set(predictions) | set(gt_data)

    for fid in all_frames:
        preds  = predictions.get(fid, [])
        gts    = gt_data.get(fid, [])
        matched = set()

        for pred in preds:
            best_iou = 0.0
            best_idx = -1
            for gi, gt in enumerate(gts):
                if gi in matched:
                    continue
                iou = calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi

            if best_iou >= iou_thr:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1

        fn += len(gts) - len(matched)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if precision + recall > 0 else 0.0)

    return {
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


def make_video_writer(path: str, w: int, h: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {path}")
    return writer


def run_person_detector(model_name: str, model_label: str, video_path: str) -> dict:
    video_name = Path(video_path).name
    video_stem = Path(video_path).stem

    print("\n" + "=" * 65)
    print(f"{model_label}  ->  {video_name}")

    # ── GT ────────────────────────────────────────────────────
    gt_data = load_gt(GT_DIR, video_stem)
    if gt_data is not None:
        total_gt = sum(len(v) for v in gt_data.values())
        print(f"GT loaded: {total_gt} annotations across {len(gt_data)} frames")
    else:
        print("GT not found — only performance metrics will be computed")

    # ── Выходное видео ────────────────────────────────────────
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
        predictions:       dict        = {}   # {frame_id: [[x,y,w,h],...]}

        frame_count = 0
        last_report = time.time()
        t0          = time.time()

        for result in model(
            video_path, stream=True,
            conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            classes=[PERSON_CLASS], verbose=False,
        ):
            spd = result.speed
            frame_times_ms.append(
                spd.get("preprocess",  0)
                + spd.get("inference",  0)
                + spd.get("postprocess", 0)
            )

            # Собираем боксы для GT evaluation
            predictions[frame_count] = []
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    predictions[frame_count].append(
                        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    )

            n_persons = len(predictions[frame_count])
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

        # ── GT evaluation ─────────────────────────────────────
        eval_metrics: dict = {}
        if gt_data is not None:
            eval_metrics = evaluate(predictions, gt_data, IOU_THRESHOLD)
            print(
                f"P={eval_metrics['precision']:.3f} | "
                f"R={eval_metrics['recall']:.3f} | "
                f"F1={eval_metrics['f1']:.3f} | "
                f"TP={eval_metrics['tp']} FP={eval_metrics['fp']} FN={eval_metrics['fn']}"
            )

        metrics = {
            "model":                  model_label,
            "video_name":             video_name,
            # производительность
            "fps":                    round(fps, 2),
            "load_time_ms":           round(load_ms, 1),
            "frame_time_avg_ms":      round(float(np.mean(arr)),           2),
            "frame_time_p50_ms":      round(float(np.percentile(arr, 50)), 2),
            "frame_time_p95_ms":      round(float(np.percentile(arr, 95)), 2),
            "frame_time_p99_ms":      round(float(np.percentile(arr, 99)), 2),
            "vram_peak_mb":           round(vram_peak, 1),
            "ram_delta_mb":           round(ram_delta, 1),
            "total_frames":           frame_count,
            # детекции
            "persons_avg":            round(float(np.mean(ppa)),           2),
            "persons_max":            int(np.max(ppa)),
            "frames_with_person_pct": round(100 * float(np.mean(ppa > 0)), 1),
            # GT метрики (пустые если GT нет)
            **eval_metrics,
            "gt_available":           gt_data is not None,
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

    # JSON — всё включая ошибки
    json_path = out / f"person_benchmark_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {json_path}")

    successful = [r for r in results if "fps" in r]
    if not successful:
        print("No successful results for CSV")
        return

    csv_path = out / f"person_benchmark_{ts}.csv"
    fields = [
        "model", "video_name",
        "fps", "load_time_ms",
        "frame_time_avg_ms", "frame_time_p50_ms",
        "frame_time_p95_ms", "frame_time_p99_ms",
        "vram_peak_mb", "ram_delta_mb", "total_frames",
        "persons_avg", "persons_max", "frames_with_person_pct",
        "precision", "recall", "f1", "tp", "fp", "fn",
        "gt_available",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(successful)
    print(f"CSV:  {csv_path}")

    # ── Таблица производительности ────────────────────────────
    W = 105
    print(f"\n{'='*W}")
    print("PERFORMANCE — sorted by FPS")
    print(f"{'='*W}")
    print(
        f"{'Model':<20} {'Video':<16} {'FPS':>7} "
        f"{'avg ms':>8} {'p95 ms':>8} {'VRAM MB':>8} "
        f"{'pers avg':>9} {'w/person%':>10}"
    )
    print("-" * W)
    for r in sorted(successful, key=lambda x: x["fps"], reverse=True):
        print(
            f"{r['model'][:20]:<20} {r['video_name'][:16]:<16} {r['fps']:>7.1f} "
            f"{r['frame_time_avg_ms']:>8.1f} {r['frame_time_p95_ms']:>8.1f} "
            f"{r['vram_peak_mb']:>8.0f} "
            f"{r['persons_avg']:>9.1f} "
            f"{r['frames_with_person_pct']:>9.1f}%"
        )

    # ── Таблица качества (только где есть GT) ─────────────────
    with_gt = [r for r in successful if r.get("gt_available")]
    if with_gt:
        print(f"\n{'='*W}")
        print("QUALITY (GT available) — sorted by F1")
        print(f"{'='*W}")
        print(
            f"{'Model':<20} {'Video':<16} "
            f"{'Precision':>10} {'Recall':>8} {'F1':>8} "
            f"{'TP':>7} {'FP':>7} {'FN':>7}"
        )
        print("-" * W)
        for r in sorted(with_gt, key=lambda x: x.get("f1", 0), reverse=True):
            print(
                f"{r['model'][:20]:<20} {r['video_name'][:16]:<16} "
                f"{r.get('precision', 0):>10.3f} {r.get('recall', 0):>8.3f} "
                f"{r.get('f1', 0):>8.3f} "
                f"{r.get('tp', 0):>7} {r.get('fp', 0):>7} {r.get('fn', 0):>7}"
            )

    # ── Ошибки ────────────────────────────────────────────────
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\nErrors: {len(errors)}")
        for r in errors:
            print(f"  {r['model']} / {r['video_name']}: {r['error']}")


def main() -> None:
    print("=" * 65)
    print("Person Detection Benchmark (no tracker)")
    print("=" * 65)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU:  {gpu.name}  {gpu.total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available — using CPU")

    print(f"Conf: {CONF_THRESHOLD}  IoU: {IOU_THRESHOLD}")
    print(f"Save video: {'YES' if SAVE_VIDEO else 'NO'}")

    gt_exists = Path(GT_DIR).exists()
    print(f"GT dir: {GT_DIR}  ({'found' if gt_exists else 'NOT FOUND — quality metrics skipped'})")

    if not Path(VIDEO_DIR).exists():
        print(f"VIDEO DIR NOT FOUND: {VIDEO_DIR}")
        sys.exit(1)

    videos = find_all_videos(VIDEO_DIR)
    if not videos:
        print(f"No videos found in {VIDEO_DIR}")
        sys.exit(1)

    print(f"\nVideos: {len(videos)}")
    for v in videos:
        stem = Path(v).stem
        gt_path = Path(GT_DIR) / f"persons_{stem}.json"
        gt_mark = "✓ GT" if gt_path.exists() else "✗ no GT"
        print(f"  {Path(v).name}  [{gt_mark}]")

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
        model_label = Path(model_name).stem
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