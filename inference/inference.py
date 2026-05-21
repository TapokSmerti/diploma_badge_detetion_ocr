#!/usr/bin/env python3
"""
YOLO / RT-DETR Benchmark + GT Evaluation

Features:
- Benchmark all YOLO/RT-DETR models
- Benchmark all videos in folder
- Save annotated videos
- Save benchmark CSV/JSON
- Evaluate against GT COCO JSON
- Precision / Recall / F1
- FPS / latency / VRAM

Dataset structure:

test_vids/
├── vid1.mp4
├── vid1.json
├── vid2.mp4
├── vid2.json
"""

from __future__ import annotations

import os
import sys
import gc
import csv
import json
import time
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

VIDEO_DIR = "../dataset/downloaded_videos/test_vids"

MODELS_ROOT = "../yolo/runs/detect/yolo/runs"

OUTPUT_DIR = "./benchmark_results"

OUTPUT_VIDEO_DIR = "./benchmark_results_videos"

CONF_THRESHOLD = 0.5

IOU_THRESHOLD = 0.5

SAVE_VIDEO = True

WARMUP_FRAMES = 5

# ============================================================


def format_time(seconds: float) -> str:

    if seconds < 60:
        return f"{seconds:.1f}s"

    if seconds < 3600:
        return f"{seconds / 60:.1f}m"

    return f"{seconds / 3600:.1f}h"


def find_all_videos(video_dir: str):

    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    return sorted([
        str(p)
        for p in Path(video_dir).rglob("*")
        if p.suffix.lower() in exts
    ])


def find_all_models(models_root: str):

    models = []

    for p in Path(models_root).rglob("best.*"):

        if p.suffix not in {".pt", ".onnx", ".torchscript"}:
            continue

        models.append({
            "path": str(p),
            "name": p.parent.parent.name,
            "format": p.suffix[1:],
            "size_mb": p.stat().st_size / 1024 / 1024,
        })

    return models


def check_onnx_cuda():

    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()

    except ImportError:
        return False


def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])

    y1 = max(box1[1], box2[1])

    x2 = min(box1[0] + box1[2], box2[0] + box2[2])

    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    area1 = box1[2] * box1[3]

    area2 = box2[2] * box2[3]

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def load_gt(gt_json_path: str):

    with open(gt_json_path, "r", encoding="utf-8") as f:

        data = json.load(f)

    gt_by_frame = {}

    for ann in data.get("annotations", []):

        frame_id = ann["image_id"]

        if frame_id not in gt_by_frame:
            gt_by_frame[frame_id] = []

        gt_by_frame[frame_id].append(ann["bbox"])

    return gt_by_frame


def evaluate_predictions(
    predictions: dict,
    gt_data: dict,
    iou_threshold: float = 0.5
):

    tp = 0
    fp = 0
    fn = 0

    frame_ids = set(predictions.keys()) | set(gt_data.keys())

    for frame_id in frame_ids:

        pred_boxes = predictions.get(frame_id, [])

        gt_boxes = gt_data.get(frame_id, [])

        matched_gt = set()

        for pred_box in pred_boxes:

            best_iou = 0

            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):

                if gt_idx in matched_gt:
                    continue

                iou = calculate_iou(pred_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:

                tp += 1

                matched_gt.add(best_gt_idx)

            else:

                fp += 1

        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if tp + fp > 0 else 0

    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def make_video_writer(output_path, width, height, fps):

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {output_path}")

    return writer


def run_model(model_info: dict, video_path: str):

    model_path = model_info["path"]

    model_name = model_info["name"]

    model_format = model_info["format"]

    video_name = Path(video_path).name

    video_stem = Path(video_path).stem

    print("\n" + "=" * 60)

    print(f"{model_name} [{model_format}] -> {video_name}")

    # ============================================================
    # GT
    # ============================================================

    gt_json_path = str(Path(video_path).with_suffix(".json"))

    gt_data = None

    if os.path.exists(gt_json_path):

        try:

            gt_data = load_gt(gt_json_path)

            print(f"GT loaded: {Path(gt_json_path).name}")

        except Exception as e:

            print(f"GT ERROR: {e}")

    else:

        print("GT NOT FOUND")

    # ============================================================
    # OUTPUT VIDEO
    # ============================================================

    output_video_path = None

    if SAVE_VIDEO:

        out_dir = Path(OUTPUT_VIDEO_DIR) / model_name

        out_dir.mkdir(parents=True, exist_ok=True)

        output_video_path = str(
            out_dir / f"{video_stem}__{model_format}.mp4"
        )

    # ============================================================

    try:

        torch.cuda.empty_cache()

        gc.collect()

        ram_before = psutil.Process().memory_info().rss / 1024 / 1024

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_load = time.perf_counter()

        model = YOLO(model_path)

        load_ms = (time.perf_counter() - t_load) * 1000

        print(f"Model loaded: {load_ms:.0f} ms")

        # ============================================================

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        # ============================================================
        # WARMUP
        # ============================================================

        warmup = 0

        for _ in model(
            video_path,
            stream=True,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        ):

            warmup += 1

            if warmup >= WARMUP_FRAMES:
                break

        # ============================================================

        predictions = {}

        frame_times = []

        frame_count = 0

        writer = None

        t0 = time.time()

        last_report = time.time()

        # ============================================================
        # INFERENCE
        # ============================================================

        for result in model(
            video_path,
            stream=True,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        ):

            predictions[frame_count] = []

            if result.boxes is not None:

                boxes = result.boxes.xyxy.cpu().numpy()

                confs = result.boxes.conf.cpu().numpy()

                classes = result.boxes.cls.cpu().numpy()

                for box, conf, cls in zip(boxes, confs, classes):

                    x1, y1, x2, y2 = box

                    w = x2 - x1

                    h = y2 - y1

                    predictions[frame_count].append([
                        float(x1),
                        float(y1),
                        float(w),
                        float(h)
                    ])

            # ============================================================
            # SAVE VIDEO
            # ============================================================

            if output_video_path is not None:

                annotated_frame = result.plot()

                if writer is None:

                    h, w = annotated_frame.shape[:2]

                    writer = make_video_writer(
                        output_video_path,
                        w,
                        h,
                        video_fps
                    )

                writer.write(annotated_frame)

            # ============================================================

            speed = result.speed

            frame_ms = (
                speed.get("preprocess", 0)
                + speed.get("inference", 0)
                + speed.get("postprocess", 0)
            )

            frame_times.append(frame_ms)

            frame_count += 1

            # ============================================================
            # PROGRESS
            # ============================================================

            now = time.time()

            if now - last_report >= 5:

                elapsed = now - t0

                fps_now = frame_count / elapsed

                eta = (
                    (total_frames - frame_count) / fps_now
                    if fps_now > 0 else 0
                )

                print(
                    f"{100 * frame_count / total_frames:.1f}% | "
                    f"{frame_count}/{total_frames} | "
                    f"{fps_now:.1f} FPS | "
                    f"ETA {format_time(eta)}"
                )

                last_report = now

        # ============================================================

        if writer is not None:

            writer.release()

            print(f"Video saved: {output_video_path}")

        total_time = time.time() - t0

        fps = frame_count / total_time if total_time > 0 else 0

        # ============================================================
        # GT EVAL
        # ============================================================

        eval_metrics = {}

        if gt_data is not None:

            eval_metrics = evaluate_predictions(
                predictions,
                gt_data,
                iou_threshold=IOU_THRESHOLD
            )

            print(
                f"P={eval_metrics['precision']:.3f} | "
                f"R={eval_metrics['recall']:.3f} | "
                f"F1={eval_metrics['f1']:.3f}"
            )

        # ============================================================

        metrics = {
            "model_name": model_name,

            "video_name": video_name,

            "format": model_format,

            "fps": round(fps, 2),

            "load_time_ms": round(load_ms, 1),

            "frame_time_avg_ms": round(
                float(np.mean(frame_times)),
                2
            ),

            "frame_time_p50_ms": round(
                float(np.percentile(frame_times, 50)),
                2
            ),

            "frame_time_p95_ms": round(
                float(np.percentile(frame_times, 95)),
                2
            ),

            "frame_time_p99_ms": round(
                float(np.percentile(frame_times, 99)),
                2
            ),

            "vram_peak_mb": round(
                torch.cuda.max_memory_allocated() / 1024 / 1024,
                1
            ) if torch.cuda.is_available() else 0,

            "ram_delta_mb": round(
                psutil.Process().memory_info().rss / 1024 / 1024 - ram_before,
                1
            ),

            "total_frames": frame_count,

            "output_video": output_video_path,

            "timestamp": datetime.now().isoformat(),

            **eval_metrics,
        }

        return metrics

    except Exception as e:

        print(f"ERROR: {e}")

        return {
            "model_name": model_name,
            "video_name": video_name,
            "error": str(e),
        }

    finally:

        try:
            del model
        except Exception:
            pass

        torch.cuda.empty_cache()

        gc.collect()


def save_results(results, output_dir):

    out = Path(output_dir)

    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ============================================================
    # JSON
    # ============================================================

    json_path = out / f"benchmark_{ts}.json"

    with open(json_path, "w", encoding="utf-8") as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"JSON saved: {json_path}")

    # ============================================================
    # CSV
    # ============================================================

    successful = [r for r in results if "fps" in r]

    if not successful:
        return

    csv_path = out / f"benchmark_{ts}.csv"

    fields = [
        "model_name",
        "video_name",
        "format",

        "fps",

        "precision",
        "recall",
        "f1",

        "tp",
        "fp",
        "fn",

        "frame_time_avg_ms",
        "frame_time_p50_ms",
        "frame_time_p95_ms",
        "frame_time_p99_ms",

        "vram_peak_mb",

        "ram_delta_mb",

        "total_frames",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore"
        )

        writer.writeheader()

        writer.writerows(successful)

    print(f"CSV saved: {csv_path}")

    # ============================================================
    # CONSOLE TABLE
    # ============================================================

    print("\n" + "=" * 100)

    print("RESULTS")

    print("=" * 100)

    for r in sorted(successful, key=lambda x: x["fps"], reverse=True):

        print(
            f"{r['model_name']:<20} "
            f"{r['video_name']:<15} "
            f"{r['fps']:>7.1f} FPS | "
            f"P={r.get('precision',0):.3f} | "
            f"R={r.get('recall',0):.3f} | "
            f"F1={r.get('f1',0):.3f} | "
            f"VRAM={r['vram_peak_mb']:.0f}MB"
        )


def main():

    print("=" * 60)

    print("YOLO Benchmark + GT Evaluation")

    print("=" * 60)

    # ============================================================

    if torch.cuda.is_available():

        gpu = torch.cuda.get_device_properties(0)

        print(f"GPU: {gpu.name}")

        print(f"VRAM: {gpu.total_memory / 1024**3:.1f} GB")

    else:

        print("CUDA NOT AVAILABLE")

    # ============================================================

    onnx_cuda = check_onnx_cuda()

    print(f"ONNX CUDA: {'YES' if onnx_cuda else 'NO'}")

    print(f"SAVE VIDEO: {'YES' if SAVE_VIDEO else 'NO'}")

    # ============================================================

    for path, label in [
        (VIDEO_DIR, "VIDEO DIR"),
        (MODELS_ROOT, "MODELS DIR")
    ]:

        if not os.path.exists(path):

            print(f"{label} NOT FOUND: {path}")

            sys.exit(1)

    # ============================================================

    videos = find_all_videos(VIDEO_DIR)

    models = find_all_models(MODELS_ROOT)

    # ============================================================

    print(f"\nVIDEOS: {len(videos)}")

    for v in videos:
        print(f"  {Path(v).name}")

    print(f"\nMODELS: {len(models)}")

    for m in models:
        print(f"  {m['name']} [{m['format']}]")

    total_runs = len(videos) * len(models)

    print(f"\nTOTAL RUNS: {total_runs}")

    print("\nContinue? (y/n): ", end="")

    if input().strip().lower() != "y":

        sys.exit(0)

    # ============================================================

    results = []

    run_idx = 0

    t0 = time.time()

    for model_info in models:

        for video_path in videos:

            run_idx += 1

            print(f"\n[{run_idx}/{total_runs}]")

            result = run_model(
                model_info,
                video_path
            )

            results.append(result)

            elapsed = time.time() - t0

            remaining = total_runs - run_idx

            if remaining > 0:

                eta = elapsed / run_idx * remaining

                print(f"ETA: {format_time(eta)}")

    # ============================================================

    save_results(results, OUTPUT_DIR)

    print("\nDONE")


if __name__ == "__main__":
    main()