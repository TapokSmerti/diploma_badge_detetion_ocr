#!/usr/bin/env python3
"""
Шаг 2 — Бенчмарк трекинга людей.

Тестирует комбинации детекторов и трекеров:
- Детекторы: топ-3 из шага 1
- Трекеры: ByteTrack, BoT-SORT (ultralytics built-in)
           OC-SORT, StrongSORT, DeepSORT (через boxmot)

Метрики:
- FPS с трекером
- Замедление относительно чистой детекции (overhead %)
- Среднее кол-во активных треков на кадр
- Средняя длина трека (стабильность)
- ID Switches
- VRAM, RAM

Требования:
    pip install boxmot
"""
from __future__ import annotations

import gc
import csv
import json
import time
import sys
from collections import defaultdict
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
OUTPUT_DIR       = "./tracking_results"
OUTPUT_VIDEO_DIR = "./tracking_videos"

# Топ-3 модели из шага 1 (по среднему F1, исключая yolo12m)
DETECTOR_MODELS = [
    "yolov10s.pt",   # F1=0.953 avg, 203 FPS
    "yolov5su.pt",   # F1=0.948 avg, 217 FPS
    "yolov8s.pt",    # F1=0.940 avg, 228 FPS
]

# Трекеры — built-in ultralytics
BUILTIN_TRACKERS = [
    "bytetrack.yaml",
    "botsort.yaml",
]

# Трекеры через boxmot
BOXMOT_TRACKERS = [
    "ocsort",
    "strongsort",
    "deepocsort",   # DeepSORT-like через boxmot
]

CONF_THRESHOLD = 0.5
IOU_THRESHOLD  = 0.45
PERSON_CLASS   = 0
WARMUP_FRAMES  = 5
SAVE_VIDEO     = True

# FPS детекции без трекера из шага 1 — для расчёта overhead
# Заполни по своим результатам
BASELINE_FPS = {
    "yolov10s": 203.0,
    "yolov5su": 217.0,
    "yolov8s":  228.0,
}

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


def compute_track_stats(track_history: dict[int, list[int]]) -> dict:
    """
    track_history: {track_id: [frame_idx, frame_idx, ...]}
    Считает статистику по трекам.
    """
    if not track_history:
        return {
            "total_tracks":       0,
            "track_len_avg":      0.0,
            "track_len_median":   0.0,
            "track_len_min":      0,
            "track_len_max":      0,
            "short_tracks_pct":   0.0,  # треки длиной < 10 кадров (шум)
        }

    lengths = [len(frames) for frames in track_history.values()]
    arr = np.array(lengths)

    return {
        "total_tracks":     len(lengths),
        "track_len_avg":    round(float(np.mean(arr)),   1),
        "track_len_median": round(float(np.median(arr)), 1),
        "track_len_min":    int(np.min(arr)),
        "track_len_max":    int(np.max(arr)),
        # Короткие треки = шум / потери трекера
        "short_tracks_pct": round(100 * float(np.mean(arr < 10)), 1),
    }


def count_id_switches(track_history: dict[int, list[int]]) -> int:
    """
    Приблизительный подсчёт ID switches:
    ID switch = трек прерывается и появляется новый в той же области.
    Без GT считаем косвенно: кол-во треков которые живут < 5 кадров
    в середине видео (не в начале/конце) — признак переназначения ID.
    Точный IDSW требует GT.
    """
    if not track_history:
        return 0

    all_frames = [f for frames in track_history.values() for f in frames]
    if not all_frames:
        return 0

    max_frame = max(all_frames)
    margin    = max_frame * 0.05  # первые и последние 5% кадров не считаем

    switches = 0
    for frames in track_history.values():
        if not frames:
            continue
        start = min(frames)
        end   = max(frames)
        length = len(frames)
        # Короткий трек в середине видео = вероятный ID switch
        if length < 5 and start > margin and end < max_frame - margin:
            switches += 1

    return switches


# ── Встроенные трекеры ultralytics ────────────────────────────────────────────

def run_builtin_tracker(
    model_name: str,
    model_label: str,
    tracker_name: str,
    video_path: str,
) -> dict:
    """ByteTrack / BoT-SORT через model.track()"""

    tracker_label = tracker_name.replace(".yaml", "")
    video_name    = Path(video_path).name
    video_stem    = Path(video_path).stem

    print(f"\n{'='*65}")
    print(f"{model_label} + {tracker_label}  ->  {video_name}")

    output_video_path: str | None = None
    if SAVE_VIDEO:
        out_dir = Path(OUTPUT_VIDEO_DIR) / f"{model_label}__{tracker_label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = str(out_dir / f"{video_stem}.mp4")

    writer: cv2.VideoWriter | None = None

    try:
        torch.cuda.empty_cache()
        gc.collect()

        ram_before = psutil.Process().memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model   = YOLO(model_name)
        load_ms = 0.0  # не меряем — модель одна для всех трекеров

        cap          = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        # Прогрев
        print(f"Warmup ({WARMUP_FRAMES} frames)...", end=" ", flush=True)
        done = 0
        for _ in model.track(
            video_path, stream=True, tracker=tracker_name,
            conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            classes=[PERSON_CLASS], verbose=False, persist=True,
        ):
            done += 1
            if done >= WARMUP_FRAMES:
                break
        print("done")

        # Инференс
        print(f"Tracking ({total_frames} frames)...")

        frame_times_ms:  list[float]          = []
        active_per_frame: list[int]            = []
        track_history:   dict[int, list[int]] = defaultdict(list)

        frame_count = 0
        last_report = time.time()
        t0          = time.time()

        for result in model.track(
            video_path, stream=True, tracker=tracker_name,
            conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            classes=[PERSON_CLASS], verbose=False, persist=True,
        ):
            spd = result.speed
            frame_times_ms.append(
                spd.get("preprocess",  0)
                + spd.get("inference",  0)
                + spd.get("postprocess", 0)
            )

            # Активные треки на этом кадре
            active = 0
            if result.boxes is not None and result.boxes.id is not None:
                ids    = result.boxes.id.cpu().numpy().astype(int)
                active = len(ids)
                for tid in ids:
                    track_history[int(tid)].append(frame_count)

            active_per_frame.append(active)
            frame_count += 1

            # Видео
            if output_video_path is not None:
                annotated = result.plot()
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = make_video_writer(output_video_path, w, h, video_fps)
                writer.write(annotated)

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

        arr  = np.array(frame_times_ms) if frame_times_ms else np.array([0.0])
        apf  = np.array(active_per_frame) if active_per_frame else np.array([0])
        base = BASELINE_FPS.get(model_label, fps)

        track_stats  = compute_track_stats(dict(track_history))
        id_switches  = count_id_switches(dict(track_history))

        metrics = {
            "model":              model_label,
            "tracker":            tracker_label,
            "video_name":         video_name,
            "fps":                round(fps, 2),
            "fps_overhead_pct":   round(100 * (base - fps) / base, 1) if base > 0 else 0,
            "frame_time_avg_ms":  round(float(np.mean(arr)),           2),
            "frame_time_p50_ms":  round(float(np.percentile(arr, 50)), 2),
            "frame_time_p95_ms":  round(float(np.percentile(arr, 95)), 2),
            "frame_time_p99_ms":  round(float(np.percentile(arr, 99)), 2),
            "vram_peak_mb":       round(vram_peak, 1),
            "ram_delta_mb":       round(ram_delta, 1),
            "total_frames":       frame_count,
            "active_tracks_avg":  round(float(np.mean(apf)), 2),
            "active_tracks_max":  int(np.max(apf)),
            "id_switches_approx": id_switches,
            **track_stats,
            "output_video":       output_video_path,
            "timestamp":          datetime.now().isoformat(),
        }

        print(
            f"✓ FPS: {fps:.1f} (overhead: {metrics['fps_overhead_pct']}%) | "
            f"avg: {metrics['frame_time_avg_ms']}ms | "
            f"VRAM: {vram_peak:.0f}MB | "
            f"tracks avg: {metrics['active_tracks_avg']:.1f} | "
            f"total tracks: {metrics['total_tracks']} | "
            f"ID sw~: {id_switches}"
        )
        return metrics

    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "model":      model_label,
            "tracker":    tracker_label,
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


# ── Трекеры через boxmot ──────────────────────────────────────────────────────

def run_boxmot_tracker(
    model_name: str,
    model_label: str,
    tracker_name: str,
    video_path: str,
) -> dict:
    """OC-SORT / StrongSORT / DeepSORT через boxmot."""

    video_name = Path(video_path).name
    video_stem = Path(video_path).stem

    print(f"\n{'='*65}")
    print(f"{model_label} + {tracker_name}  ->  {video_name}")

    try:
        from boxmot import create_tracker
    except ImportError:
        print("ERROR: boxmot not installed. Run: pip install boxmot")
        return {
            "model":      model_label,
            "tracker":    tracker_name,
            "video_name": video_name,
            "error":      "boxmot not installed",
            "timestamp":  datetime.now().isoformat(),
        }

    output_video_path: str | None = None
    if SAVE_VIDEO:
        out_dir = Path(OUTPUT_VIDEO_DIR) / f"{model_label}__{tracker_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = str(out_dir / f"{video_stem}.mp4")

    writer: cv2.VideoWriter | None = None

    try:
        torch.cuda.empty_cache()
        gc.collect()

        ram_before = psutil.Process().memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = YOLO(model_name)

        # Создаём трекер через boxmot
        tracker = create_tracker(
            tracker_type=tracker_name,
            tracker_config=None,   # дефолтный конфиг
            reid_weights=None,     # без ReID для скорости
            device=device,
            half=False,
        )

        cap          = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Прогрев детектора
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
        tracker.reset()
        print("done")

        print(f"Tracking ({total_frames} frames)...")

        frame_times_ms:   list[float]          = []
        active_per_frame: list[int]            = []
        track_history:    dict[int, list[int]] = defaultdict(list)

        frame_count = 0
        last_report = time.time()
        t0          = time.time()

        # boxmot работает покадрово через cv2
        cap = cv2.VideoCapture(video_path)

        for result in model(
            video_path, stream=True,
            conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
            classes=[PERSON_CLASS], verbose=False,
        ):
            t_frame = time.perf_counter()

            # Читаем кадр для boxmot (нужен BGR numpy)
            ret, frame = cap.read()
            if not ret:
                break

            # Детекции в формате [x1,y1,x2,y2,conf,cls]
            dets = np.empty((0, 6))
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy  = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy().reshape(-1, 1)
                clss  = result.boxes.cls.cpu().numpy().reshape(-1, 1)
                dets  = np.hstack([xyxy, confs, clss])

            # Обновляем трекер
            tracks = tracker.update(dets, frame)  # → [x1,y1,x2,y2,id,conf,cls,...]

            frame_ms = (time.perf_counter() - t_frame) * 1000
            frame_times_ms.append(frame_ms)

            active = 0
            if tracks is not None and len(tracks) > 0:
                active = len(tracks)
                for track in tracks:
                    tid = int(track[4])
                    track_history[tid].append(frame_count)

            active_per_frame.append(active)
            frame_count += 1

            # Видео с bbox
            if output_video_path is not None:
                if writer is None:
                    writer = make_video_writer(output_video_path, width, height, video_fps)
                # Рисуем треки на кадре
                if tracks is not None and len(tracks) > 0:
                    for track in tracks:
                        x1, y1, x2, y2, tid = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{tid}", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                writer.write(frame)

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

        cap.release()

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

        arr  = np.array(frame_times_ms) if frame_times_ms else np.array([0.0])
        apf  = np.array(active_per_frame) if active_per_frame else np.array([0])
        base = BASELINE_FPS.get(model_label, fps)

        track_stats = compute_track_stats(dict(track_history))
        id_switches = count_id_switches(dict(track_history))

        metrics = {
            "model":              model_label,
            "tracker":            tracker_name,
            "video_name":         video_name,
            "fps":                round(fps, 2),
            "fps_overhead_pct":   round(100 * (base - fps) / base, 1) if base > 0 else 0,
            "frame_time_avg_ms":  round(float(np.mean(arr)),           2),
            "frame_time_p50_ms":  round(float(np.percentile(arr, 50)), 2),
            "frame_time_p95_ms":  round(float(np.percentile(arr, 95)), 2),
            "frame_time_p99_ms":  round(float(np.percentile(arr, 99)), 2),
            "vram_peak_mb":       round(vram_peak, 1),
            "ram_delta_mb":       round(ram_delta, 1),
            "total_frames":       frame_count,
            "active_tracks_avg":  round(float(np.mean(apf)), 2),
            "active_tracks_max":  int(np.max(apf)),
            "id_switches_approx": id_switches,
            **track_stats,
            "output_video":       output_video_path,
            "timestamp":          datetime.now().isoformat(),
        }

        print(
            f"✓ FPS: {fps:.1f} (overhead: {metrics['fps_overhead_pct']}%) | "
            f"avg: {metrics['frame_time_avg_ms']}ms | "
            f"VRAM: {vram_peak:.0f}MB | "
            f"tracks avg: {metrics['active_tracks_avg']:.1f} | "
            f"total tracks: {metrics['total_tracks']} | "
            f"ID sw~: {id_switches}"
        )
        return metrics

    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "model":      model_label,
            "tracker":    tracker_name,
            "video_name": video_name,
            "error":      str(e),
            "timestamp":  datetime.now().isoformat(),
        }

    finally:
        if writer is not None:
            writer.release()
        try:
            del model
            del tracker
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()


# ── Сохранение результатов ────────────────────────────────────────────────────

def save_results(results: list[dict]) -> None:
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out / f"tracking_benchmark_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {json_path}")

    successful = [r for r in results if "fps" in r]
    if not successful:
        print("No successful results for CSV")
        return

    csv_path = out / f"tracking_benchmark_{ts}.csv"
    fields = [
        "model", "tracker", "video_name",
        "fps", "fps_overhead_pct",
        "frame_time_avg_ms", "frame_time_p50_ms",
        "frame_time_p95_ms", "frame_time_p99_ms",
        "vram_peak_mb", "ram_delta_mb", "total_frames",
        "active_tracks_avg", "active_tracks_max",
        "id_switches_approx",
        "total_tracks", "track_len_avg", "track_len_median",
        "track_len_min", "track_len_max", "short_tracks_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(successful)
    print(f"CSV:  {csv_path}")

    # ── Таблица производительности ────────────────────────────
    W = 110
    print(f"\n{'='*W}")
    print("TRACKING RESULTS — sorted by FPS")
    print(f"{'='*W}")
    print(
        f"{'Model':<12} {'Tracker':<12} {'Video':<12} {'FPS':>7} "
        f"{'overhead%':>10} {'avg ms':>8} {'p95 ms':>8} "
        f"{'VRAM MB':>8} {'trk avg':>8} {'ID sw~':>7}"
    )
    print("-" * W)
    for r in sorted(successful, key=lambda x: x["fps"], reverse=True):
        print(
            f"{r['model'][:12]:<12} {r['tracker'][:12]:<12} "
            f"{r['video_name'][:12]:<12} {r['fps']:>7.1f} "
            f"{r.get('fps_overhead_pct', 0):>9.1f}% "
            f"{r['frame_time_avg_ms']:>8.1f} {r['frame_time_p95_ms']:>8.1f} "
            f"{r['vram_peak_mb']:>8.0f} "
            f"{r.get('active_tracks_avg', 0):>8.1f} "
            f"{r.get('id_switches_approx', 0):>7}"
        )

    # ── Таблица стабильности треков ───────────────────────────
    print(f"\n{'='*W}")
    print("TRACK STABILITY — sorted by track_len_avg (longer = more stable)")
    print(f"{'='*W}")
    print(
        f"{'Model':<12} {'Tracker':<12} {'Video':<12} "
        f"{'total trk':>10} {'len avg':>8} {'len med':>8} "
        f"{'len max':>8} {'short%':>8} {'ID sw~':>7}"
    )
    print("-" * W)
    for r in sorted(successful, key=lambda x: x.get("track_len_avg", 0), reverse=True):
        print(
            f"{r['model'][:12]:<12} {r['tracker'][:12]:<12} "
            f"{r['video_name'][:12]:<12} "
            f"{r.get('total_tracks', 0):>10} "
            f"{r.get('track_len_avg', 0):>8.1f} "
            f"{r.get('track_len_median', 0):>8.1f} "
            f"{r.get('track_len_max', 0):>8} "
            f"{r.get('short_tracks_pct', 0):>7.1f}% "
            f"{r.get('id_switches_approx', 0):>7}"
        )

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\nErrors: {len(errors)}")
        for r in errors:
            print(f"  {r['model']} + {r['tracker']} / {r['video_name']}: {r['error']}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("Step 2 — Tracking Benchmark")
    print("=" * 65)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU:  {gpu.name}  {gpu.total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available — using CPU")

    print(f"Conf: {CONF_THRESHOLD}  IoU: {IOU_THRESHOLD}")
    print(f"Save video: {'YES' if SAVE_VIDEO else 'NO'}")

    # Проверяем boxmot
    try:
        import boxmot
        print(f"boxmot: ✓ {boxmot.__version__}")
        has_boxmot = True
    except ImportError:
        print("boxmot: ✗ NOT INSTALLED — OC-SORT/StrongSORT/DeepSORT will be skipped")
        print("  Install: pip install boxmot")
        has_boxmot = False

    if not Path(VIDEO_DIR).exists():
        print(f"VIDEO DIR NOT FOUND: {VIDEO_DIR}")
        sys.exit(1)

    videos = find_all_videos(VIDEO_DIR)
    if not videos:
        print(f"No videos found in {VIDEO_DIR}")
        sys.exit(1)

    # Все комбинации
    combos = []
    for model_name in DETECTOR_MODELS:
        model_label = Path(model_name).stem
        for tracker in BUILTIN_TRACKERS:
            combos.append(("builtin", model_name, model_label, tracker))
        if has_boxmot:
            for tracker in BOXMOT_TRACKERS:
                combos.append(("boxmot", model_name, model_label, tracker))

    total_runs = len(combos) * len(videos)

    print(f"\nVideos: {len(videos)}")
    for v in videos:
        print(f"  {Path(v).name}")

    print(f"\nCombinations: {len(combos)}")
    for kind, _, label, tracker in combos:
        print(f"  {label} + {tracker}  [{kind}]")

    print(f"\nTotal runs: {len(combos)} combos × {len(videos)} videos = {total_runs}")
    print("\nContinue? (y/n): ", end="")
    if input().strip().lower() != "y":
        sys.exit(0)

    results:  list[dict] = []
    t_global = time.time()
    run_idx  = 0

    for kind, model_name, model_label, tracker in combos:
        for video_path in videos:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}]")

            if kind == "builtin":
                result = run_builtin_tracker(model_name, model_label, tracker, video_path)
            else:
                result = run_boxmot_tracker(model_name, model_label, tracker, video_path)

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