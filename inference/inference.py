#!/usr/bin/env python3
"""
Бенчмаркинг YOLO-моделей в форматах pt, onnx, torchscript.
Запускает модели последовательно (GPU не любит параллельные контексты).
Сохраняет видео с bbox в benchmark_results_videos/<model_name>/<video_name>.
"""
from __future__ import annotations

import os
import sys
import time
import json
import gc
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

import torch
import numpy as np
import psutil
from ultralytics import YOLO
import cv2


# ========== КОНФИГУРАЦИЯ ==========
VIDEO_DIR = "../dataset/downloaded_videos/test_vids"   # папка с видео
MODELS_ROOT = "../yolo/runs/detect/yolo/runs"
OUTPUT_DIR = "./benchmark_results"
OUTPUT_VIDEO_DIR = "./benchmark_results_videos"        # папка для видео с bbox
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
WARMUP_FRAMES = 5   # кадров для прогрева модели перед замером
SAVE_VIDEO = True   # False — отключить сохранение видео (быстрее)
# ==================================


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}с"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}мин"
    else:
        return f"{seconds / 3600:.1f}ч"


def find_all_videos(video_dir: str) -> list[str]:
    """Находит все видеофайлы в папке рекурсивно."""
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    root = Path(video_dir)
    videos = sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in exts)
    return videos


def find_all_models(models_root: str) -> list[dict]:
    """Находит все модели в форматах pt, onnx, torchscript."""
    models = []
    root_path = Path(models_root)

    for model_file in sorted(root_path.rglob("*")):
        if model_file.suffix not in {".pt", ".onnx", ".torchscript"}:
            continue

        # Берём только файлы с именем best.*
        if model_file.stem != "best":
            continue

        # Структура: runs/<model_name>/weights/best.pt
        # parent = weights/, parent.parent = model_name/
        model_name = model_file.parent.parent.name

        models.append({
            "path": str(model_file),
            "format": model_file.suffix[1:],  # без точки
            "name": model_name,
            "size_mb": model_file.stat().st_size / 1024 / 1024,
        })

    return models


def check_onnx_cuda() -> bool:
    """Проверяет, доступен ли CUDA для ONNX Runtime."""
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def make_video_writer(
    output_path: str,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    """Создаёт VideoWriter с кодеком mp4v."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось открыть VideoWriter для {output_path}")
    return writer


def run_model(model_info: dict, video_path: str) -> dict:
    """
    Запускает инференс одной модели на одном видео и возвращает метрики.
    Если SAVE_VIDEO=True — сохраняет видео с нарисованными bbox.
    """
    model_path = model_info["path"]
    model_format = model_info["format"]
    model_name = model_info["name"]
    video_name = Path(video_path).name
    video_stem = Path(video_path).stem

    print(f"\n{'─'*60}")
    print(f"▶ {model_name} [{model_format.upper()}]  {model_info['size_mb']:.1f} MB")
    print(f"  Видео: {video_name}")

    # ONNX без CUDA-провайдера — пропускаем
    if model_format == "onnx" and not check_onnx_cuda():
        print("  ⚠ ONNX CUDA недоступен — пропускаем модель")
        return {
            "model_name": model_name,
            "video_name": video_name,
            "format": model_format,
            "skipped": True,
            "skip_reason": "ONNX CUDA unavailable",
            "timestamp": datetime.now().isoformat(),
        }

    # Путь для сохранения видео: benchmark_results_videos/<model_name>/<video_stem>.<format>.mp4
    output_video_path: str | None = None
    if SAVE_VIDEO:
        video_out_dir = Path(OUTPUT_VIDEO_DIR) / model_name
        video_out_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = str(video_out_dir / f"{video_stem}__{model_format}.mp4")

    try:
        # --- Загрузка ---
        torch.cuda.empty_cache()
        gc.collect()

        ram_before = psutil.Process().memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

        t_load = time.perf_counter()
        model = YOLO(model_path)
        load_ms = (time.perf_counter() - t_load) * 1000

        ram_after = psutil.Process().memory_info().rss / 1024 / 1024
        vram_after = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        print(f"  Загрузка: {load_ms:.0f}ms  |  VRAM δ: {vram_after - vram_before:+.1f}MB")

        # --- Получаем параметры видео ---
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # --- Прогрев ---
        print(f"  Прогрев ({WARMUP_FRAMES} кадров)...", end=" ", flush=True)
        warmup_done = 0
        for _ in model(video_path, stream=True, conf=CONF_THRESHOLD,
                       iou=IOU_THRESHOLD, verbose=False):
            warmup_done += 1
            if warmup_done >= WARMUP_FRAMES:
                break
        print("готово")

        # --- Инференс + запись видео ---
        save_info = f" + запись видео → {Path(output_video_path).name}" if output_video_path else ""
        print(f"  Инференс ({total_frames} кадров){save_info}...")

        frame_times_ms: list[float] = []
        frame_count = 0
        last_report = time.time()
        t_inference_start = time.time()

        writer: cv2.VideoWriter | None = None

        for result in model(video_path, stream=True, conf=CONF_THRESHOLD,
                            iou=IOU_THRESHOLD, verbose=False):

            # Время на этот кадр
            spd = result.speed
            frame_ms = spd.get("preprocess", 0) + spd.get("inference", 0) + spd.get("postprocess", 0)
            frame_times_ms.append(frame_ms)
            frame_count += 1

            # Сохраняем кадр с bbox
            if output_video_path is not None:
                # result.plot() возвращает BGR numpy array с нарисованными bbox
                annotated_frame = result.plot()
                if writer is None:
                    h, w = annotated_frame.shape[:2]
                    writer = make_video_writer(output_video_path, w, h, video_fps)
                writer.write(annotated_frame)

            now = time.time()
            if now - last_report >= 5.0:
                elapsed = now - t_inference_start
                fps_so_far = frame_count / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_count) / fps_so_far if fps_so_far > 0 else 0
                pct = 100 * frame_count / total_frames
                print(f"    {pct:5.1f}%  {frame_count}/{total_frames}  "
                      f"{fps_so_far:.1f} FPS  осталось ~{format_time(eta)}")
                last_report = now

        if writer is not None:
            writer.release()
            print(f"  💾 Видео сохранено: {output_video_path}")

        total_time = time.time() - t_inference_start
        fps = frame_count / total_time if total_time > 0 else 0
        vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

        arr = np.array(frame_times_ms) if frame_times_ms else np.array([0.0])
        metrics = {
            "model_name": model_name,
            "video_name": video_name,
            "format": model_format,
            "model_path": model_path,
            "model_size_mb": round(model_info["size_mb"], 2),
            "load_time_ms": round(load_ms, 1),
            "total_frames": frame_count,
            "total_time_seconds": round(total_time, 2),
            "fps": round(fps, 2),
            "frame_time_avg_ms": round(float(np.mean(arr)), 2),
            "frame_time_p50_ms": round(float(np.percentile(arr, 50)), 2),
            "frame_time_p95_ms": round(float(np.percentile(arr, 95)), 2),
            "frame_time_p99_ms": round(float(np.percentile(arr, 99)), 2),
            "vram_peak_mb": round(vram_peak, 1),
            "ram_delta_mb": round(ram_after - ram_before, 1),
            "output_video": output_video_path,
            "timestamp": datetime.now().isoformat(),
        }

        print(f"  ✓ FPS: {fps:.1f}  |  avg: {metrics['frame_time_avg_ms']}ms  "
              f"|  p95: {metrics['frame_time_p95_ms']}ms  "
              f"|  VRAM peak: {vram_peak:.0f}MB")
        return metrics

    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        # Закрываем writer если был открыт
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
        return {
            "model_name": model_name,
            "video_name": video_name,
            "format": model_format,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }

    finally:
        try:
            del model
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()


def save_results(results: list[dict], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON — всё включая ошибки и пропуски
    json_path = out / f"benchmark_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 JSON: {json_path}")

    # CSV — только успешные
    successful = [r for r in results if "fps" in r]
    if not successful:
        print("  Нет успешных результатов для CSV")
        return

    csv_path = out / f"benchmark_{ts}.csv"
    fields = [
        "model_name", "video_name", "format", "model_size_mb", "load_time_ms",
        "fps", "total_time_seconds", "frame_time_avg_ms",
        "frame_time_p50_ms", "frame_time_p95_ms", "frame_time_p99_ms",
        "vram_peak_mb", "ram_delta_mb", "total_frames", "output_video",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(successful)
    print(f"📊 CSV:  {csv_path}")

    # Таблица в консоль — группируем по видео
    print(f"\n{'═'*90}")
    print("РЕЗУЛЬТАТЫ  (сортировка по FPS ↓)")
    print(f"{'═'*90}")
    header = (f"{'Модель':<20} {'Видео':<15} {'Формат':<12} "
              f"{'FPS':>7} {'avg ms':>8} {'p95 ms':>8} {'VRAM MB':>8} {'Размер':>8}")
    print(header)
    print("─" * 90)

    for r in sorted(successful, key=lambda x: x["fps"], reverse=True):
        vname = r.get("video_name", "")[:14]
        print(
            f"{r['model_name'][:20]:<20} {vname:<15} {r['format']:<12} "
            f"{r['fps']:>7.1f} {r['frame_time_avg_ms']:>8.1f} "
            f"{r['frame_time_p95_ms']:>8.1f} {r['vram_peak_mb']:>8.0f} "
            f"{r['model_size_mb']:>7.1f}M"
        )

    best = max(successful, key=lambda x: x["fps"])
    print(f"\n🏆 Лучшая: {best['model_name']} [{best['format'].upper()}] "
          f"на {best.get('video_name','')} — {best['fps']:.1f} FPS")

    skipped = [r for r in results if r.get("skipped")]
    errors = [r for r in results if "error" in r]
    if skipped:
        names = ', '.join(r['model_name'] + '/' + r['format'] for r in skipped[:3])
        print(f"⏭ Пропущено: {len(skipped)}  ({names}{'...' if len(skipped) > 3 else ''})")
    if errors:
        names = ', '.join(r['model_name'] + '/' + r['format'] for r in errors[:3])
        print(f"✗ Ошибки:   {len(errors)}  ({names}{'...' if len(errors) > 3 else ''})")


def main():
    print("=" * 60)
    print("YOLO Benchmark")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU:  {gpu.name}")
        print(f"VRAM: {gpu.total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA недоступна — используем CPU")

    onnx_cuda = check_onnx_cuda()
    print(f"ONNX CUDA: {'✓' if onnx_cuda else '✗ (ONNX-модели будут пропущены)'}")
    if not onnx_cuda:
        print("  Для CUDA 12: pip install onnxruntime-gpu==1.18.0")
        print("  Для CUDA 11: pip install onnxruntime-gpu==1.16.3")

    print(f"Сохранение видео с bbox: {'✓' if SAVE_VIDEO else '✗'}")

    for path, label in [(VIDEO_DIR, "Папка с видео"), (MODELS_ROOT, "Папка с моделями")]:
        if not os.path.exists(path):
            print(f"\n✗ {label} не найден: {path}")
            sys.exit(1)

    videos = find_all_videos(VIDEO_DIR)
    if not videos:
        print(f"Видео не найдены в {VIDEO_DIR}")
        sys.exit(1)

    models = find_all_models(MODELS_ROOT)
    if not models:
        print(f"Модели не найдены в {MODELS_ROOT}")
        sys.exit(1)

    # Группируем модели по формату для наглядности
    by_format: dict[str, int] = {}
    for m in models:
        by_format[m["format"]] = by_format.get(m["format"], 0) + 1

    print(f"\nНайдено видео: {len(videos)}")
    for v in videos:
        print(f"  {Path(v).name}")

    print(f"\nНайдено моделей: {len(models)}")
    for fmt, cnt in sorted(by_format.items()):
        skipped_mark = " (будут пропущены — нет CUDA)" if fmt == "onnx" and not onnx_cuda else ""
        print(f"  {fmt:>12}: {cnt}{skipped_mark}")

    total_runs = len(models) * len(videos)
    print(f"\nВсего запусков: {len(models)} моделей × {len(videos)} видео = {total_runs}")
    if SAVE_VIDEO:
        print(f"Видео с bbox → {OUTPUT_VIDEO_DIR}/<model_name>/<video>__<format>.mp4")

    print("\nПродолжить? (y/n): ", end="")
    if input().strip().lower() != "y":
        sys.exit(0)

    results = []
    t_start = time.time()
    run_idx = 0

    for i, model_info in enumerate(models, 1):
        for j, video_path in enumerate(videos, 1):
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] модель {i}/{len(models)}, видео {j}/{len(videos)}", end="")
            result = run_model(model_info, video_path)
            results.append(result)

            elapsed = time.time() - t_start
            remaining = total_runs - run_idx
            if run_idx > 0 and remaining > 0:
                eta = elapsed / run_idx * remaining
                print(f"  Прогресс: {run_idx}/{total_runs} | осталось ~{format_time(eta)}")

    save_results(results, OUTPUT_DIR)


if __name__ == "__main__":
    main()