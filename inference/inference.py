
"""
Бенчмаркинг YOLO-моделей в форматах pt, onnx, torchscript.
Запускает модели последовательно (GPU не любит параллельные контексты),
но собирает результаты быстро благодаря правильному замеру времени.
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
VIDEO_PATH = "../dataset/downloaded_videos/test_vids/vid1.mp4"
MODELS_ROOT = "../yolo/runs/detect/yolo/runs"
OUTPUT_DIR = "./benchmark_results"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
WARMUP_FRAMES = 5   # кадров для прогрева модели перед замером
# ==================================


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}с"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}мин"
    else:
        return f"{seconds / 3600:.1f}ч"


def find_all_models(models_root: str) -> list[dict]:
    """Находит все модели в форматах pt, onnx, torchscript."""
    models = []
    root_path = Path(models_root)

    for model_file in sorted(root_path.rglob("*")):
        if model_file.suffix not in {".pt", ".onnx", ".torchscript"}:
            continue
        
        stem = model_file.stem
        if stem != "best":
            continue

        # Берём имя родительской папки модели (yolov8n, yolo11s, ...)
        # Структура: runs/<model_name>/weights/best.pt
        parts = model_file.parts
        try:
            runs_idx = parts.index(root_path.name)
            model_name = parts[runs_idx + 1]
        except (ValueError, IndexError):
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


def run_model(model_info: dict, video_path: str) -> dict:
    """
    Запускает инференс одной модели и возвращает метрики.
    Работает в основном процессе — GPU эффективнее при одном контексте.
    """
    model_path = model_info["path"]
    model_format = model_info["format"]
    model_name = model_info["name"]

    print(f"\n{'─'*60}")
    print(f"▶ {model_name} [{model_format.upper()}]  {model_info['size_mb']:.1f} MB")

    # ONNX без CUDA-провайдера — пропускаем, не тратим время
    if model_format == "onnx" and not check_onnx_cuda():
        print("  ⚠ ONNX CUDA недоступен (нужна libcublasLt.so.11, проверь версию CUDA/onnxruntime)")
        print("  → Пропускаем модель")
        return {
            "model_name": model_name,
            "format": model_format,
            "skipped": True,
            "skip_reason": "ONNX CUDA unavailable",
            "timestamp": datetime.now().isoformat(),
        }

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

        # --- Прогрев ---
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"  Прогрев ({WARMUP_FRAMES} кадров)...", end=" ", flush=True)
        warmup_done = 0
        for _ in model(video_path, stream=True, conf=CONF_THRESHOLD,
                       iou=IOU_THRESHOLD, verbose=False):
            warmup_done += 1
            if warmup_done >= WARMUP_FRAMES:
                break
        print("готово")

        # --- Инференс ---
        print(f"  Инференс ({total_frames} кадров)...")
        frame_times_ms: list[float] = []
        frame_count = 0
        last_report = time.time()
        t_inference_start = time.time()  # используем time.time() везде для единообразия

        for result in model(video_path, stream=True, conf=CONF_THRESHOLD,
                            iou=IOU_THRESHOLD, verbose=False):
            # Правильный способ замерить время одного кадра — через speed из result
            # result.speed = {"preprocess": ms, "inference": ms, "postprocess": ms}
            spd = result.speed
            frame_ms = spd.get("preprocess", 0) + spd.get("inference", 0) + spd.get("postprocess", 0)
            frame_times_ms.append(frame_ms)
            frame_count += 1

            now = time.time()
            if now - last_report >= 5.0:
                elapsed = now - t_inference_start
                fps_so_far = frame_count / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_count) / fps_so_far if fps_so_far > 0 else 0
                pct = 100 * frame_count / total_frames
                print(f"    {pct:5.1f}%  {frame_count}/{total_frames}  "
                      f"{fps_so_far:.1f} FPS  осталось ~{format_time(eta)}")
                last_report = now

        total_time = time.time() - t_inference_start
        fps = frame_count / total_time if total_time > 0 else 0
        vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

        arr = np.array(frame_times_ms) if frame_times_ms else np.array([0.0])
        metrics = {
            "model_name": model_name,
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
            "timestamp": datetime.now().isoformat(),
        }

        print(f"  ✓ FPS: {fps:.1f}  |  avg: {metrics['frame_time_avg_ms']}ms  "
              f"|  p95: {metrics['frame_time_p95_ms']}ms  "
              f"|  VRAM peak: {vram_peak:.0f}MB")
        return metrics

    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return {
            "model_name": model_name,
            "format": model_format,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }

    finally:
        # Явно освобождаем модель перед следующей
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
        "model_name", "format", "model_size_mb", "load_time_ms",
        "fps", "total_time_seconds", "frame_time_avg_ms",
        "frame_time_p50_ms", "frame_time_p95_ms", "frame_time_p99_ms",
        "vram_peak_mb", "ram_delta_mb", "total_frames",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(successful)
    print(f"📊 CSV:  {csv_path}")

    # Таблица в консоль
    print(f"\n{'═'*80}")
    print("РЕЗУЛЬТАТЫ  (сортировка по FPS ↓)")
    print(f"{'═'*80}")
    header = f"{'Модель':<20} {'Формат':<12} {'FPS':>7} {'avg ms':>8} {'p95 ms':>8} {'VRAM MB':>8} {'Размер':>8}"
    print(header)
    print("─" * 80)

    for r in sorted(successful, key=lambda x: x["fps"], reverse=True):
        print(
            f"{r['model_name'][:20]:<20} {r['format']:<12} "
            f"{r['fps']:>7.1f} {r['frame_time_avg_ms']:>8.1f} "
            f"{r['frame_time_p95_ms']:>8.1f} {r['vram_peak_mb']:>8.0f} "
            f"{r['model_size_mb']:>7.1f}M"
        )

    best = max(successful, key=lambda x: x["fps"])
    print(f"\n🏆 Лучшая: {best['model_name']} [{best['format'].upper()}] — {best['fps']:.1f} FPS")

    skipped = [r for r in results if r.get("skipped")]
    errors = [r for r in results if "error" in r]
    if skipped:
        print(f"⏭ Пропущено: {len(skipped)}  ({', '.join(r['model_name']+'/'+r['format'] for r in skipped[:3])}{'...' if len(skipped)>3 else ''})")
    if errors:
        print(f"✗ Ошибки:   {len(errors)}  ({', '.join(r['model_name']+'/'+r['format'] for r in errors[:3])}{'...' if len(errors)>3 else ''})")


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
        print("\n  Чтобы починить ONNX CUDA, установи совместимую версию:")
        print("  pip install onnxruntime-gpu==1.18.0   # для CUDA 12")
        print("  или: pip install onnxruntime-gpu==1.16.3  # для CUDA 11")

    for path, label in [(VIDEO_PATH, "Видео"), (MODELS_ROOT, "Модели")]:
        if not os.path.exists(path):
            print(f"\n✗ {label} не найден: {path}")
            sys.exit(1)

    models = find_all_models(MODELS_ROOT)
    if not models:
        print(f"Модели не найдены в {MODELS_ROOT}")
        sys.exit(1)

    # Группируем для наглядности
    by_format: dict[str, int] = {}
    for m in models:
        by_format[m["format"]] = by_format.get(m["format"], 0) + 1

    print(f"\nНайдено моделей: {len(models)}")
    for fmt, cnt in sorted(by_format.items()):
        skipped_mark = " (будут пропущены — нет CUDA)" if fmt == "onnx" and not onnx_cuda else ""
        print(f"  {fmt:>12}: {cnt}{skipped_mark}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    duration = total_frames / fps_video if fps_video > 0 else 0
    print(f"\nВидео: {total_frames} кадров, {fps_video:.1f} FPS, ~{format_time(duration)}")
    print(f"\nЗапуск последовательного бенчмарка (GPU эффективнее без параллелизма)")
    print("Продолжить? (y/n): ", end="")
    if input().strip().lower() != "y":
        sys.exit(0)

    results = []
    t_start = time.time()

    for i, model_info in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}]", end="")
        result = run_model(model_info, VIDEO_PATH)
        results.append(result)

        elapsed = time.time() - t_start
        done = i
        remaining = len(models) - done
        if done > 0 and remaining > 0:
            eta = elapsed / done * remaining
            print(f"  Общий прогресс: {done}/{len(models)} | осталось ~{format_time(eta)}")

    save_results(results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
