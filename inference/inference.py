#!/usr/bin/env python3
"""
Бенчмаркинг 17 моделей в 3 форматах (pt, onnx, torchscript)
С выводом времени ожидания и прогресса в консоль
"""

import os
import sys
import time
import json
import gc
import csv
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import multiprocessing as mp

import torch
import numpy as np
import psutil
from ultralytics import YOLO
import cv2

# ========== КОНФИГУРАЦИЯ ==========
VIDEO_PATH = "../dataset/downloaded_videos/vid1.mp4"  # Укажи путь к тестовому видео
MODELS_ROOT = "../yolo/runs/detect/yolo/runs"  # Корневая папка со всеми моделями (17 папок)
OUTPUT_DIR = "./benchmark_results"  # Куда сохранять результаты
SAVE_OUTPUT_VIDEOS = True  # Сохранять видео с детекциями для каждой модели
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
PARALLEL_MODELS = 8  # Количество моделей, запускаемых параллельно (настрой под свой GPU)
# ==================================

# Глобальные счётчики для прогресса
completed_count = 0
total_count = 0
progress_lock = threading.Lock()


def format_time(seconds):
    """Форматирует время в читаемый вид"""
    if seconds < 60:
        return f"{seconds:.1f} сек"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} мин"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} ч"


def estimate_inference_time(model_path: str, video_path: str, num_test_frames: int = 30) -> float:
    """
    Быстро оценивает время инференса на небольшом количестве кадров
    Возвращает预估 полное время в секундах
    """
    try:
        # Загружаем модель
        model = YOLO(model_path)
        
        # Открываем видео и получаем общее количество кадров
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Тестируем на первых N кадрах
        test_times = []
        frame_count = 0
        
        for result in model(video_path, stream=True, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False):
            if frame_count >= num_test_frames:
                break
            
            start_time = time.perf_counter()
            # Просто ждём завершения кадра (он уже обработан в result)
            frame_time = (time.perf_counter() - start_time) * 1000
            test_times.append(frame_time)
            frame_count += 1
        
        if test_times:
            avg_time_per_frame_ms = np.mean(test_times)
            estimated_total_seconds = (avg_time_per_frame_ms * total_frames) / 1000
            return estimated_total_seconds
        else:
            return 0
    
    except Exception as e:
        print(f"  ⚠️ Ошибка оценки времени: {e}")
        return 0


def inference_worker(model_info: dict, video_path: str, output_dir: str, worker_id: int):
    """
    Запускает инференс для одной модели (работает в отдельном процессе)
    """
    global completed_count, total_count
    
    model_path = model_info['path']
    model_format = model_info['format']
    model_name = model_info['name']
    
    start_total = time.time()
    
    try:
        print(f"\n[Воркер {worker_id}] 🔄 Загрузка: {model_name} ({model_format})")
        
        # Замеряем память до загрузки
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()
        
        ram_before = psutil.Process().memory_info().rss / 1024 / 1024
        vram_before = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        
        # Загружаем модель
        load_start = time.perf_counter()
        model = YOLO(model_path)
        load_time_ms = (time.perf_counter() - load_start) * 1000
        
        ram_after = psutil.Process().memory_info().rss / 1024 / 1024
        vram_after = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        
        print(f"[Воркер {worker_id}] ✅ Модель загружена за {load_time_ms:.0f}ms "
              f"(VRAM: {vram_after - vram_before:.1f}MB)")
        
        # Запускаем инференс
        print(f"[Воркер {worker_id}] 🎬 Запуск инференса на видео...")
        
        # Переменные для замеров
        frame_times = []
        frame_count = 0
        
        # Получаем общее количество кадров для прогресса
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        inference_start = time.perf_counter()
        
        # Инференс с прогрессом
        last_print_time = time.time()
        
        for result in model(video_path, stream=True, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False):
            frame_start = time.perf_counter()
            
            frame_time = (time.perf_counter() - frame_start) * 1000
            frame_times.append(frame_time)
            frame_count += 1
            
            # Печатаем прогресс каждые 5 секунд
            current_time = time.time()
            if current_time - last_print_time >= 5:
                progress = (frame_count / total_frames) * 100
                elapsed = current_time - inference_start
                eta = (elapsed / frame_count) * (total_frames - frame_count) if frame_count > 0 else 0
                
                print(f"[Воркер {worker_id}] 📊 {model_name}: {progress:.1f}% "
                      f"({frame_count}/{total_frames} кадров) | "
                      f"Прошло: {format_time(elapsed)} | "
                      f"Осталось: {format_time(eta)}")
                last_print_time = current_time
        
        total_time = time.perf_counter() - inference_start
        fps = frame_count / total_time if total_time > 0 else 0
        
        # Замеряем память после инференса
        vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        
        # Собираем метрики
        metrics = {
            "model_name": model_name,
            "format": model_format,
            "model_path": model_path,
            "model_size_mb": model_info['size_mb'],
            "load_time_ms": load_time_ms,
            "total_frames": frame_count,
            "total_time_seconds": total_time,
            "fps": fps,
            "inference_time_avg_ms": np.mean(frame_times) if frame_times else 0,
            "inference_time_p50_ms": np.percentile(frame_times, 50) if frame_times else 0,
            "inference_time_p95_ms": np.percentile(frame_times, 95) if frame_times else 0,
            "inference_time_p99_ms": np.percentile(frame_times, 99) if frame_times else 0,
            "vram_peak_mb": vram_peak,
            "ram_used_mb": ram_after - ram_before,
            "timestamp": datetime.now().isoformat()
        }
        
        total_elapsed = time.time() - start_total
        print(f"[Воркер {worker_id}] ✅ {model_name} ({model_format}) - "
              f"FPS: {fps:.1f} | Время: {format_time(total_time)} | "
              f"Всего: {format_time(total_elapsed)}")
        
        return metrics
    
    except Exception as e:
        print(f"[Воркер {worker_id}] ❌ Ошибка {model_name}: {e}")
        return {
            "model_name": model_name,
            "format": model_format,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
    
    finally:
        with progress_lock:
            completed_count += 1
            remaining = total_count - completed_count
            if remaining > 0:
                print(f"\n📈 Общий прогресс: {completed_count}/{total_count} моделей завершено. "
                      f"Осталось: {remaining}")


def estimate_all_models(models: list, video_path: str) -> dict:
    """Оценивает время выполнения для всех моделей"""
    print("\n" + "=" * 60)
    print("🔮 Оценка времени выполнения...")
    print("=" * 60)
    
    estimations = {}
    total_estimated = 0
    
    for i, model in enumerate(models):
        print(f"  Оценка {i+1}/{len(models)}: {model['name']} ({model['format']})...", end=" ", flush=True)
        est_time = estimate_inference_time(model['path'], video_path, num_test_frames=20)
        estimations[f"{model['name']}_{model['format']}"] = est_time
        total_estimated += est_time
        print(f"~{format_time(est_time)}")
    
    # С учётом параллельности
    parallel_time = total_estimated / PARALLEL_MODELS if PARALLEL_MODELS > 0 else total_estimated
    
    print("\n" + "=" * 60)
    print("📊 Оценка времени (приблизительная):")
    print(f"  Последовательно: ~{format_time(total_estimated)}")
    print(f"  Параллельно ({PARALLEL_MODELS} процессов): ~{format_time(parallel_time)}")
    print("=" * 60)
    
    return estimations


def find_all_models(models_root: str):
    """Находит все модели в форматах pt, onnx, torchscript"""
    models = []
    root_path = Path(models_root)
    
    for model_file in root_path.rglob("*"):
        if model_file.suffix in [".pt", ".onnx", ".torchscript"]:
            # Пытаемся извлечь имя модели из пути
            model_name = model_file.parent.name if model_file.parent != root_path else model_file.stem
            models.append({
                "path": str(model_file),
                "format": model_file.suffix[1:],
                "name": model_name,
                "size_mb": model_file.stat().st_size / (1024 * 1024)
            })
    
    return models


def run_benchmark_parallel(models: list, video_path: str, output_dir: str, max_workers: int):
    """Запускает бенчмаркинг параллельно"""
    global completed_count, total_count
    
    total_count = len(models)
    completed_count = 0
    
    print("\n" + "=" * 60)
    print(f"🚀 Запуск бенчмаркинга {len(models)} моделей")
    print(f"   Параллельных процессов: {max_workers}")
    print(f"   Видео: {video_path}")
    print("=" * 60)
    
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Запускаем все задачи
        futures = {}
        for i, model in enumerate(models):
            future = executor.submit(inference_worker, model, video_path, output_dir, i)
            futures[future] = model
        
        # Собираем результаты по мере завершения
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ Бенчмаркинг завершён за {format_time(total_elapsed)}")
    print("=" * 60)
    
    return results


def save_results(results: list, output_dir: str):
    """Сохраняет результаты в JSON и CSV"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = output_dir / f"benchmark_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 JSON сохранён: {json_path}")
    
    # Сохраняем CSV
    csv_path = output_dir / f"benchmark_{timestamp}.csv"
    successful = [r for r in results if 'error' not in r]
    
    if successful:
        fieldnames = ['model_name', 'format', 'model_size_mb', 'load_time_ms', 'fps',
                     'total_time_seconds', 'inference_time_avg_ms', 'inference_time_p95_ms',
                     'vram_peak_mb', 'ram_used_mb', 'total_frames']
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in successful:
                row = {k: r.get(k, '') for k in fieldnames}
                writer.writerow(row)
        print(f"📊 CSV сохранён: {csv_path}")
    
    # Выводим таблицу результатов в консоль
    print("\n" + "=" * 80)
    print("📈 РЕЗУЛЬТАТЫ БЕНЧМАРКИНГА")
    print("=" * 80)
    print(f"{'Модель':<30} {'Формат':<10} {'FPS':>8} {'Время':>10} {'VRAM':>10} {'Загрузка':>10}")
    print("-" * 80)
    
    for r in sorted(successful, key=lambda x: x['fps'], reverse=True):
        print(f"{r['model_name'][:28]:<30} {r['format']:<10} "
              f"{r['fps']:>8.1f} {format_time(r['total_time_seconds']):>10} "
              f"{r['vram_peak_mb']:>9.1f}MB {r['load_time_ms']:>9.0f}ms")
    
    print("=" * 80)
    
    # Лучшая модель по FPS
    if successful:
        best = max(successful, key=lambda x: x['fps'])
        print(f"\n🏆 Лучшая по скорости: {best['model_name']} ({best['format']}) - {best['fps']:.1f} FPS")


def main():
    print("=" * 60)
    print("🚀 Бенчмаркинг моделей детекции")
    print("=" * 60)
    
    # Проверяем CUDA
    if torch.cuda.is_available():
        print(f"📟 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("📟 CUDA не доступна, используем CPU")
    
    # Проверяем пути
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Видео не найдено: {VIDEO_PATH}")
        sys.exit(1)
    
    if not os.path.exists(MODELS_ROOT):
        print(f"❌ Папка с моделями не найдена: {MODELS_ROOT}")
        sys.exit(1)
    
    # Ищем модели
    print(f"\n📁 Поиск моделей в: {MODELS_ROOT}")
    models = find_all_models(MODELS_ROOT)
    print(f"   Найдено моделей: {len(models)}")
    
    if not models:
        print("   Модели не найдены")
        sys.exit(1)
    
    # Оцениваем время
    estimate_all_models(models, VIDEO_PATH)
    
    # Запрашиваем подтверждение
    print("\n⏳ Запустить бенчмаркинг? (y/n): ", end="")
    response = input().strip().lower()
    if response != 'y':
        print("Отменено.")
        sys.exit(0)
    
    # Запускаем бенчмаркинг
    results = run_benchmark_parallel(models, VIDEO_PATH, OUTPUT_DIR, PARALLEL_MODELS)
    
    # Сохраняем результаты
    save_results(results, OUTPUT_DIR)


if __name__ == "__main__":
    # Для multiprocessing на Windows нужно это, но на Linux и так работает
    # На всякий случай оставляем
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()