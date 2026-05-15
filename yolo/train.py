"""
Параллельное обучение нескольких YOLO моделей на одном GPU.

Каждый эксперимент запускается в отдельном процессе через multiprocessing.
Волны: эксперименты группируются по max_parallel штук и запускаются пакетами.

Запуск:
    python yolo/train_parallel.py
    python yolo/train_parallel.py --config yolo/experiments.yaml --max-parallel 3
    python yolo/train_parallel.py --only yolov8n yolov8s yolo11n
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Утилиты конфига (те же что в train.py)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = Path(__file__).parent / "experiments.yaml"

TRAIN_PARAMS = {
    "epochs", "imgsz", "batch", "patience", "workers", "device",
    "pretrained", "degrees", "fliplr", "flipud", "scale", "shear",
    "mixup", "copy_paste", "perspective", "hsv_h", "hsv_s", "hsv_v",
    "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
    "box", "cls", "dfl", "close_mosaic", "mosaic", "erasing",
}

# Примерное потребление VRAM (GB) при batch=64, imgsz=640
# Используется только для планирования волн — не влияет на обучение
VRAM_ESTIMATE = {
    "n": 3,   # nano
    "s": 5,   # small
    "m": 8,   # medium
    "l": 14,  # large
    "x": 20,  # xlarge
}


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_yaml(cfg: dict, config_path: Path) -> str:
    raw = cfg.get("data", {}).get("yaml", "")
    root = config_path.resolve().parent.parent
    for candidate in [Path(raw), (root / raw).resolve(), Path(raw).resolve()]:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"data.yaml не найден: {raw}\n"
        f"  Запусти merge_dataset.py"
    )


def build_experiment(cfg: dict, exp: dict) -> dict:
    return {**cfg.get("defaults", {}), **exp}


def estimate_vram(model_name: str) -> int:
    """Оценивает потребление VRAM по суффиксу модели."""
    for suffix, gb in VRAM_ESTIMATE.items():
        # yolov8n.pt → 'n', yolo11s.pt → 's', etc.
        stem = Path(model_name).stem.lower()
        if stem.endswith(suffix):
            return gb
    return 6  # fallback


# ──────────────────────────────────────────────────────────────────────────────
# Функция обучения — запускается в дочернем процессе
# ──────────────────────────────────────────────────────────────────────────────

def _train_worker(
    model_name: str,
    run_name: str,
    data_yaml: str,
    params: dict,
    project: str,
    export_cfg: dict,
    result_queue: mp.Queue,
    gpu_id: int = 0,
):
    """Запускается в отдельном процессе. Результат кладёт в result_queue."""
    # Изолируем GPU для этого процесса через env (на случай multi-GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Перенаправляем stdout в файл чтобы не мешать выводу других процессов
    log_dir = Path(project) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"

    try:
        from ultralytics import YOLO
        import torch

        device = params.get("device", "0")
        if device == "":
            device = "0" if torch.cuda.is_available() else "cpu"

        print(f"[{run_name}] Старт на device={device}, GPU mem free: "
              f"{torch.cuda.mem_get_info(0)[0]/1e9:.1f}GB" if torch.cuda.is_available() else
              f"[{run_name}] Старт на CPU", flush=True)

        model = YOLO(model_name)
        train_kwargs = {k: v for k, v in params.items() if k in TRAIN_PARAMS}
        # device переопределяем — дочерний процесс всегда видит только свой GPU
        train_kwargs["device"] = device

        t0 = time.time()
        results = model.train(
                    data=data_yaml,
                    project=project,
                    name=run_name,
                    save=True,
                    plots=True,       # ← графики results.png, PR_curve.png и др.
                    val=True,
                    verbose=True,     # прогресс виден в терминале
                    exist_ok=True,
                    **train_kwargs,
                )

        elapsed = time.time() - t0
        rd = results.results_dict if hasattr(results, "results_dict") else {}
        metrics = {
            "mAP50":     rd.get("metrics/mAP50(B)",    0.0),
            "mAP50_95":  rd.get("metrics/mAP50-95(B)", 0.0),
            "precision": rd.get("metrics/precision(B)", 0.0),
            "recall":    rd.get("metrics/recall(B)",    0.0),
        }
        best_pt = str(Path(project) / run_name / "weights" / "best.pt")

        # Экспорт
        exported = {}
        if export_cfg and export_cfg.get("enabled") and Path(best_pt).exists():
            try:
                exp_model = YOLO(best_pt)
                for fmt in export_cfg.get("formats", ["onnx"]):
                    kwargs = dict(format=fmt, imgsz=640,
                                  simplify=export_cfg.get("simplify", True))
                    if fmt == "onnx":
                        kwargs["opset"] = export_cfg.get("onnx_opset", 17)
                    path = exp_model.export(**kwargs)
                    exported[fmt] = str(path)
            except Exception as e:
                exported["error"] = str(e)

        result_queue.put({
            "name": run_name, "model": model_name,
            "status": "ok", "best_pt": best_pt,
            "elapsed_min": elapsed / 60,
            "metrics": metrics, "exported": exported,
            "log": str(log_file),
        })
        print(f"[{run_name}] ✅ готово за {elapsed/60:.1f} мин  "
              f"mAP50={metrics['mAP50']:.4f}", flush=True)

    except Exception as e:
        tb = traceback.format_exc()
        result_queue.put({
            "name": run_name, "model": model_name,
            "status": "error", "error": str(e), "traceback": tb,
            "best_pt": "", "elapsed_min": 0,
            "metrics": {"mAP50":0,"mAP50_95":0,"precision":0,"recall":0},
            "exported": {}, "log": str(log_file),
        })
        print(f"[{run_name}] ❌ ошибка: {e}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Планировщик волн
# ──────────────────────────────────────────────────────────────────────────────

def run_waves(experiments: list, data_yaml: str, project: str,
              export_cfg: dict, max_parallel: int, total_vram: int, gpu_id: int) -> list:
    """
    Делит список экспериментов на волны по max_parallel и запускает их.
    Волна = группа процессов которые стартуют одновременно и ждут завершения.
    """
    results = []
    waves = [experiments[i:i+max_parallel] for i in range(0, len(experiments), max_parallel)]

    print(f"\nВсего экспериментов: {len(experiments)}")
    print(f"Волн: {len(waves)} (по {max_parallel} параллельно)")
    print(f"GPU: {gpu_id}  Доступно VRAM: ~{total_vram} GB\n")

    for wave_idx, wave in enumerate(waves):
        est = sum(estimate_vram(e["model"]) for e in wave)
        print(f"{'─'*55}")
        print(f"Волна {wave_idx+1}/{len(waves)}: "
              f"{[e['name'] for e in wave]}  (~{est} GB VRAM)")
        print(f"{'─'*55}")

        q = mp.Queue()
        procs = []

        for exp in wave:
            params = exp.copy()
            p = mp.Process(
                target=_train_worker,
                args=(exp["model"], exp["name"], data_yaml, params,
                      project, export_cfg, q, gpu_id),
                daemon=True,
            )
            p.start()
            procs.append(p)
            # Небольшая задержка чтобы CUDA контексты не конфликтовали при инициализации
            time.sleep(3)

        # Ждём завершения всех процессов в волне
        for p in procs:
            p.join()

        # Собираем результаты
        while not q.empty():
            results.append(q.get())

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Вывод итогов
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(results: list):
    print(f"\n{'='*70}")
    print(f"{'Модель':<14} {'mAP50':>7} {'mAP50-95':>9} {'P':>7} {'R':>7} {'Время':>8} Статус")
    print(f"{'─'*70}")
    for r in sorted(results, key=lambda x: x["metrics"]["mAP50"], reverse=True):
        m = r["metrics"]
        status = "✅" if r["status"] == "ok" else "❌"
        print(f"{r['name']:<14} {m['mAP50']:>7.4f} {m['mAP50_95']:>9.4f} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{r['elapsed_min']:>7.1f}м  {status}")
    print(f"{'='*70}")

    failed = [r for r in results if r["status"] == "error"]
    if failed:
        print(f"\nПровалившиеся ({len(failed)}):")
        for r in failed:
            print(f"  {r['name']}: {r.get('error','?')}")
            print(f"  Лог: {r.get('log','')}")


def save_csv(results: list, out: str = "./yolo/runs/summary.csv"):
    import csv
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name","model","status","mAP50","mAP50_95",
            "precision","recall","elapsed_min","best_pt","log"
        ])
        w.writeheader()
        for r in results:
            w.writerow({
                "name": r["name"], "model": r["model"], "status": r["status"],
                "best_pt": r.get("best_pt",""), "log": r.get("log",""),
                "elapsed_min": f"{r['elapsed_min']:.1f}",
                **{k: f"{v:.4f}" for k, v in r["metrics"].items()},
            })
    print(f"\n📊 Таблица: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default=str(DEFAULT_CONFIG))
    p.add_argument("--only",         nargs="+", metavar="NAME")
    p.add_argument("--skip",         nargs="+", metavar="NAME")
    p.add_argument("--max-parallel", type=int, default=3,
                   help="Сколько моделей обучать одновременно (default: 3)")
    p.add_argument("--total-vram",   type=int, default=48,
                   help="Доступно VRAM в GB (default: 48 для RTX 6000 Ada)")
    p.add_argument("--gpu",          type=int, default=0,
                   help="Индекс GPU (default: 0)")
    p.add_argument("--no-export",    action="store_true")
    return p.parse_args()


def main():
    # ВАЖНО: на Windows multiprocessing требует этой защиты
    mp.set_start_method("spawn", force=True)

    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Конфиг не найден: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    data_yaml = resolve_data_yaml(cfg, config_path)
    project   = cfg.get("defaults", {}).get("project", "./yolo/runs")
    export_cfg = {} if args.no_export else cfg.get("export", {})

    experiments = cfg.get("experiments", [])
    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]
    if args.skip:
        experiments = [e for e in experiments if e["name"] not in args.skip]

    if not experiments:
        print("❌ Нет экспериментов для запуска")
        sys.exit(1)

    # Строим полные параметры (defaults + exp)
    full_experiments = [build_experiment(cfg, e) for e in experiments]

    results = run_waves(
        experiments=full_experiments,
        data_yaml=data_yaml,
        project=project,
        export_cfg=export_cfg,
        max_parallel=args.max_parallel,
        total_vram=args.total_vram,
        gpu_id=args.gpu,
    )

    print_summary(results)
    save_csv(results)


if __name__ == "__main__":
    main()