"""
Параллельное обучение нескольких YOLO моделей.

Каждая модель запускается как отдельный subprocess (не daemon),
поэтому ultralytics может свободно использовать свой DataLoader multiprocessing.

Запуск:
    python yolo/train_parallel.py
    python yolo/train_parallel.py --max-parallel 3
    python yolo/train_parallel.py --only yolov8n yolov8s yolo11n
"""

import argparse
import subprocess
import sys
import time
import json
import os
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).parent / "experiments.yaml"

TRAIN_PARAMS = {
    "epochs", "imgsz", "batch", "patience", "workers", "device",
    "pretrained", "degrees", "fliplr", "flipud", "scale", "shear",
    "mixup", "copy_paste", "perspective", "hsv_h", "hsv_s", "hsv_v",
    "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
    "box", "cls", "dfl", "close_mosaic", "mosaic", "erasing",
}

VRAM_ESTIMATE = {"n": 3, "s": 5, "m": 8, "l": 14, "x": 20}


# ──────────────────────────────────────────────────────────────────────────────
# Конфиг
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_yaml(cfg: dict, config_path: Path) -> str:
    raw = cfg.get("data", {}).get("yaml", "")
    root = config_path.resolve().parent.parent
    for candidate in [(root / raw).resolve(), Path(raw).resolve()]:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"data.yaml не найден: {raw}\nЗапусти merge_dataset.py")


def build_experiment(cfg: dict, exp: dict) -> dict:
    return {**cfg.get("defaults", {}), **exp}


def estimate_vram(model_name: str) -> int:
    stem = Path(model_name).stem.lower()
    for suffix, gb in VRAM_ESTIMATE.items():
        if stem.endswith(suffix):
            return gb
    return 6


# ──────────────────────────────────────────────────────────────────────────────
# Worker-скрипт (запускается как subprocess)
# ──────────────────────────────────────────────────────────────────────────────

# Этот код вставляется в начало каждого subprocess через -c
WORKER_SCRIPT = '''
import sys, json, time, os
from pathlib import Path

args    = json.loads(sys.argv[1])
model_n = args["model"]
run_n   = args["name"]
data_y  = args["data_yaml"]
params  = args["params"]
project = args["project"]
exp_cfg = args.get("export_cfg", {})

TRAIN_PARAMS = {
    "epochs","imgsz","batch","patience","workers","device",
    "pretrained","degrees","fliplr","flipud","scale","shear",
    "mixup","copy_paste","perspective","hsv_h","hsv_s","hsv_v",
    "lr0","lrf","momentum","weight_decay","warmup_epochs",
    "box","cls","dfl","close_mosaic","mosaic","erasing",
}

try:
    from ultralytics import YOLO
    import torch

    device = params.get("device", "0")
    if device == "":
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"[{run_n}] device={device}  model={model_n}", flush=True)

    model = YOLO(model_n)
    train_kwargs = {k: v for k, v in params.items() if k in TRAIN_PARAMS}
    train_kwargs["device"] = device

    t0 = time.time()
    results = model.train(
        data=data_y,
        project=project,
        name=run_n,
        save=True,
        plots=True,
        val=True,
        verbose=True,
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
    best_pt = str(Path(project) / "detect" / project / run_n / "weights" / "best.pt")
    # ultralytics может класть в разные места — ищем best.pt
    for candidate in [
        Path(project) / run_n / "weights" / "best.pt",
        Path(project) / "detect" / project / run_n / "weights" / "best.pt",
        Path(results.save_dir) / "weights" / "best.pt",
    ]:
        if candidate.exists():
            best_pt = str(candidate)
            break

    # Экспорт
    exported = {}
    if exp_cfg.get("enabled") and Path(best_pt).exists():
        exp_model = YOLO(best_pt)
        for fmt in exp_cfg.get("formats", ["onnx"]):
            try:
                kw = dict(format=fmt, imgsz=640, simplify=exp_cfg.get("simplify", True))
                if fmt == "onnx":
                    kw["opset"] = exp_cfg.get("onnx_opset", 17)
                path = exp_model.export(**kw)
                exported[fmt] = str(path)
                print(f"[{run_n}] exported {fmt}: {path}", flush=True)
            except Exception as e:
                exported[fmt] = f"ERROR: {e}"

    result = {
        "name": run_n, "model": model_n, "status": "ok",
        "best_pt": best_pt, "elapsed_min": elapsed / 60,
        "metrics": metrics, "exported": exported,
    }
    print(f"__RESULT__:{json.dumps(result)}", flush=True)

except Exception as e:
    import traceback
    result = {
        "name": run_n, "model": model_n, "status": "error",
        "error": str(e), "traceback": traceback.format_exc(),
        "best_pt": "", "elapsed_min": 0,
        "metrics": {"mAP50":0,"mAP50_95":0,"precision":0,"recall":0},
        "exported": {},
    }
    print(f"__RESULT__:{json.dumps(result)}", flush=True)
    sys.exit(1)
'''


# ──────────────────────────────────────────────────────────────────────────────
# Запуск волны
# ──────────────────────────────────────────────────────────────────────────────

def launch_experiment(exp: dict, data_yaml: str, project: str,
                      export_cfg: dict) -> subprocess.Popen:
    """Запускает один эксперимент как отдельный subprocess."""
    params = {k: v for k, v in exp.items() if k not in ("name", "model")}
    payload = json.dumps({
        "model":      exp["model"],
        "name":       exp["name"],
        "data_yaml":  data_yaml,
        "params":     params,
        "project":    project,
        "export_cfg": export_cfg,
    })
    proc = subprocess.Popen(
        [sys.executable, "-c", WORKER_SCRIPT, payload],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc


def collect_output(name: str, proc: subprocess.Popen) -> dict:
    """Читает stdout процесса, выводит в терминал с префиксом, парсит результат."""
    result = None
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("__RESULT__:"):
            try:
                result = json.loads(line[len("__RESULT__:"):])
            except Exception:
                pass
        else:
            print(f"[{name}] {line}", flush=True)
    proc.wait()
    if result is None:
        result = {
            "name": name, "model": "", "status": "error",
            "error": f"Process exited with code {proc.returncode}, no result found",
            "best_pt": "", "elapsed_min": 0,
            "metrics": {"mAP50":0,"mAP50_95":0,"precision":0,"recall":0},
            "exported": {},
        }
    return result


def run_waves(experiments: list, data_yaml: str, project: str,
              export_cfg: dict, max_parallel: int, total_vram: int) -> list:
    all_results = []
    waves = [experiments[i:i+max_parallel]
             for i in range(0, len(experiments), max_parallel)]

    print(f"\nЭкспериментов: {len(experiments)}  |  Волн: {len(waves)}  |  "
          f"Параллельно: {max_parallel}  |  VRAM: ~{total_vram} GB\n")

    for wi, wave in enumerate(waves):
        est = sum(estimate_vram(e["model"]) for e in wave)
        names = [e["name"] for e in wave]
        print(f"{'─'*60}")
        print(f"Волна {wi+1}/{len(waves)}: {names}  (~{est} GB VRAM)")
        print(f"{'─'*60}")

        # Запускаем все процессы волны
        procs = {}
        for exp in wave:
            proc = launch_experiment(exp, data_yaml, project, export_cfg)
            procs[exp["name"]] = proc
            time.sleep(2)  # небольшая пауза чтобы CUDA контексты не конфликтовали

        # Читаем вывод параллельно через потоки
        import threading
        results_wave = {}

        def read_proc(name, proc):
            results_wave[name] = collect_output(name, proc)

        threads = [
            threading.Thread(target=read_proc, args=(name, proc), daemon=True)
            for name, proc in procs.items()
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for name in [e["name"] for e in wave]:
            r = results_wave.get(name)
            if r:
                status = "✅" if r["status"] == "ok" else "❌"
                m = r["metrics"]
                print(f"\n{status} {name}: mAP50={m['mAP50']:.4f}  "
                      f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                      f"{r['elapsed_min']:.1f}м")
                all_results.append(r)

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Итоги
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(results: list):
    print(f"\n{'='*70}")
    print(f"{'Модель':<14} {'mAP50':>7} {'mAP50-95':>9} {'P':>7} {'R':>7} {'Время':>8}  Статус")
    print(f"{'─'*70}")
    for r in sorted(results, key=lambda x: x["metrics"]["mAP50"], reverse=True):
        m = r["metrics"]
        s = "✅" if r["status"] == "ok" else "❌"
        print(f"{r['name']:<14} {m['mAP50']:>7.4f} {m['mAP50_95']:>9.4f} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{r['elapsed_min']:>7.1f}м  {s}")
    print(f"{'='*70}")

    failed = [r for r in results if r["status"] == "error"]
    if failed:
        print(f"\n❌ Провалилось: {len(failed)}")
        for r in failed:
            print(f"  {r['name']}: {r.get('error', '?')}")


def save_csv(results: list, out: str = "./yolo/runs/summary.csv"):
    import csv
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name","model","status","mAP50","mAP50_95",
            "precision","recall","elapsed_min","best_pt"
        ])
        w.writeheader()
        for r in results:
            w.writerow({
                "name": r["name"], "model": r["model"],
                "status": r["status"], "best_pt": r.get("best_pt",""),
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
    p.add_argument("--max-parallel", type=int, default=3)
    p.add_argument("--total-vram",   type=int, default=48)
    p.add_argument("--device",       help="Переопределить device для всех: 0, cpu, ...")
    p.add_argument("--no-export",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Конфиг не найден: {config_path}")
        sys.exit(1)

    cfg        = load_config(config_path)
    data_yaml  = resolve_data_yaml(cfg, config_path)
    project    = cfg.get("defaults", {}).get("project", "./yolo/runs")
    export_cfg = {} if args.no_export else cfg.get("export", {})

    experiments = cfg.get("experiments", [])
    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]
    if args.skip:
        experiments = [e for e in experiments if e["name"] not in args.skip]

    full = [build_experiment(cfg, e) for e in experiments]
    if args.device:
        for e in full:
            e["device"] = args.device

    print(f"Конфиг:  {config_path}")
    print(f"Датасет: {data_yaml}")
    for e in full:
        print(f"  • {e['name']}  ({e['model']})")

    results = run_waves(full, data_yaml, project, export_cfg,
                        args.max_parallel, args.total_vram)
    print_summary(results)
    save_csv(results)


if __name__ == "__main__":
    main()