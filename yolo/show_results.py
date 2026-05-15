"""
Показывает результаты всех обученных экспериментов:
  - Таблицу метрик
  - Список сохранённых файлов (веса, графики, ONNX)
  - Открывает графики в браузере (опционально)

Запуск:
    python yolo/show_results.py
    python yolo/show_results.py --open-plots   # открыть графики в браузере
    python yolo/show_results.py --run yolov8n  # только один эксперимент
"""

import argparse
import os
import webbrowser
from pathlib import Path


RUNS_DIR = Path("./yolo/runs")

# Какие файлы ищем в папке эксперимента
WEIGHT_FILES = ["weights/best.pt", "weights/last.pt"]
EXPORT_FILES = ["weights/best.onnx", "weights/best.torchscript",
                "weights/best_openvino_model", "weights/best.engine"]
PLOT_FILES   = [
    "results.png",          # loss + mAP по эпохам
    "confusion_matrix.png", # матрица ошибок
    "confusion_matrix_normalized.png",
    "PR_curve.png",         # Precision-Recall кривая
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
    "val_batch0_pred.jpg",  # пример предсказаний на val
    "val_batch1_pred.jpg",
    "val_batch0_labels.jpg",# ground truth для сравнения
]


def sizeof_fmt(num: int) -> str:
    for unit in ("B","KB","MB","GB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def read_metrics(run_dir: Path) -> dict:
    """Читает последнюю строку results.csv → метрики."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    try:
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) < 2:
            return {}
        header = [h.strip() for h in lines[0].split(",")]
        last   = [v.strip() for v in lines[-1].split(",")]
        return dict(zip(header, last))
    except Exception:
        return {}


def show_run(run_dir: Path, open_plots: bool = False) -> dict:
    name = run_dir.name
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")

    # Метрики из results.csv
    metrics = read_metrics(run_dir)
    MAP_KEY  = "         metrics/mAP50(B)"   # ultralytics добавляет пробелы
    MAP95_KEY= "      metrics/mAP50-95(B)"
    P_KEY    = "   metrics/precision(B)"
    R_KEY    = "      metrics/recall(B)"

    def get_metric(d, *keys):
        for k in keys:
            for dk in d:
                if dk.strip() == k.strip():
                    try: return float(d[dk])
                    except: pass
        return None

    mAP50 = get_metric(metrics, "metrics/mAP50(B)")
    mAP95 = get_metric(metrics, "metrics/mAP50-95(B)")
    prec  = get_metric(metrics, "metrics/precision(B)")
    rec   = get_metric(metrics, "metrics/recall(B)")

    if mAP50 is not None:
        print(f"  mAP50:      {mAP50:.4f}")
        print(f"  mAP50-95:   {mAP95:.4f}" if mAP95 else "")
        print(f"  Precision:  {prec:.4f}" if prec else "")
        print(f"  Recall:     {rec:.4f}" if rec else "")
    else:
        print("  Метрики не найдены (обучение ещё идёт или упало)")

    # Веса
    print(f"\n  Веса:")
    for rel in WEIGHT_FILES:
        p = run_dir / rel
        if p.exists():
            print(f"    ✅ {rel:35s} {sizeof_fmt(p.stat().st_size)}")
        else:
            print(f"    ·  {rel}")

    # Экспортированные форматы
    found_exports = []
    for rel in EXPORT_FILES:
        p = run_dir / rel
        if p.exists():
            found_exports.append((rel, p.stat().st_size))
    if found_exports:
        print(f"\n  Экспорт:")
        for rel, sz in found_exports:
            print(f"    ✅ {rel:35s} {sizeof_fmt(sz)}")
    else:
        print(f"\n  Экспорт: не найден (запусти export отдельно)")

    # Графики
    found_plots = []
    for rel in PLOT_FILES:
        p = run_dir / rel
        if p.exists():
            found_plots.append(p)

    print(f"\n  Графики ({len(found_plots)}):")
    for p in found_plots:
        print(f"    📊 {p.name}")
        if open_plots:
            webbrowser.open(str(p.resolve()))

    return {
        "name": name,
        "mAP50": mAP50, "mAP50_95": mAP95,
        "precision": prec, "recall": rec,
        "has_best_pt": (run_dir / "weights/best.pt").exists(),
        "exports": [r for r, _ in found_exports],
        "plots": [p.name for p in found_plots],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir",   default=str(RUNS_DIR))
    parser.add_argument("--run",        help="Показать только этот эксперимент")
    parser.add_argument("--open-plots", action="store_true",
                        help="Открыть все графики в браузере")
    parser.add_argument("--summary",    action="store_true",
                        help="Только сводная таблица без деталей")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"❌ Папка runs не найдена: {runs_dir}")
        print("   Сначала запусти train.py или train_parallel.py")
        return

    # Находим папки экспериментов
    run_dirs = sorted([
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "results.csv").exists()
    ])

    if args.run:
        run_dirs = [d for d in run_dirs if d.name == args.run]

    if not run_dirs:
        print(f"Нет завершённых экспериментов в {runs_dir}")
        return

    print(f"Найдено экспериментов: {len(run_dirs)}")

    all_results = []
    for rd in run_dirs:
        r = show_run(rd, open_plots=args.open_plots)
        all_results.append(r)

    # Сводная таблица
    valid = [r for r in all_results if r["mAP50"] is not None]
    if valid:
        print(f"\n{'='*65}")
        print(f"{'Модель':<18} {'mAP50':>7} {'mAP50-95':>9} {'P':>7} {'R':>7}  Веса")
        print(f"{'─'*65}")
        for r in sorted(valid, key=lambda x: x["mAP50"] or 0, reverse=True):
            pt = "✅" if r["has_best_pt"] else "❌"
            p  = f"{r['precision']:.4f}" if r["precision"] else "  —   "
            rc = f"{r['recall']:.4f}"    if r["recall"]    else "  —   "
            print(f"{r['name']:<18} {r['mAP50']:>7.4f} "
                  f"{(r['mAP50_95'] or 0):>9.4f} {p:>7} {rc:>7}  {pt}")
        print(f"{'='*65}")

    print(f"\nЧтобы открыть графики:")
    print(f"  python yolo/show_results.py --open-plots")
    print(f"  python yolo/show_results.py --run yolov8n --open-plots")


if __name__ == "__main__":
    main()