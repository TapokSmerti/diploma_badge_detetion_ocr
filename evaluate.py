"""
Сравнительная оценка методов детектирования бейджей.
Выводит таблицу: mAP, Precision, Recall, F1, FPS.

Использование:
    python evaluate.py --yolo ./yolo/runs/badge_detector/weights/best.pt
                       --hog  ./hog_svm/model.pkl
                       --data ./yolo/dataset_merged/images/test
                       --labels ./yolo/dataset_merged/labels/test
"""
import argparse
import time
import cv2
import numpy as np
from pathlib import Path


def compute_iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)


def load_ground_truth(label_path: str, img_w: int, img_h: int) -> list:
    boxes = []
    if not Path(label_path).exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = int((cx - bw/2) * img_w)
            y1 = int((cy - bh/2) * img_h)
            x2 = int((cx + bw/2) * img_w)
            y2 = int((cy + bh/2) * img_h)
            boxes.append((x1, y1, x2, y2))
    return boxes


def evaluate_detector(detector_fn, img_dir: str, label_dir: str,
                       iou_thresh: float = 0.5) -> dict:
    """
    Универсальная оценка детектора.
    detector_fn(img) → list of (x1,y1,x2,y2,score)
    """
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    img_files = list(img_dir.glob("*.[jp][pn]g"))

    tp_total = fp_total = fn_total = 0
    all_scores = []
    all_labels = []
    total_time = 0.0

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        label_path = label_dir / (img_path.stem + ".txt")
        gt_boxes = load_ground_truth(str(label_path), w, h)

        t0 = time.perf_counter()
        predictions = detector_fn(img)  # list of (x1,y1,x2,y2,score)
        total_time += time.perf_counter() - t0

        # Матч предсказаний с GT
        matched_gt = set()
        for pred in predictions:
            x1, y1, x2, y2, score = pred
            best_iou = 0
            best_gt = -1
            for j, gt in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = compute_iou((x1, y1, x2, y2), gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = j
            if best_iou >= iou_thresh and best_gt >= 0:
                tp_total += 1
                matched_gt.add(best_gt)
                all_scores.append(score)
                all_labels.append(1)
            else:
                fp_total += 1
                all_scores.append(score)
                all_labels.append(0)

        fn_total += len(gt_boxes) - len(matched_gt)

    precision = tp_total / (tp_total + fp_total + 1e-9)
    recall    = tp_total / (tp_total + fn_total + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    fps       = len(img_files) / (total_time + 1e-9)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fps": fps,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "n_images": len(img_files),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo",   help="Путь к best.pt")
    parser.add_argument("--hog",    help="Путь к model.pkl")
    parser.add_argument("--data",   required=True, help="Папка с тестовыми изображениями")
    parser.add_argument("--labels", required=True, help="Папка с тестовыми разметками")
    parser.add_argument("--iou",    type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=200,
                        help="Ограничение для скользящего окна (медленно)")
    args = parser.parse_args()

    results = {}

    # ── YOLOv8 ──
    if args.yolo:
        from yolo.detect import YoloDetector
        yolo = YoloDetector(args.yolo)
        def yolo_fn(img):
            dets = yolo.detect(img)
            return [(*d["bbox"], d["score"]) for d in dets]
        print("Оцениваем YOLOv8...")
        results["YOLOv8"] = evaluate_detector(yolo_fn, args.data, args.labels, args.iou)

    # ── HOG + SVM ──
    if args.hog:
        import sys
        sys.path.insert(0, ".")
        from hog_svm.detect import HogSvmDetector
        hog_det = HogSvmDetector(args.hog)
        def hog_fn(img):
            dets = hog_det.detect(img, min_conf=0.65, step_ratio=0.15)
            return [(*d["bbox"], d["score"]) for d in dets]
        print("Оцениваем HOG+SVM (медленно на большом датасете)...")
        results["HOG+SVM"] = evaluate_detector(hog_fn, args.data, args.labels, args.iou)

    # ── Таблица ──
    print("\n" + "="*65)
    print(f"{'Метод':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FPS':>8}")
    print("-"*65)
    for name, m in results.items():
        print(f"{name:<12} {m['precision']:>10.3f} {m['recall']:>8.3f} "
              f"{m['f1']:>8.3f} {m['fps']:>8.1f}")
    print("="*65)
    print(f"IoU threshold: {args.iou}")

    # Сохраняем в CSV для диплома
    import csv
    with open("evaluation_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method","precision","recall","f1","fps","tp","fp","fn"])
        w.writeheader()
        for name, m in results.items():
            w.writerow({"method": name, **m})
    print("Результаты сохранены: evaluation_results.csv")


if __name__ == "__main__":
    main()