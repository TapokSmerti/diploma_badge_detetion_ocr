"""
Детектирование бейджей в реальном времени с веб-камеры.
Использует обученную YOLOv8 модель.

Запуск:
    python webcam_inference.py
    python webcam_inference.py --weights yolo/runs/badge_detector/weights/best.pt
    python webcam_inference.py --weights best.pt --conf 0.6 --camera 0
"""
import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from collections import deque


def parse_args():
    parser = argparse.ArgumentParser(description="Badge detector — live webcam inference")
    parser.add_argument("--weights", default="runs/detect/yolo/runs/badge_detector/weights/best.pt",
                        help="Путь к best.pt")
    parser.add_argument("--conf",    type=float, default=0.5,
                        help="Порог уверенности (0.0–1.0)")
    parser.add_argument("--camera",  type=int,   default=0,
                        help="Индекс камеры (0 = встроенная)")
    parser.add_argument("--width",   type=int,   default=1280)
    parser.add_argument("--height",  type=int,   default=720)
    parser.add_argument("--save",    action="store_true",
                        help="Записывать видео в файл output.mp4")
    return parser.parse_args()


def draw_detections(frame: np.ndarray, results, fps: float) -> np.ndarray:
    """Рисует bbox, метки, FPS и счётчик на кадре."""
    h, w = frame.shape[:2]
    det_count = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            label = f"badge {conf:.2f}"
            det_count += 1

            # Цвет зависит от уверенности: зелёный → жёлтый → красный
            green = int(255 * conf)
            color = (0, green, 255 - green)

            # Bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Подложка под текст
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # HUD — верхний левый угол
    hud_lines = [
        f"FPS: {fps:.1f}",
        f"Badges: {det_count}",
        "Q — exit  |  S — screen  |  P — pause",
    ]
    for i, line in enumerate(hud_lines):
        y = 28 + i * 24
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def main():
    args = parse_args()

    # ── Загрузка модели ────────────────────────────────────────────────────
    weights = Path(args.weights)
    if not weights.exists():
        print(f"❌ Файл весов не найден: {weights}")
        print("   Укажи путь через --weights path/to/best.pt")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Установи ultralytics: pip install ultralytics")
        return

    print(f"Загружаю модель: {weights}")
    model = YOLO(str(weights))
    print(f"✅ Модель загружена. Классы: {model.names}")

    # ── Открытие камеры ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть камеру (индекс {args.camera})")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Камера: {actual_w}x{actual_h}")

    # ── Запись видео (опционально) ─────────────────────────────────────────
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output.mp4", fourcc, 20, (actual_w, actual_h))
        print("🔴 Запись в output.mp4")

    # ── FPS: скользящее среднее по 30 кадрам ──────────────────────────────
    fps_buf = deque(maxlen=30)
    t_prev  = time.perf_counter()
    paused  = False
    screenshot_n = 0

    print("\n▶ Запуск. Нажми Q для выхода, S для скриншота, P для паузы.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ Не удалось получить кадр")
                break

            # Инференс
            results = model(frame, conf=args.conf, verbose=False)

            # FPS
            t_now = time.perf_counter()
            fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now
            fps = sum(fps_buf) / len(fps_buf)

            display_frame = draw_detections(frame.copy(), results, fps)

            if writer:
                writer.write(display_frame)

        cv2.imshow("Badge Detector (YOLOv8)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            screenshot_n += 1
            fname = f"screenshot_{screenshot_n:03d}.jpg"
            cv2.imwrite(fname, display_frame)
            print(f"📸 Скриншот сохранён: {fname}")
        elif key == ord("p"):
            paused = not paused
            print("⏸ Пауза" if paused else "▶ Возобновлено")

    cap.release()
    if writer:
        writer.release()
        print("✅ Видео сохранено: output.mp4")
    cv2.destroyAllWindows()
    print("Выход.")


if __name__ == "__main__":
    main()