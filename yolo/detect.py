"""
Детектирование бейджей с помощью обученной YOLOv8 модели.
"""
import cv2
import numpy as np
import sys
from pathlib import Path


class YoloDetector:
    def __init__(self, weights_path: str, conf_threshold: float = 0.5):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Установи: pip install ultralytics")
        self.model = YOLO(weights_path)
        self.conf = conf_threshold

    def detect(self, img: np.ndarray) -> list[dict]:
        """Возвращает список {'bbox': (x1,y1,x2,y2), 'score': float, 'class': str}."""
        results = self.model(img, conf=self.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf[0])
            cls = results.names[int(box.cls[0])]
            detections.append({"bbox": (x1, y1, x2, y2), "score": score, "class": cls})
        return detections

    def draw(self, img: np.ndarray, detections: list[dict]) -> np.ndarray:
        out = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            score = det["score"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 100, 0), 2)
            label = f"{det['class']} {score:.2f}"
            cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        return out

    def detect_webcam(self):
        """Детектирование в реальном времени с веб-камеры."""
        cap = cv2.VideoCapture(0)
        print("Нажми Q для выхода")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            dets = self.detect(frame)
            frame = self.draw(frame, dets)
            cv2.imshow("Badge Detector (YOLOv8)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    weights = sys.argv[1] if len(sys.argv) > 1 else "./yolo/runs/badge_detector/weights/best.pt"
    mode = sys.argv[2] if len(sys.argv) > 2 else "image"

    detector = YoloDetector(weights)

    if mode == "webcam":
        detector.detect_webcam()
    else:
        img_path = sys.argv[2] if len(sys.argv) > 2 else "test.jpg"
        img = cv2.imread(img_path)
        if img is None:
            print(f"Не удалось открыть: {img_path}")
            sys.exit(1)
        dets = detector.detect(img)
        print(f"Найдено: {len(dets)} бейджей")
        for d in dets:
            print(f"  {d['bbox']}  score={d['score']:.3f}")
        result = detector.draw(img, dets)
        cv2.imwrite("result_yolo.jpg", result)
        print("Результат: result_yolo.jpg")