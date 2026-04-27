"""
Детектор на основе HOG + SVM со скользящим окном и NMS.
"""
import cv2
import numpy as np
import joblib
from pathlib import Path


def non_max_suppression(boxes: list, scores: list, iou_thresh: float = 0.4) -> list:
    """Подавление немаксимальных откликов (NMS)."""
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1 + 1) * np.maximum(0, iy2 - iy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou < iou_thresh]
    return keep


class HogSvmDetector:
    def __init__(self, model_path: str = "./hog_svm/model.pkl"):
        data = joblib.load(model_path)
        self.clf = data["clf"]
        self.hog_params = data["hog_params"]
        self.patch_size = data["patch_size"]  # (w, h)
        p = self.hog_params
        # Создаём HOGDescriptor позиционно, без kwargs
        self.hog = cv2.HOGDescriptor(
            p["winSize"], p["blockSize"], p["blockStride"], p["cellSize"], p["nbins"]
        )

    def detect(
        self,
        img: np.ndarray,
        scale_factor: float = 1.25,
        step_ratio: float = 0.1,
        min_conf: float = 0.6,
        min_size: tuple = (32, 32),
        max_size: tuple = None,
    ) -> list:
        """
        Скользящее окно с пирамидой масштабов.
        Возвращает список {'bbox': (x1,y1,x2,y2), 'score': float}.
        """
        h, w = img.shape[:2]
        pw, ph = self.patch_size
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        candidates = []
        scores_list = []

        scale = 1.0
        while True:
            scaled_w = int(w / scale)
            scaled_h = int(h / scale)
            if scaled_w < pw or scaled_h < ph:
                break

            resized = cv2.resize(gray, (scaled_w, scaled_h))
            step_x = max(1, int(pw * step_ratio))
            step_y = max(1, int(ph * step_ratio))

            for y in range(0, scaled_h - ph + 1, step_y):
                for x in range(0, scaled_w - pw + 1, step_x):
                    patch = resized[y:y + ph, x:x + pw]
                    feat = self.hog.compute(patch).flatten().reshape(1, -1)
                    prob = self.clf.predict_proba(feat)[0][1]
                    if prob >= min_conf:
                        x1 = int(x * scale)
                        y1 = int(y * scale)
                        x2 = int((x + pw) * scale)
                        y2 = int((y + ph) * scale)
                        real_w = x2 - x1
                        real_h = y2 - y1
                        if real_w < min_size[0] or real_h < min_size[1]:
                            continue
                        candidates.append((x1, y1, x2, y2))
                        scores_list.append(float(prob))

            scale *= scale_factor

        keep = non_max_suppression(candidates, scores_list)
        return [{"bbox": candidates[i], "score": scores_list[i]} for i in keep]

    def draw(self, img: np.ndarray, detections: list) -> np.ndarray:
        out = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            score = det["score"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"badge {score:.2f}"
            cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        return out


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print(f"Не удалось открыть: {img_path}")
        sys.exit(1)

    detector = HogSvmDetector()
    dets = detector.detect(img, min_conf=0.65)
    print(f"Найдено: {len(dets)} бейджей")
    result = detector.draw(img, dets)
    cv2.imwrite("result_hog.jpg", result)
    print("Результат сохранён: result_hog.jpg")