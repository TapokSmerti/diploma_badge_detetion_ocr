"""
Обучение HOG + SVM детектора бейджей.
Сохраняет модель в hog_svm/model.pkl
"""
import os
import cv2
import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

PATCH_SIZE = (64, 128)  # (ширина, высота)

# cv2.HOGDescriptor НЕ принимает kwargs — только позиционные аргументы кортежами
HOG_PARAMS = {
    "winSize":     PATCH_SIZE,
    "blockSize":   (16, 16),
    "blockStride": (8, 8),
    "cellSize":    (8, 8),
    "nbins":       9,
}


def make_hog() -> cv2.HOGDescriptor:
    p = HOG_PARAMS
    return cv2.HOGDescriptor(
        p["winSize"], p["blockSize"], p["blockStride"], p["cellSize"], p["nbins"]
    )


def compute_hog(img: np.ndarray) -> np.ndarray:
    """Вычисляет HOG-дескриптор для патча."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.resize(gray, PATCH_SIZE)
    return make_hog().compute(gray).flatten()


def load_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Загружает позитивы и негативы, возвращает (X, y)."""
    data_dir = Path(data_dir)
    X, y = [], []

    for label, folder in ((1, "positives"), (0, "negatives")):
        img_dir = data_dir / folder
        if not img_dir.exists():
            raise FileNotFoundError(f"Папка не найдена: {img_dir}\n"
                                    "Сначала запусти prepare_data.py")
        files = list(img_dir.glob("*.jpg"))
        print(f"  {folder}: {len(files)} файлов")
        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                continue
            feat = compute_hog(img)
            X.append(feat)
            y.append(label)

    return np.array(X), np.array(y)


def train(data_dir: str = "./hog_svm/data", model_path: str = "./hog_svm/model.pkl"):
    print("Загружаем данные...")
    X, y = load_dataset(data_dir)
    print(f"Всего: {len(X)} примеров, размер дескриптора: {X.shape[1]}")
    print(f"Распределение: {y.sum()} позитивов, {(y==0).sum()} негативов")

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale",
                    probability=True, random_state=42)),
    ])

    print("\nКросс-валидация (5 фолдов)...")
    scores = cross_val_score(clf, X, y, cv=5, scoring="f1", n_jobs=-1)
    print(f"F1 (CV): {scores.mean():.3f} +/- {scores.std():.3f}")

    print("\nОбучаем финальную модель...")
    clf.fit(X, y)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump({"clf": clf, "hog_params": HOG_PARAMS, "patch_size": PATCH_SIZE}, model_path)
    print(f"Модель сохранена: {model_path}")

    y_pred = clf.predict(X)
    print("\nОтчёт на train:")
    print(classification_report(y, y_pred, target_names=["background", "badge"]))


if __name__ == "__main__":
    train()