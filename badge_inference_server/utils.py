# utils.py
"""
Вспомогательные функции для отрисовки и работы с изображениями
"""
import cv2
import numpy as np
from typing import List
from model import Detection


def draw_detections(frame: np.ndarray, detections: List[Detection], 
                   fps: float, show_fps: bool = True) -> np.ndarray:
    """
    Рисует детекции на кадре
    
    Args:
        frame: Исходный кадр
        detections: Список детекций
        fps: Текущий FPS
        show_fps: Показывать FPS на кадре
    
    Returns:
        Кадр с нарисованными детекциями
    """
    display = frame.copy()
    
    # Рисуем каждый обнаруженный объект
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        
        # Цвет зависит от уверенности (зеленый -> желтый -> красный)
        green = int(255 * det.confidence)
        color = (0, green, 255 - green)
        
        # Рисуем прямоугольник
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        
        # Подготовка текста
        label = f"{det.class_name} {det.confidence:.2f}"
        
        # Рисуем фон для текста
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        
        # Рисуем текст
        cv2.putText(display, label, (x1 + 2, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    # HUD информация
    if show_fps:
        hud_lines = [
            f"FPS: {fps:.1f}",
            f"Badges: {len(detections)}",
        ]
        
        for i, line in enumerate(hud_lines):
            y = 30 + i * 25
            # Тень
            cv2.putText(display, line, (11, y+1),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
            # Основной текст
            cv2.putText(display, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
    
    return display


def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """
    Кодирует кадр в JPEG
    
    Args:
        frame: Кадр в формате BGR
        quality: Качество JPEG (0-100)
    
    Returns:
        JPEG байты
    """
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()


def decode_jpeg_to_frame(jpeg_bytes: bytes) -> np.ndarray:
    """
    Декодирует JPEG в кадр
    
    Args:
        jpeg_bytes: JPEG байты
    
    Returns:
        Кадр в формате BGR
    """
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)