#!/usr/bin/env python3
"""
Детектирование бейджей в реальном времени с веб-камеры.
Поддерживает форматы: .pt (PyTorch), .onnx, .torchscript
Использует GPU если доступен.

Запуск:
    python webcam_inference.py --weights best.pt
    python webcam_inference.py --weights model.onnx --conf 0.6
    python webcam_inference.py --weights model.torchscript --device cuda
    python webcam_inference.py --weights best.pt --backend tensorrt  # Экспериментально
"""
import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Union, Optional, Tuple, List, Any

# Проверка зависимостей
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch не установлен. Установите: pip install torch")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️ Ultralytics не установлен. Установите: pip install ultralytics")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX Runtime не установлен. Установите: pip install onnxruntime-gpu")


@dataclass
class Detection:
    """Класс для хранения информации о детекции"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


class FPSMeter:
    """Измеритель FPS с плавным усреднением"""
    def __init__(self, alpha: float = 0.95):
        self.alpha = alpha
        self.fps = 0
        self.last_time = time.perf_counter()
        self.frame_count = 0
    
    def update(self) -> float:
        """Обновляет FPS и возвращает текущее значение"""
        current_time = time.perf_counter()
        delta = current_time - self.last_time
        self.last_time = current_time
        self.frame_count += 1
        
        if delta > 0:
            current_fps = 1.0 / delta
            self.fps = self.alpha * self.fps + (1 - self.alpha) * current_fps
        
        return self.fps
    
    def get_fps(self) -> float:
        return self.fps


class ModelInference:
    """Универсальный класс для инференса с поддержкой разных форматов"""
    
    def __init__(self, weights_path: str, conf_threshold: float = 0.5, 
                 device: str = "auto", backend: str = "auto"):
        """
        Инициализация модели
        
        Args:
            weights_path: Путь к файлу весов
            conf_threshold: Порог уверенности
            device: Устройство (auto, cpu, cuda, mps)
            backend: Бэкенд для ONNX (auto, cuda, tensorrt, openvino, cpu)
        """
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.device = self._setup_device(device)
        self.model_type = self._detect_model_type()
        self.model = None
        self.input_shape = (640, 640)  # Стандартный размер входа YOLO
        self.class_names = {0: "badge"}  # Дефолтное имя класса
        
        # Загрузка модели
        self._load_model(backend)
        
    def _setup_device(self, device: str) -> str:
        """Настройка устройства для инференса"""
        if device != "auto":
            return device
            
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ Использую GPU: {device_name}")
            print(f"   Видеопамять: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return "cuda"
        elif TORCH_AVAILABLE and hasattr(torch, 'mps') and torch.mps.is_available():
            print("✅ Использую Apple MPS (Metal Performance Shaders)")
            return "mps"
        else:
            print("⚠️ GPU не найден, использую CPU")
            return "cpu"
    
    def _detect_model_type(self) -> str:
        """Определение типа модели по расширению файла"""
        ext = self.weights_path.suffix.lower()
        
        if ext == '.onnx':
            return 'onnx'
        elif ext == '.torchscript' or '.torchscript' in str(self.weights_path).lower():
            return 'torchscript'
        elif ext == '.pt':
            return 'pytorch'
        else:
            raise ValueError(f"Неподдерживаемый формат: {ext}. Используйте .pt, .onnx или .torchscript")
    
    def _load_model(self, backend: str):
        """Загрузка модели в зависимости от типа"""
        print(f"\n🔧 Загрузка модели: {self.weights_path}")
        print(f"   Тип: {self.model_type.upper()}")
        print(f"   Устройство: {self.device.upper()}")
        
        if self.model_type == 'pytorch':
            self._load_pytorch_model()
        elif self.model_type == 'onnx':
            self._load_onnx_model(backend)
        elif self.model_type == 'torchscript':
            self._load_torchscript_model()
        
        # Прогрев модели
        self._warmup()
        print("✅ Модель готова к работе\n")
    
    def _load_pytorch_model(self):
        """Загрузка PyTorch модели через Ultralytics"""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Установите ultralytics: pip install ultralytics")
        
        self.model = YOLO(str(self.weights_path))
        
        # Перемещение на GPU если нужно
        if self.device == 'cuda':
            self.model.to('cuda')
        elif self.device == 'mps':
            self.model.to('mps')
        
        # Получение имён классов
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
        print(f"   Классы: {self.class_names}")
    
    def _load_onnx_model(self, backend: str):
        """Загрузка ONNX модели"""
        if not ONNX_AVAILABLE:
            raise ImportError("Установите onnxruntime: pip install onnxruntime-gpu")
        
        # Настройка провайдеров для ONNX Runtime
        available_providers = ort.get_available_providers()
        print(f"   Доступные провайдеры: {available_providers}")
        
        if backend == "auto":
            # Выбираем лучший доступный провайдер
            preferred_order = [
                'TensorrtExecutionProvider',  # TensorRT (быстрее всего)
                'CUDAExecutionProvider',       # CUDA
                'OpenVINOExecutionProvider',   # Intel OpenVINO
                'CPUExecutionProvider'         # CPU
            ]
            providers = [p for p in preferred_order if p in available_providers]
        elif backend == "cuda":
            providers = ['CUDAExecutionProvider']
        elif backend == "tensorrt":
            providers = ['TensorrtExecutionProvider']
        elif backend == "openvino":
            providers = ['OpenVINOExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        if not providers:
            providers = ['CPUExecutionProvider']
            print("   ⚠️ GPU провайдеры не найдены, использую CPU")
        
        # Создание сессии
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.model = ort.InferenceSession(
            str(self.weights_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # Получение информации о входе
        input_info = self.model.get_inputs()[0]
        if 'shape' in dir(input_info):
            self.input_shape = (input_info.shape[2], input_info.shape[3])
        
        print(f"   Провайдер: {self.model.get_providers()[0]}")
        print(f"   Входной размер: {self.input_shape}")
    
    def _load_torchscript_model(self):
        """Загрузка TorchScript модели"""
        if not TORCH_AVAILABLE:
            raise ImportError("Установите torch: pip install torch")
        
        # Загрузка модели
        self.model = torch.jit.load(str(self.weights_path), map_location='cpu')
        
        # Перемещение на устройство
        if self.device == 'cuda':
            self.model = self.model.cuda()
        elif self.device == 'mps':
            self.model = self.model.to('mps')
        
        self.model.eval()
        
        # Попытка получить имена классов (если сохранены)
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
    
    def _warmup(self):
        """Прогрев модели для стабильной производительности"""
        print("   Прогрев модели...")
        dummy_input = np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8)
        
        for _ in range(3):
            _ = self.inference(dummy_input)
        
        if self.device == 'cuda' and TORCH_AVAILABLE:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    def inference(self, frame: np.ndarray) -> List[Detection]:
        """
        Запуск инференса на одном кадре
        
        Args:
            frame: Изображение в формате BGR (numpy array)
            
        Returns:
            Список детекций
        """
        if self.model_type == 'pytorch':
            return self._inference_pytorch(frame)
        elif self.model_type == 'onnx':
            return self._inference_onnx(frame)
        elif self.model_type == 'torchscript':
            return self._inference_torchscript(frame)
        else:
            return []
    
    def _inference_pytorch(self, frame: np.ndarray) -> List[Detection]:
        """Инференс через PyTorch/Ultralytics"""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name
                    ))
        
        return detections
    
    def _inference_onnx(self, frame: np.ndarray) -> List[Detection]:
        """Инференс через ONNX Runtime"""
        # Подготовка изображения
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_shape)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # Добавляем batch dimension
        
        # Инференс
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: img})
        
        # Пост-обработка результатов YOLO
        detections = self._postprocess_yolo_output(outputs[0], frame.shape)
        
        return detections
    
    def _inference_torchscript(self, frame: np.ndarray) -> List[Detection]:
        """Инференс через TorchScript"""
        # Подготовка изображения
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_shape)
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW, добавляем batch
        
        if self.device == 'cuda':
            img = img.cuda()
        elif self.device == 'mps':
            img = img.to('mps')
        
        # Инференс
        with torch.no_grad():
            outputs = self.model(img)
        
        # Перемещаем результат на CPU для обработки
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().numpy()
        
        # Пост-обработка
        detections = self._postprocess_yolo_output(outputs, frame.shape)
        
        return detections
    
    def _postprocess_yolo_output(self, output: np.ndarray, original_shape: Tuple[int, int]) -> List[Detection]:
        """
        Пост-обработка выходов YOLO
        
        Args:
            output: Выход модели shape [batch, num_detections, 6] или [batch, 84, num_detections]
            original_shape: Оригинальный размер изображения (height, width)
        """
        detections = []
        
        # Определяем формат выхода
        if len(output.shape) == 3:
            # Формат [batch, num_classes+5, num_boxes]
            if output.shape[1] == 84:  # YOLOv8: 80 классов + 4 координаты + объектность
                output = output.transpose(0, 2, 1)  # [batch, num_boxes, 84]
            
            # Обработка каждого бокса
            for detection in output[0]:
                if len(detection) < 6:
                    continue
                    
                x1, y1, x2, y2 = detection[:4]
                conf = float(detection[4]) if len(detection) > 4 else float(np.max(detection[4:]))
                class_scores = detection[4:] if len(detection) > 5 else detection[4:5]
                class_id = int(np.argmax(class_scores))
                
                # Фильтрация по уверенности
                if conf < self.conf_threshold:
                    continue
                
                # Масштабирование координат к оригинальному размеру
                h, w = original_shape[:2]
                x1 = int(x1 * w / self.input_shape[1])
                y1 = int(y1 * h / self.input_shape[0])
                x2 = int(x2 * w / self.input_shape[1])
                y2 = int(y2 * h / self.input_shape[0])
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, f"class_{class_id}")
                ))
        
        return detections


def draw_detections(frame: np.ndarray, detections: List[Detection], 
                   fps: float, show_fps: bool = True) -> np.ndarray:
    """Рисует детекции на кадре"""
    display = frame.copy()
    
    # Рисуем каждый обнаруженный бейдж
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        
        # Цвет зависит от уверенности: зелёный → жёлтый → красный
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
            "Q - Exit | S - Screenshot | P - Pause | I - Info",
        ]
        
        for i, line in enumerate(hud_lines):
            y = 28 + i * 24
            # Тень
            cv2.putText(display, line, (11, y+1),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            # Основной текст
            cv2.putText(display, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    
    return display


def show_model_info(model: ModelInference):
    """Отображение информации о модели"""
    info_lines = [
        "="*50,
        "МОДЕЛЬ ЗАГРУЖЕНА",
        f"Тип: {model.model_type.upper()}",
        f"Устройство: {model.device.upper()}",
        f"Порог уверенности: {model.conf_threshold}",
        f"Размер входа: {model.input_shape}",
        f"Классы: {model.class_names}",
        "="*50
    ]
    
    for line in info_lines:
        print(line)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Badge detector — поддержка всех форматов весов с GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", type=str, required=True,
                       help="Путь к весам (.pt, .onnx, .torchscript)")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Порог уверенности (0.0-1.0)")
    parser.add_argument("--camera", type=int, default=0,
                       help="Индекс камеры (0 = встроенная)")
    parser.add_argument("--width", type=int, default=1280,
                       help="Ширина кадра")
    parser.add_argument("--height", type=int, default=720,
                       help="Высота кадра")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Устройство для инференса")
    parser.add_argument("--backend", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "tensorrt", "openvino"],
                       help="Бэкенд для ONNX")
    parser.add_argument("--save", action="store_true",
                       help="Записывать видео в файл output.mp4")
    parser.add_argument("--no-fps", action="store_true",
                       help="Не показывать FPS")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Проверка существования файла весов
    if not Path(args.weights).exists():
        print(f"❌ Файл весов не найден: {args.weights}")
        return
    
    # Инициализация модели
    try:
        model = ModelInference(
            weights_path=args.weights,
            conf_threshold=args.conf,
            device=args.device,
            backend=args.backend
        )
        show_model_info(model)
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return
    
    # Открытие камеры
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть камеру (индекс {args.camera})")
        return
    
    # Настройка разрешения
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷 Камера: {actual_w}x{actual_h}")
    
    # Видео-запись
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output.mp4", fourcc, 20, (actual_w, actual_h))
        print("🔴 Запись видео в output.mp4")
    
    # Инициализация
    fps_meter = FPSMeter()
    paused = False
    screenshot_n = 0
    show_info = True
    inference_times = deque(maxlen=100)
    
    print("\n▶ ЗАПУСК")
    print("   Управление:")
    print("   Q - Выход")
    print("   S - Скриншот")
    print("   P - Пауза")
    print("   I - Информация о производительности")
    print("-"*50)
    
    while True:
        if not paused:
            # Захват кадра
            ret, frame = cap.read()
            if not ret:
                print("❌ Не удалось получить кадр")
                break
            
            # Инференс с замером времени
            start_time = time.perf_counter()
            detections = model.inference(frame)
            inference_time = (time.perf_counter() - start_time) * 1000  # в мс
            inference_times.append(inference_time)
            
            # Обновление FPS
            fps = fps_meter.update()
            
            # Отрисовка
            display_frame = draw_detections(frame, detections, fps, not args.no_fps)
            
            # Дополнительная информация
            if show_info and not args.no_fps:
                avg_time = sum(inference_times) / len(inference_times)
                info_y = actual_h - 60
                
                info_lines = [
                    f"Инференс: {inference_time:.1f} мс (ср: {avg_time:.1f} мс)",
                    f"Детекций: {len(detections)}",
                    f"Устройство: {model.device.upper()}"
                ]
                
                for i, line in enumerate(info_lines):
                    y = info_y + i * 22
                    cv2.putText(display_frame, line, (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            
            # Запись видео
            if writer:
                writer.write(display_frame)
        
        # Отображение
        cv2.imshow("Badge Detector (Multi-Format + GPU)", display_frame)
        
        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_n += 1
            fname = f"screenshot_{screenshot_n:03d}.jpg"
            cv2.imwrite(fname, display_frame)
            print(f"📸 Скриншот сохранён: {fname}")
        elif key == ord('p'):
            paused = not paused
            print("⏸ Пауза" if paused else "▶ Возобновлено")
        elif key == ord('i'):
            show_info = not show_info
            print(f"ℹ️ Детальная информация: {'вкл' if show_info else 'выкл'}")
    
    # Очистка ресурсов
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Статистика
    if inference_times:
        print("\n📊 СТАТИСТИКА ЗА СЕССИЮ:")
        print(f"   Средний FPS: {fps_meter.get_fps():.1f}")
        print(f"   Среднее время инференса: {sum(inference_times)/len(inference_times):.1f} мс")
        print(f"   Мин. время инференса: {min(inference_times):.1f} мс")
        print(f"   Макс. время инференса: {max(inference_times):.1f} мс")
    
    print("\n👋 Выход")


if __name__ == "__main__":
    main()