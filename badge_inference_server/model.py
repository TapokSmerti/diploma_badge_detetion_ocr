# model.py
"""
Класс для инференса модели. Поддерживает .pt, .onnx, .torchscript
"""
import time
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from collections import deque

# Проверка зависимостей
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


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
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.device = self._setup_device(device)
        self.model_type = self._detect_model_type()
        self.model = None
        self.input_shape = (640, 640)
        self.class_names = {0: "badge"}
        self.backend_used = backend
        
        self._load_model(backend)
        
    def _setup_device(self, device: str) -> str:
        """Настройка устройства для инференса"""
        if device != "auto":
            return device
            
        if TORCH_AVAILABLE and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ Использую GPU: {device_name}")
            return "cuda"
        elif TORCH_AVAILABLE and hasattr(torch, 'mps') and torch.mps.is_available():
            print("✅ Использую Apple MPS")
            return "mps"
        else:
            print("⚠️ GPU не найден, использую CPU")
            return "cpu"
    
    def _detect_model_type(self) -> str:
        ext = self.weights_path.suffix.lower()
        if ext == '.onnx':
            return 'onnx'
        elif ext == '.torchscript' or '.torchscript' in str(self.weights_path).lower():
            return 'torchscript'
        elif ext == '.pt':
            return 'pytorch'
        else:
            raise ValueError(f"Неподдерживаемый формат: {ext}")
    
    def _load_model(self, backend: str):
        """Загрузка модели"""
        print(f"\n🔧 Загрузка модели: {self.weights_path}")
        print(f"   Тип: {self.model_type.upper()}")
        print(f"   Устройство: {self.device.upper()}")
        
        if self.model_type == 'pytorch':
            self._load_pytorch_model()
        elif self.model_type == 'onnx':
            self._load_onnx_model(backend)
        elif self.model_type == 'torchscript':
            self._load_torchscript_model()
        
        self._warmup()
        print("✅ Модель готова к работе\n")
    
    def _load_pytorch_model(self):
        """Загрузка PyTorch модели через Ultralytics"""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Установите ultralytics: pip install ultralytics")
        
        self.model = YOLO(str(self.weights_path))
        
        if self.device == 'cuda':
            self.model.to('cuda')
        elif self.device == 'mps':
            self.model.to('mps')
        
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
        print(f"   Классы: {self.class_names}")
    
    def _load_onnx_model(self, backend: str):
        """Загрузка ONNX модели"""
        if not ONNX_AVAILABLE:
            raise ImportError("Установите onnxruntime: pip install onnxruntime-gpu")
        
        available_providers = ort.get_available_providers()
        print(f"   Доступные провайдеры: {available_providers}")
        
        if backend == "auto":
            preferred_order = [
                'TensorrtExecutionProvider',
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
            providers = [p for p in preferred_order if p in available_providers]
        else:
            providers = [f'{backend.upper()}ExecutionProvider' if backend != 'cpu' else 'CPUExecutionProvider']
        
        if not providers:
            providers = ['CPUExecutionProvider']
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.model = ort.InferenceSession(
            str(self.weights_path),
            sess_options=sess_options,
            providers=providers
        )
        
        self.backend_used = self.model.get_providers()[0]
        print(f"   Провайдер: {self.backend_used}")
    
    def _load_torchscript_model(self):
        """Загрузка TorchScript модели"""
        if not TORCH_AVAILABLE:
            raise ImportError("Установите torch: pip install torch")
        
        self.model = torch.jit.load(str(self.weights_path), map_location='cpu')
        
        if self.device == 'cuda':
            self.model = self.model.cuda()
        elif self.device == 'mps':
            self.model = self.model.to('mps')
        
        self.model.eval()
        
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
    
    def _warmup(self):
        """Прогрев модели"""
        print("   Прогрев модели...")
        dummy_input = np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8)
        
        for _ in range(3):
            _ = self.inference(dummy_input)
        
        if self.device == 'cuda' and TORCH_AVAILABLE:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    def inference(self, frame: np.ndarray) -> List[Detection]:
        """Запуск инференса на одном кадре"""
        if self.model_type == 'pytorch':
            return self._inference_pytorch(frame)
        elif self.model_type == 'onnx':
            return self._inference_onnx(frame)
        elif self.model_type == 'torchscript':
            return self._inference_torchscript(frame)
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
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_shape)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: img})
        
        return self._postprocess_yolo_output(outputs[0], frame.shape)
    
    def _inference_torchscript(self, frame: np.ndarray) -> List[Detection]:
        """Инференс через TorchScript"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_shape)
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)
        
        if self.device == 'cuda':
            img = img.cuda()
        elif self.device == 'mps':
            img = img.to('mps')
        
        with torch.no_grad():
            outputs = self.model(img)
        
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().numpy()
        
        return self._postprocess_yolo_output(outputs, frame.shape)
    
    def _postprocess_yolo_output(self, output: np.ndarray, original_shape: Tuple[int, int]) -> List[Detection]:
        """Пост-обработка выходов YOLO"""
        detections = []
        
        if len(output.shape) == 3:
            if output.shape[1] == 84:
                output = output.transpose(0, 2, 1)
            
            for detection in output[0]:
                if len(detection) < 6:
                    continue
                    
                x1, y1, x2, y2 = detection[:4]
                conf = float(detection[4]) if len(detection) > 4 else float(np.max(detection[4:]))
                class_scores = detection[4:] if len(detection) > 5 else detection[4:5]
                class_id = int(np.argmax(class_scores))
                
                if conf < self.conf_threshold:
                    continue
                
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