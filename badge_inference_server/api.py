# api.py
"""
FastAPI маршруты для веб-сервера
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import time
from typing import Optional

from model import ModelInference, FPSMeter
from utils import draw_detections, encode_frame_to_jpeg, decode_jpeg_to_frame
from html_templates import HTML_PAGE


def create_app(model: ModelInference, fps_meter: FPSMeter) -> FastAPI:
    """
    Создание FastAPI приложения с маршрутами
    
    Args:
        model: Экземпляр класса ModelInference
        fps_meter: Экземпляр класса FPSMeter
    
    Returns:
        FastAPI приложение
    """
    app = FastAPI(title="Badge Detection API", version="1.0.0")
    
    # Настройка CORS для доступа с любых устройств
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Хранилище для статистики
    inference_times = []
    
    @app.get("/")
    async def root():
        """Главная страница"""
        return HTMLResponse(HTML_PAGE)
    
    @app.get("/health")
    async def health():
        """Health check для мониторинга"""
        return {"status": "ok", "model_loaded": model.model is not None}
    
    @app.post("/detect")
    async def detect(file: UploadFile = File(...)):
        """
        Принимает кадр, возвращает кадр с детекциями
        """
        # Чтение и декодирование изображения
        contents = await file.read()
        frame = decode_jpeg_to_frame(contents)
        
        if frame is None:
            return {"error": "Invalid image"}
        
        # Инференс
        start_time = time.perf_counter()
        detections = model.inference(frame)
        inference_time = (time.perf_counter() - start_time) * 1000
        inference_times.append(inference_time)
        
        # Ограничиваем размер списка
        if len(inference_times) > 100:
            inference_times.pop(0)
        
        # Обновляем FPS
        fps = fps_meter.update()
        
        # Рисуем детекции
        display_frame = draw_detections(frame, detections, fps, show_fps=True)
        
        # Кодируем в JPEG
        jpeg_bytes = encode_frame_to_jpeg(display_frame, quality=85)
        
        # Возвращаем с заголовками для статистики
        return StreamingResponse(
            BytesIO(jpeg_bytes),
            media_type="image/jpeg",
            headers={
                "X-FPS": str(round(fps, 1)),
                "X-Detections": str(len(detections)),
                "X-Inference-Time": str(round(inference_time, 1))
            }
        )
    
    @app.get("/info")
    async def info():
        """Информация о модели"""
        return {
            "model_type": model.model_type,
            "device": model.device,
            "backend": getattr(model, 'backend_used', 'auto'),
            "conf_threshold": model.conf_threshold,
            "classes": model.class_names,
            "input_shape": model.input_shape
        }
    
    @app.get("/stats")
    async def stats():
        """Статистика производительности"""
        avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
        return {
            "avg_inference_ms": round(avg_time, 1),
            "min_inference_ms": round(min(inference_times), 1) if inference_times else 0,
            "max_inference_ms": round(max(inference_times), 1) if inference_times else 0,
            "current_fps": round(fps_meter.get_fps(), 1),
            "total_frames_processed": fps_meter.frame_count,
            "samples": len(inference_times)
        }
    
    return app