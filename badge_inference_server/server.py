#!/usr/bin/env python3
# server.py
"""
Главный файл для запуска веб-сервера инференса
"""
import argparse
import sys
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from model import ModelInference, FPSMeter
from api import create_app
import uvicorn


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Badge Detection Web Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
    # Запуск с PyTorch моделью на GPU
    python server.py --weights best.pt --device cuda --port 8000
    
    # Запуск с ONNX моделью
    python server.py --weights model.onnx --backend tensorrt
    
    # Запуск с CPU
    python server.py --weights best.pt --device cpu --host 0.0.0.0 --port 8080
        """
    )
    
    parser.add_argument(
        "--weights", "-w", 
        type=str, 
        required=True,
        help="Путь к файлу весов (.pt, .onnx, .torchscript)"
    )
    
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.5,
        help="Порог уверенности (0.0-1.0), по умолчанию: 0.5"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Устройство для инференса, по умолчанию: auto"
    )
    
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "tensorrt", "openvino"],
        help="Бэкенд для ONNX моделей, по умолчанию: auto"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Адрес для запуска сервера, по умолчанию: 0.0.0.0 (все интерфейсы)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Порт для запуска сервера, по умолчанию: 8000"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Автоматическая перезагрузка при изменении кода (только для разработки)"
    )
    
    return parser.parse_args()


def main():
    """Основная функция запуска"""
    args = parse_args()
    
    # Проверка существования файла весов
    if not Path(args.weights).exists():
        print(f"❌ Файл весов не найден: {args.weights}")
        sys.exit(1)
    
    print("="*60)
    print("🎯 BADGE DETECTION WEB SERVER")
    print("="*60)
    
    # Загрузка модели
    try:
        model = ModelInference(
            weights_path=args.weights,
            conf_threshold=args.conf,
            device=args.device,
            backend=args.backend
        )
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        sys.exit(1)
    
    # Инициализация FPS метра
    fps_meter = FPSMeter()
    
    # Создание FastAPI приложения
    app = create_app(model, fps_meter)
    
    # Информация о запуске
    print("="*60)
    print("🚀 ЗАПУСК СЕРВЕРА")
    print(f"   Адрес: http://{args.host}:{args.port}")
    print(f"   Локальный доступ: http://localhost:{args.port}")
    print(f"   Веса: {args.weights}")
    print(f"   Устройство: {model.device.upper()}")
    print(f"   Порог уверенности: {model.conf_threshold}")
    print("="*60)
    print("\n🌐 Откройте в браузере на ноутбуке:")
    print(f"   http://<IP_ВАШЕГО_СЕРВЕРА>:{args.port}")
    print("\n📋 Доступные эндпоинты:")
    print(f"   /      - Веб-интерфейс")
    print(f"   /detect - POST эндпоинт для детекции")
    print(f"   /info   - Информация о модели")
    print(f"   /stats  - Статистика производительности")
    print(f"   /health - Health check")
    print("\n⏹ Нажмите Ctrl+C для остановки\n")
    
    # Запуск сервера
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()