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
        description="Badge Detection