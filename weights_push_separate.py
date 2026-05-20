# push_models_one_by_one.py
import subprocess
from pathlib import Path

base = Path("yolo/runs/detect/yolo/runs")
models = sorted([d for d in base.iterdir() if d.is_dir()])

for i, model in enumerate(models, 1):
    weights = model / "weights" / "best.pt"
    if not weights.exists():
        print(f"Skip {model.name} - no best.pt")
        continue
    
    print(f"\n[{i}/{len(models)}] Pushing {model.name}...")
    
    # Добавить файл
    subprocess.run(["git", "add", str(weights)], check=True)
    
    # Коммит
    subprocess.run(["git", "commit", "-m", f"Add {model.name} weights"], check=True)
    
    # Пуш
    subprocess.run(["git", "push"], check=True)
    
    print(f"✓ {model.name} pushed successfully")

print("\n✅ All models pushed!")