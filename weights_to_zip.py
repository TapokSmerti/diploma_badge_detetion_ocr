import zipfile
from pathlib import Path

base = Path("/yolo/runs/detect/yolo/runs")
zip_path = "best_weights.zip"

with zipfile.ZipFile(zip_path, 'w') as zipf:
    for model_dir in base.iterdir():
        if model_dir.is_dir():
            weights_file = model_dir / "weights" / "best.pt"
            if weights_file.exists():
                zipf.write(weights_file, f"{model_dir.name}_best.pt")
                print(f"Added: {model_dir.name}")

print(f"Done! Archive: {zip_path}")