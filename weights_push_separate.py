import subprocess
from pathlib import Path

base = Path("yolo/runs/detect/yolo/runs")
models = sorted([d for d in base.iterdir() if d.is_dir()])

weight_files = [
    "best.pt",
    "best.torchscript",
    "best.onnx"
]

for i, model in enumerate(models, 1):

    for weight_name in weight_files:
        weight = model / "weights" / weight_name

        if not weight.exists():
            print(f"Skip {model.name}/{weight_name} - file not found")
            continue

        print(f"\n[{i}/{len(models)}] Pushing {model.name}/{weight_name}...")

        # add
        subprocess.run(
            ["git", "add", str(weight)],
            check=True
        )

        # commit
        subprocess.run(
            ["git", "commit", "-m", f"Add {model.name} {weight_name}"],
            check=True
        )

        # push
        subprocess.run(
            ["git", "push"],
            check=True
        )

        print(f"✓ {model.name}/{weight_name} pushed successfully")

print("\n✅ All weights pushed!")