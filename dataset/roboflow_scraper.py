from roboflow import Roboflow

file = 'api.txt'
with open(file) as f:
    api_key = f.read()
    print(api_key)
rf = Roboflow(api_key)

workspace = "ai-b8pvg"
project_name = "id-card-l8shn"

project = rf.workspace(workspace).project(project_name)

versions = project.versions()
print(f"Доступные версии: {[v.name for v in versions]}")

dataset = project.version(1).download("yolov8")

print(f"Датасет сохранен в: {dataset.location}")