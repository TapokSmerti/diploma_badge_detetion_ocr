import os
from roboflow import Roboflow

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths to your files
api_file = os.path.join(script_dir, 'api.txt')
dataset_file = os.path.join(script_dir, 'datasets.txt')

# Load API key
with open(api_file) as f:
    api_key = f.read().strip()

rf = Roboflow(api_key=api_key)

# Read the list of URLs
with open(dataset_file, 'r') as f:
    urls = [line.strip() for line in f if line.strip()]

for url in urls:
    try:
        # Parse workspace and project from the URL
        # URL format: https://universe.roboflow.com/workspace/project
        parts = url.replace('http://', '').replace('https://', '').split('/')
        workspace_id = parts[1]
        project_id = parts[2]
        
        print(f"\n--- Processing: {workspace_id}/{project_id} ---")
        
        # Connect to project
        project = rf.workspace(workspace_id).project(project_id)
        
        # Get the latest version number
        versions = project.versions()
        if not versions:
            print(f"Skipping {project_id}: No versions found.")
            continue
        
        v_id = versions[0].id
        latest_version = v_id.split('/')[-1]
        print(f"Downloading version {latest_version}...")
        
        # Download to a folder named after the project
        location = f"./dataset/{project_id}"
        project.version(latest_version).download("yolov8", location=location)
        
        print(f"Done! Saved to {location}")
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")

print("\nAll downloads complete.")
