import os
import json

# Path to the directory containing the folders and JSON file
base_path = 'ILSVRC2012_img_val_subset'
json_filename = 'imagenet_class_index.json'

# Rename folders
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    if os.path.isdir(folder_path) and folder_name.isdigit():
        new_name = f"{int(folder_name):03d}"
        new_path = os.path.join(base_path, new_name)
        os.rename(folder_path, new_path)
        print(f"Renamed folder {folder_name} → {new_name}")

# Rename JSON keys
json_path = os.path.join(os.path.dirname(__file__), json_filename)
with open(json_path, 'r') as f:
    data = json.load(f)

new_data = {f"{int(k):03d}": v for k, v in data.items()}

with open(json_path, 'w') as f:
    json.dump(new_data, f, indent=4)

print(f"Updated JSON keys in {json_filename}")