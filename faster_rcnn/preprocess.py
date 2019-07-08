import json
import numpy as np

from constants import feature_size

def get_localisation_data(file_path):
    targets = []
    with open(file_path) as json_file:  
        data = json.load(json_file)
    data_index = 0
    while str(data_index) in data:
        picture_data = data[str(data_index)]
        target = np.zeros((feature_size, feature_size, 1))
        for blob in (picture_data["army"] + picture_data["enemy"]):
            if (blob["alive"] == True):
                x = int(blob["x"] * feature_size)
                y = int(blob["y"] * feature_size)
                x_minus = max(0, x - 1)
                y_minus = max(0, y - 1)
                x_plus = min(feature_size - 1, x + 1)
                y_plus = min(feature_size - 1, y + 1)
                for a in [x_minus, x, x_plus]:
                    for b in [y_minus, y, y_plus]:
                        target[a][b] = 1
        targets.append((data_index, target))
        data_index += 1
    return targets

statuses = {
    "normal": 0,
    "hat": 1,
    "ghost": 2,
}

def get_classification_data(file_path):
    images_data = []
    with open(file_path) as json_file:
        data = json.load(json_file)
    data_index = 0
    while str(data_index) in data:
        picture_data = data[str(data_index)]
        all_blobs = picture_data["army"] + picture_data["enemy"]
        blob_id = [j for j, b in enumerate(all_blobs) if (b["alive"] == True)][0]
        if blob_id >= 3:
            is_enemy = 1
        else:
            is_enemy = 0
        blob = all_blobs[blob_id]
        status = statuses[blob["status"]]
        label = is_enemy * 3 + status
        images_data.append((data_index, label))
        data_index += 1
    return images_data
