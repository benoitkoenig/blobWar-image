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
                for a in [x - 2, x, x + 2]:
                    for b in [y - 2, y, y + 2]:
                        if (a >= 0) & (a < feature_size) & (b >= 0) & (b < feature_size):
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

# def get_classification_data(file_path):
#     targets = []
#     with open(file_path) as json_file:
#         data = json.load(json_file)
#     data_index = 0
#     while str(data_index) in data:
#         picture_data = data[str(data_index)]
#         target = np.zeros((feature_size, feature_size, 1))
#         all_blobs = picture_data["army"] + picture_data["enemy"]
#         blob_ids = [j for j, b in enumerate(all_blobs) if (b["alive"] == True)]
#         for blob_id in blob_ids:
#             blob = all_blobs[blob_id]
#             x = int(blob["x"] * feature_size)
#             y = int(blob["y"] * feature_size)
#             class_blob = 1 + statuses[blob["status"]]
#             if (blob_id >= 3):
#                 class_blob += 3
#             for a in [x - 2, x, x + 2]:
#                 for b in [y - 2, y, y + 2]:
#                     if (a >= 0) & (a < feature_size) & (b >= 0) & (b < feature_size):
#                         target[a][b] = class_blob
#         targets.append((data_index, target))
#         data_index += 1
#     return targets
