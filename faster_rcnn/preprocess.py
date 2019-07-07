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
            x = int(blob["x"] * feature_size)
            y = int(blob["y"] * feature_size)
            target[x][y][0] = 1
        targets.append((str(data_index), target))
        data_index += 1
    return targets
