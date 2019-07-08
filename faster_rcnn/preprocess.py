import numpy as np

from constants import feature_size

statuses = {
    "normal": 0,
    "hat": 1,
    "ghost": 2,
}

def get_localization_data(picture_data):
    target = np.zeros((feature_size, feature_size, 1), dtype=np.int)
    all_blobs = picture_data["army"] + picture_data["enemy"]
    blob_ids = [j for j, b in enumerate(all_blobs) if (b["alive"] == True)]
    for blob_id in blob_ids:
        blob = all_blobs[blob_id]
        x = int(blob["x"] * feature_size)
        y = int(blob["y"] * feature_size)
        label = 1 + statuses[blob["status"]]
        if (blob_id >= 3):
            label += 3
        for a in np.arange(x - 4, x + 5):
            for b in np.arange(y - 4, y + 4):
                if (a >= 0) & (a < feature_size) & (b >= 0) & (b < feature_size):
                    target[a][b] = label
    return target
