import numpy as np

from constants import feature_size, anchor_size

statuses = {
    "normal": 0,
    "hat": 1,
    "ghost": 2,
}

s = 6

def get_localization_data(picture_data):
    target = np.zeros((feature_size, feature_size), dtype=np.int)
    bounding_box_target = np.zeros((feature_size, feature_size, 4))
    all_blobs = picture_data["army"] + picture_data["enemy"]
    blob_ids = [j for j, b in enumerate(all_blobs) if (b["alive"] == True)]
    for blob_id in blob_ids:
        blob = all_blobs[blob_id]
        x = int(blob["x"] * feature_size)
        y = int(blob["y"] * feature_size)
        label = 1 + statuses[blob["status"]]
        if (blob_id >= 3):
            label += 3
        for a in np.arange(y - s, y + s):
            for b in np.arange(x - s, x + s):
                if (a >= 0) & (a < feature_size) & (b >= 0) & (b < feature_size):
                    target[a][b] = label
                    bounding_box_target[a][b][0] = x
                    bounding_box_target[a][b][1] = y
                    bounding_box_target[a][b][2] = 2 * s
                    bounding_box_target[a][b][3] = 2 * s
    return target, bounding_box_target
