import numpy as np

from constants import feature_size, blob_size

statuses = {
    "normal": 0,
    "hat": 1,
    "ghost": 2,
}

s = blob_size // 2

def get_localization_data(picture_data):
    true_labels = np.zeros((1, feature_size, feature_size), dtype=np.int)
    true_boxes = np.zeros((1, feature_size, feature_size, 4), dtype=np.float32)
    true_probs = np.zeros((1, feature_size, feature_size, 1), dtype=np.float32)
    all_blobs = picture_data["army"] + picture_data["enemy"]
    blob_ids = [j for j, b in enumerate(all_blobs) if (b["alive"] == True)]
    for blob_id in blob_ids:
        blob = all_blobs[blob_id]
        x = int(blob["x"] * feature_size)
        y = int(blob["y"] * feature_size)
        label = 1 + statuses[blob["status"]] + int(blob_id >= 3) * 3
        for a in np.arange(y - s, y + s):
            for b in np.arange(x - s, x + s):
                if (a >= 0) & (a < feature_size) & (b >= 0) & (b < feature_size):
                    true_labels[0][a][b] = label
                    true_boxes[0][a][b] = [x - s - b, y - s - a, x + s - b, y + s - a]
                    true_probs[0][a][b] = [1]
    return true_labels, true_boxes, true_probs
