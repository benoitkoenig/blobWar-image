import numpy as np

from blob_mask import blob_mask, blob_mask_dim
from constants import feature_size, anchor_size, real_image_width, real_image_height

statuses = {
    "normal": 0,
    "hat": 1,
    "ghost": 2,
}

s = 6
y_offset = 3 # The blob picture is not centered vertically around it's position

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
        for a in np.arange(y - s - y_offset, y + s - y_offset):
            for b in np.arange(x - s, x + s):
                if (a >= 0) & (a < feature_size) & (b >= 0) & (b < feature_size):
                    target[a][b] = label
                    bounding_box_target[a][b][0] = x
                    bounding_box_target[a][b][1] = y - y_offset
                    bounding_box_target[a][b][2] = 2 * s
                    bounding_box_target[a][b][3] = 2 * s
    return target, bounding_box_target

def get_true_mask(data):
    all_blobs = data["army"] + data["enemy"]
    mask = np.zeros((real_image_height, real_image_width), dtype=np.int)
    for (blob_id, blob) in enumerate(all_blobs):
        if (blob["alive"]):
            x_init = int(blob["x"] * 742) - 26
            y_init = int(blob["y"] * 594) - 31

            for i in range(blob_mask_dim[0]):
                for j in range(blob_mask_dim[1]):
                    y = i + y_init
                    x = j + x_init
                    if (x >= 0) & (y >= 0) & (y < real_image_height) & (x < real_image_width):
                        if (blob_mask[i][j] == 1):
                            mask[y][x] = blob_id + 1
    return mask
