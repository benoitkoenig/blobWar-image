import numpy as np

from blob_mask import blob_mask, blob_mask_dim
from constants import image_height, image_width

def get_true_mask(data):
    all_blobs = data["army"] + data["enemy"]
    all_masks = []
    for blob in all_blobs:
        if (blob["alive"]):
            mask = np.zeros((image_height, image_width, 1), dtype=np.int) - 1
            if (abs(blob["x"] - .5) < .4) & (abs(blob["y"] - .5) < .4):
                y_k = 1
            else:
                y_k = -1

            x_init = int(blob["x"] * 742) - 26
            y_init = int(blob["y"] * 594) - 31

            for i in range(blob_mask_dim[0]):
                for j in range(blob_mask_dim[1]):
                    y = i + y_init
                    x = j + x_init
                    if (x >= 0) & (y >= 0) & (y < image_height) & (x < image_width):
                        if (blob_mask[i][j] == 1):
                            mask[y][x][0] = 1
            all_masks.append((mask, y_k))
    return all_masks
