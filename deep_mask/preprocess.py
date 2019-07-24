import numpy as np

from blob_mask import blob_mask, blob_mask_dim
from constants import image_height, image_width

def get_true_mask(data):
    all_blobs = data["army"] + data["enemy"]
    for blob in all_blobs:
        mask = np.zeros((image_height, image_width, 1), dtype=np.int)
        if (abs(blob["x"] - .5) < .4) & (abs(blob["y"] - .5) < .4):
            y_k = 1
        else:
            y_k = -1

        x_init = int(blob["x"] * 742) - 26
        y_init = int(blob["y"] * 594) - 33

        for i in range(blob_mask_dim[0]):
            for j in range(blob_mask_dim[1]):
                y = i + y_init
                x = j + x_init
                if (x >= 0) & (y >= 0) & (y < image_height) & (x < image_width):
                    mask[y][x][0] = blob_mask[i][j]
    return [(mask, y_k)]
