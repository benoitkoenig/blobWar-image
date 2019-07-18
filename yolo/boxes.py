import numpy as np

from constants import feature_size, nb_class, real_image_size

def get_boxes(tf_preds):
    preds = tf_preds.numpy()
    boxes = preds[:, :, :, 4]
    probs = np.reshape(preds[:, :, :, 4], (-1))
    sorted_boxes_ids = np.argsort(-1 * probs)

    sorted_boxes_ids = sorted_boxes_ids[:6]
    boxes = []

    for id in sorted_boxes_ids:
        x_tile = id % feature_size
        y_tile = id // feature_size

        label = np.argmax(preds[0, y_tile, x_tile, 5:5+nb_class])

        xmin = preds[0, y_tile, x_tile, 0]
        ymin = preds[0, y_tile, x_tile, 1]
        xmax = preds[0, y_tile, x_tile, 2]
        ymax = preds[0, y_tile, x_tile, 3]

        x = x_tile + (xmin + xmax) / 2
        y = y_tile + (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        box = {
            "x": x * real_image_size[0] / feature_size,
            "y": y * real_image_size[1] / feature_size,
            "width": width * real_image_size[0] / feature_size,
            "height": height * real_image_size[1] / feature_size,
            "label": label,
        }
        boxes.append(box)

    return boxes
