import numpy as np

from constants import feature_size, nb_class, real_image_size

def get_final_coords(boxes_input):
    boxes = np.copy(boxes_input)
    for i in range(feature_size):
        for j in range(feature_size):
            boxes[0, i, j][0] += j
            boxes[0, i, j][1] += i
            boxes[0, i, j][2] += j
            boxes[0, i, j][3] += i
    return boxes

def get_boxes(preds):
    boxes = preds.numpy()
    boxes = get_final_coords(boxes)
    boxes = boxes.reshape((-1, 5 + nb_class))

    probs = np.reshape(boxes[:, 4], (-1))
    sorted_boxes_ids = np.argsort(-1 * probs)
    boxes = boxes[sorted_boxes_ids]

    boxes = boxes[:6]
    output = []

    for b in boxes:
        label = np.argmax(b[5:5+nb_class] - 5)
        xmin = b[0]
        ymin = b[1]
        xmax = b[2]
        ymax = b[3]

        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        box = {
            "x": x * real_image_size[0] / feature_size,
            "y": y * real_image_size[1] / feature_size,
            "width": width * real_image_size[0] / feature_size,
            "height": height * real_image_size[1] / feature_size,
            "label": label,
        }
        output.append(box)

    return output
