import numpy as np

from constants import feature_size, nb_class, real_image_size, max_boxes, overlap_thresh

def non_max_suppression_fast(input_boxes):
    boxes = np.copy(input_boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)

    picked_boxes = []

    while (len(boxes) > 0) & (len(picked_boxes) < max_boxes):
        current_box = boxes[0]
        current_area = area[0]
        picked_boxes.append(current_box)
        boxes = boxes[1:]
        area = area[1:]

        xx1_intersection = np.maximum(boxes[:, 0], current_box[0])
        yy1_intersection = np.maximum(boxes[:, 1], current_box[1])
        xx2_intersection = np.minimum(boxes[:, 2], current_box[2])
        yy2_intersection = np.minimum(boxes[:, 3], current_box[3])

        ww_intersection = np.maximum(0, xx2_intersection - xx1_intersection)
        hh_intersection = np.maximum(0, yy2_intersection - yy1_intersection)

        area_intersection = ww_intersection * hh_intersection

        area_union = current_area + area - area_intersection

        overlap = area_intersection / area_union # division of ints resulting in a float

        ids_to_delete = np.where(overlap > overlap_thresh)[0]
        boxes = np.delete(boxes, ids_to_delete, axis=0)
        area = np.delete(area, ids_to_delete, axis=0)

    return picked_boxes

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

    boxes = boxes[boxes[:, 0] < boxes[:, 2]]
    boxes = boxes[boxes[:, 1] < boxes[:, 3]]

    high_certitude_boxes = np.where(boxes[:, 4] > 10)[0]
    boxes = boxes[high_certitude_boxes]

    boxes = non_max_suppression_fast(boxes)

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
