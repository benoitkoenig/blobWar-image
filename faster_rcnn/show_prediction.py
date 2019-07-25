
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from boxes import get_boxes, get_final_box
from constants import feature_size, real_image_height, real_image_width, max_boxes
from img import get_img
from prediction import get_prediction, get_prediction_mask

labels_colors = [
    "#000000",
    "#0000FF",
    "#000088",
    "#8888FF",
    "#FF0000",
    "#880000",
    "#FF8888",
]

labels_name = [
    "Nothing",
    "Ally",
    "Ally with hat",
    "Ally ghost",
    "Ennemy",
    "Ennemy with hat",
    "Ennemy ghost",
]

def show(id, img_path, boxes, mask):
    fig = plt.figure(figsize=(20, 10))
    fig.canvas.set_window_title("Faster-RCNN and Mask-RCNN: review image {}".format(id))

    plt.subplot(1, 2, 1)
    image = plt.imread(img_path)
    ax = plt.gca()
    for b in boxes:
        edgecolor = labels_colors[b["label"]]
        ax.annotate(labels_name[b["label"]], xy=(b["x1"], b["y2"] + 20))
        rect = patches.Rectangle((b["x1"],b["y1"]), b["x2"] - b["x1"], b["y2"] - b["y1"], edgecolor=edgecolor, facecolor='none')
        ax.add_patch(rect)
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    image = plt.imread(img_path)
    for i in range(max_boxes):
        specific_mask = np.array(mask != i + 1, dtype=np.int)
        image[:, :, i % 3] = image[:, :, i % 3] * specific_mask
        image[:, :, (i+1) % 3] = image[:, :, (i+1) % 3] * specific_mask
    plt.imshow(image)

    plt.show()

def show_mask(data_index):
    img_path = "../pictures/pictures_detect_local_evaluate_100/{}.png".format(data_index)
    img = get_img(img_path)

    boxes, probs, classification_logits, regression_values = get_prediction(img)
    bounding_boxes = []
    for i in range(len(boxes)):
        if probs[i] > .9:
            x1, y1, x2, y2 = get_final_box(boxes[i], regression_values[i], limit_border=False)
            label = np.argmax(classification_logits[i])
            bounding_boxes.append({
                "x1": x1 * real_image_width // feature_size,
                "y1": y1 * real_image_height // feature_size,
                "x2": x2 * real_image_width // feature_size,
                "y2": y2 * real_image_height // feature_size,
                "label": label,
            })

    mask = get_prediction_mask(img)

    show(data_index, img_path, bounding_boxes, mask)

index = None
if len(sys.argv) > 1:
    if (sys.argv[1].isdigit()):
        index = eval(sys.argv[1])

if (index == None):
    print("Usage: python show_mask.py [0-99]")
else:
    show_mask(str(index))
