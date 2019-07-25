
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from boxes import get_boxes, get_final_box
from classifier import Classifier
from constants import feature_size, real_image_height, real_image_width
from feature_mapper import FeatureMapper
from img import get_img
from regr import Regr
from roi_pooling import RoiPooling
from rpn import Rpn

feature_mapper = FeatureMapper()
rpn = Rpn()
roi_pooling = RoiPooling()
classifier = Classifier()
regr = Regr()

feature_mapper.load_weights("./weights/feature_mapper")
rpn.load_weights("./weights/rpn")
classifier.load_weights("./weights/classifier")
regr.load_weights("./weights/regr")

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

def show(id, img_path, boxes):
    print(boxes)
    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title("YOLO: review image {}".format(id))
    ax = fig.add_axes([0,0,1,1])
    image = plt.imread(img_path)
    plt.imshow(image)

    for b in boxes:
        edgecolor = labels_colors[b["label"]]
        ax.annotate(labels_name[b["label"]], xy=(b["x1"], b["y2"] + 20))
        rect = patches.Rectangle((b["x1"],b["y1"]), b["x2"] - b["x1"], b["y2"] - b["y1"], edgecolor=edgecolor, facecolor='none')
        ax.add_patch(rect)

    plt.show()

def get_prediction(img):
    features = feature_mapper(img)
    rpn_map = rpn(features)

    boxes, probs = get_boxes(rpn_map)
    feature_areas = roi_pooling(features, boxes)

    classification_logits = classifier(feature_areas)
    regression_values = regr(feature_areas)

    return boxes, probs, classification_logits.numpy(), regression_values.numpy()

def show_prediction(data_index):
    img_path = "../pictures/pictures_detect_local_evaluate_100/{}.png".format(data_index)
    img = get_img(img_path)
    boxes, probs, classification_logits, regression_values = get_prediction(img)

    final_boxes = []
    for i in range(len(boxes)):
        if probs[i] > .9:
            x1, y1, x2, y2 = get_final_box(boxes[i], regression_values[i], limit_border=False)
            label = np.argmax(classification_logits[i])
            final_boxes.append({
                "x1": x1 * real_image_width // feature_size,
                "y1": y1 * real_image_height // feature_size,
                "x2": x2 * real_image_width // feature_size,
                "y2": y2 * real_image_height // feature_size,
                "label": label,
            })

    show(data_index, img_path, final_boxes)

index = None
if len(sys.argv) > 1:
    if (sys.argv[1].isdigit()):
        index = eval(sys.argv[1])

if (index == None):
    print("Usage: python show_prediction.py [0-99]")
else:
    show_prediction(str(index))
