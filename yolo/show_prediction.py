
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from boxes import get_boxes
from img import get_img
from preprocess import get_localization_data
from yolo import Yolo

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
    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title("YOLO: review image {}".format(id))

    #add axes to the image
    ax = fig.add_axes([0,0,1,1])

    # read and plot the image
    image = plt.imread(img_path)
    plt.imshow(image)

    # iterating over the image for different objects
    for b in boxes:
        width = b["width"]
        height = b["height"]
        x = b["x"] - width / 2
        y = b["y"] - height / 2
        edgecolor = labels_colors[b["label"]]
        ax.annotate(labels_name[b["label"]], xy=(x + width - 40, y + 20))
        rect = patches.Rectangle((x,y), width, height, edgecolor=edgecolor, facecolor='none')        
        ax.add_patch(rect)

    plt.show()

def show_prediction(data_index):
    with open("../data/data_detect_local_evaluate_100.json") as json_file:
        data = json.load(json_file)

    if (data_index not in data):
        print("Index {} out of range".format(data_index))
        return

    yolo = Yolo()
    yolo.load_weights("./weights/yolo")

    img_path = "../pictures/pictures_detect_local_evaluate_100/{}.png".format(data_index)
    img = get_img(img_path)
    preds = yolo(img)
    boxes = get_boxes(preds)
    show(data_index, img_path, boxes)

index = None
if len(sys.argv) > 1:
    if (sys.argv[1].isdigit()):
        index = eval(sys.argv[1])

if (index == None):
    print("Usage: python show_prediction.py [0-99]")
else:
    show_prediction(str(index))


