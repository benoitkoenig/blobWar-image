import json
import matplotlib.pyplot as plt
import numpy as np
import sys

from constants import image_height, image_width
from img import get_img
from preprocess import get_true_mask

def show(id, img_path, true_masks):
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title("Deep Mask: Check mask {}".format(id))
    (mask, _) = true_masks[0]
    mask = mask[:, :, 0]

    plt.subplot(1, 2, 1)
    image = plt.imread(img_path)
    image = np.array(image)
    image[:, :, 0] *= mask
    image[:, :, 1] *= mask
    image[:, :, 2] *= mask
    image = image.tolist()
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    image = plt.imread(img_path)
    image = np.array(image)
    image[:, :, 0] *= 1 - mask
    image[:, :, 1] *= 1 - mask
    image[:, :, 2] *= 1 - mask
    image = image.tolist()
    plt.imshow(image)

    plt.show()

def show_mask(index):
    with open("../data/data_classification_train.json") as json_file:
        data = json.load(json_file)
    if index not in data:
        print("Index {} out of range".format(index))
        return
    img_path = "../pictures/pictures_classification_train/{}.png".format(index)
    true_masks = get_true_mask(data[index])
    show(index, img_path, true_masks)

index = None
if len(sys.argv) > 1:
    if (sys.argv[1].isdigit()):
        index = eval(sys.argv[1])

if (index == None):
    print("Usage: python show_mask.py [0-99]")
else:
    show_mask(str(index))
