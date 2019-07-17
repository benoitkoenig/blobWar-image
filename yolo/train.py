import json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from img import get_img
from loss import calculate_loss
from preprocess import get_localization_data
from yolo import Yolo

def train():
    yolo = Yolo()
    yolo.load_weights("./weights/yolo")

    opt = Adam(learning_rate=5e-5)
    with open("../data/data_detect_local_train.json") as json_file:
        data = json.load(json_file)
    data_index = 0
    while str(data_index) in data:
        img = get_img("../pictures/pictures_detect_local_train/{}.png".format(data_index))
        true_labels, true_boxes, true_preds = get_localization_data(data[str(data_index)])

        def get_loss():
            preds = yolo(img)
            return calculate_loss(preds, true_labels, true_boxes, true_preds)

        opt.minimize(get_loss, [yolo.trainable_weights])

        if (data_index % 100 == 99):
            yolo.save_weights("./weights/yolo")
        data_index += 1

train()
