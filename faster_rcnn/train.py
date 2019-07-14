import json
import numpy as np
import tensorflow as tf
from tensorflow.image import decode_jpeg, resize
from tensorflow.io import read_file
from tensorflow.nn import sigmoid_cross_entropy_with_logits, sparse_softmax_cross_entropy_with_logits
from tensorflow.train import AdamOptimizer

tf.enable_eager_execution()

from classifier import Classifier
from constants import image_size, nb_class, feature_size
from feature_mapper import FeatureMapper
from losses import get_localization_loss, get_regression_loss, get_classification_loss
from preprocess import get_localization_data
from roi_utility import rpn_to_roi
from rpn import Rpn
from tracking import save_data

def get_img(img_path):
    img = read_file(img_path)
    img = decode_jpeg(img, channels=3)
    img = resize(img, [image_size, image_size])
    img = 1 - img/255. # We would rather have the whole white void area be full of zeros than ones
    return img

def get_labels_boxes(boxes, target, data_index): # TODO: remove data_index once we're done
    labels_boxes = []
    to_print = []
    for j in range(len(boxes)):
        b = boxes[j]
        x1 = b[0]
        y1 = b[1]
        x2 = b[2]
        y2 = b[3]
        t = np.reshape(target[x1:x2, y1:y2], (-1))
        t = np.delete(t, np.where(t == 0))
        to_print.append(len(t))
        if (len(t) == 0):
            labels_boxes.append(0)
        else:
            (classes, occurences) = np.unique(t, return_counts=True)
            k = np.argmax(occurences)
            labels_boxes.append(classes[k])
    print(data_index, to_print)
    return labels_boxes

def train():
    feature_mapper = FeatureMapper()
    rpn = Rpn()
    classifier = Classifier()

    feature_mapper.load_weights("./weights/feature_mapper")
    rpn.load_weights("./weights/rpn")
    classifier.load_weights("./weights/classifier")

    opt = AdamOptimizer(1e-4)
    with open("../data/data_classification_train.json") as json_file:
        data = json.load(json_file)
    data_index = 0
    while str(data_index) in data:
        raw_data = data[str(data_index)]
        target, regresion_map = get_localization_data(raw_data)
        img = get_img("../pictures/pictures_classification_train/{}.png".format(data_index))
        img = tf.convert_to_tensor([img])

        def get_loss():
            features = feature_mapper(img)
            (rpn_class, regr) = rpn(features)
            boxes, _ = rpn_to_roi(rpn_class, regr)
            classification_logits = classifier(features, boxes)

            labels_boxes = get_labels_boxes(boxes, target, data_index) # Remove data_index once we're done

            localization_loss = get_localization_loss(rpn_class, target)
            regr_loss = get_regression_loss(regr, regresion_map)
            classification_loss = get_classification_loss(classification_logits, labels_boxes)

            save_data(data_index, raw_data, boxes.tolist(), [a.numpy().tolist() for a in classification_logits], labels_boxes)

            return localization_loss + regr_loss + classification_loss

        opt.minimize(get_loss)

        data_index += 1
        if (data_index % 100 == 99):
            feature_mapper.save_weights("./weights/feature_mapper")
            rpn.save_weights("./weights/rpn")
            classifier.save_weights("./weights/classifier")

train()
