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
from losses import get_localization_loss, get_classification_loss, get_regression_loss
from preprocess import get_localization_data
from regr import Regr
from roi_utility import rpn_to_roi
from rpn import Rpn
from tracking import save_data

def get_img(img_path):
    img = read_file(img_path)
    img = decode_jpeg(img, channels=3)
    img = resize(img, [image_size, image_size])
    img = 1 - img/255. # We would rather have the whole white void area be full of zeros than ones
    return img

def get_labels_boxes(boxes, target):
    labels_boxes = []
    for b in boxes:
        x1 = b[0]
        y1 = b[1]
        x2 = b[2]
        y2 = b[3]
        t = np.reshape(target[y1:y2, x1:x2], (-1))
        t = np.delete(t, np.where(t == 0))
        if (len(t) == 0):
            labels_boxes.append(0)
        else:
            (classes, occurences) = np.unique(t, return_counts=True)
            k = np.argmax(occurences)
            labels_boxes.append(classes[k])
    return labels_boxes

def get_boxes_precision(boxes, regression_values, target):
    precision = []
    for i in range(len(boxes)):
        b = boxes[i]
        regr = regression_values[i]

        box_center_x = (b[2] + b[0]) / 2
        box_center_y = (b[3] + b[1]) / 2
        box_w = b[2] - b[0]
        box_h = b[3] - b[1]

        final_box_center_x = box_center_x + regr[0]
        final_box_center_y = box_center_y + regr[1]
        final_box_w = box_w + regr[2] # This is not the right correction for proper faster-rcnn
        final_box_h = box_h + regr[3] # But it is better suited to our case

        x1 = int(round(final_box_center_x - final_box_w / 2))
        x2 = int(round(final_box_center_x + final_box_w / 2))
        y1 = int(round(final_box_center_y - final_box_h / 2))
        y2 = int(round(final_box_center_y + final_box_h / 2))

        x1 = max(x1, 0)
        x2 = min(x2, feature_size - 1)
        y1 = max(y1, 0)
        y2 = min(y2, feature_size - 1)

        t = np.reshape(target[y1:y2, x1:x2], (-1))
        total_area = len(t)
        t = np.delete(t, np.where(t == 0))
        non_zero_area = len(t)
        precision.append([non_zero_area, total_area])
    return precision

def train():
    feature_mapper = FeatureMapper()
    rpn = Rpn()
    classifier = Classifier()
    regr = Regr()

    feature_mapper.load_weights("./weights/feature_mapper")
    rpn.load_weights("./weights/rpn")
    classifier.load_weights("./weights/classifier")
    regr.load_weights("./weights/regr")

    opt = AdamOptimizer(1e-4)
    with open("../data/data_classification_train.json") as json_file:
        data = json.load(json_file)
    data_index = 0
    while str(data_index) in data:
        raw_data = data[str(data_index)]
        target, bounding_box_target = get_localization_data(raw_data)
        img = get_img("../pictures/pictures_classification_train/{}.png".format(data_index))
        img = tf.convert_to_tensor([img])

        def get_loss():
            features = feature_mapper(img)
            rpn_class = rpn(features)
            boxes, probs = rpn_to_roi(rpn_class)

            classification_logits = classifier(features, boxes)
            regression_values = regr(features, boxes)

            labels_boxes = get_labels_boxes(boxes, target)

            localization_loss = get_localization_loss(rpn_class, target)
            regression_loss = get_regression_loss(regression_values, boxes, bounding_box_target, probs)
            classification_loss = get_classification_loss(classification_logits, labels_boxes, probs)

            no_regr_boxes_precision = get_boxes_precision(boxes, np.zeros(regression_values.shape), target)
            final_boxes_precision = get_boxes_precision(boxes, regression_values.numpy(), target)
            save_data(data_index, raw_data, boxes.tolist(), [a.numpy().tolist() for a in classification_logits], labels_boxes, no_regr_boxes_precision, final_boxes_precision)

            return localization_loss + classification_loss + regression_loss

        opt.minimize(get_loss)

        data_index += 1
        if (data_index % 100 == 99):
            feature_mapper.save_weights("./weights/feature_mapper")
            rpn.save_weights("./weights/rpn")
            classifier.save_weights("./weights/classifier")
            regr.save_weights("./weights/regr")

train()
