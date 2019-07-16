import json
import numpy as np
import tensorflow as tf
from tensorflow.image import decode_jpeg, resize
from tensorflow.io import read_file
from tensorflow.nn import sigmoid_cross_entropy_with_logits, sparse_softmax_cross_entropy_with_logits
from tensorflow.train import AdamOptimizer

tf.enable_eager_execution()

from boxes import get_boxes, get_boxes_precision, get_labels_boxes
from classifier import Classifier
from constants import image_size
from feature_mapper import FeatureMapper
from losses import get_localization_loss, get_classification_loss, get_regression_loss
from preprocess import get_localization_data
from regr import Regr
from rpn import Rpn
from tracking import save_data

def get_img(img_path):
    img = read_file(img_path)
    img = decode_jpeg(img, channels=3)
    img = resize(img, [image_size, image_size])
    img = 1 - img/255. # We would rather have the whole white void area be full of zeros than ones
    img = tf.convert_to_tensor([img])
    return img

def train():
    feature_mapper = FeatureMapper()
    rpn = Rpn()
    classifier = Classifier()
    regr = Regr()

    feature_mapper.load_weights("./weights/feature_mapper")
    rpn.load_weights("./weights/rpn")
    classifier.load_weights("./weights/classifier")
    regr.load_weights("./weights/regr")

    opt = AdamOptimizer(5e-5)
    with open("../data/data_detect_local_train.json") as json_file:
        data = json.load(json_file)
    data_index = 0
    while str(data_index) in data:
        raw_data = data[str(data_index)]
        target, bounding_box_target = get_localization_data(raw_data)
        img = get_img("../pictures/pictures_detect_local_train/{}.png".format(data_index))

        def get_loss():
            features = feature_mapper(img)
            rpn_map = rpn(features)
            boxes, probs = get_boxes(rpn_map)

            classification_logits = classifier(features, boxes)
            regression_values = regr(features, boxes)

            labels_boxes = get_labels_boxes(boxes, target)

            localization_loss = get_localization_loss(rpn_map, target)
            regression_loss = get_regression_loss(regression_values, boxes, bounding_box_target, probs)
            classification_loss = get_classification_loss(classification_logits, labels_boxes, probs)

            no_regr_boxes_precision = get_boxes_precision(boxes, np.zeros(regression_values.shape), target)
            final_boxes_precision = get_boxes_precision(boxes, regression_values.numpy(), target)
            save_data(data_index, raw_data, boxes.tolist(), [a.numpy().tolist() for a in classification_logits], labels_boxes, no_regr_boxes_precision, final_boxes_precision, probs.tolist())

            return localization_loss + classification_loss + regression_loss

        opt.minimize(get_loss)

        data_index += 1
        if (data_index % 100 == 99):
            feature_mapper.save_weights("./weights/feature_mapper")
            rpn.save_weights("./weights/rpn")
            classifier.save_weights("./weights/classifier")
            regr.save_weights("./weights/regr")

train()
