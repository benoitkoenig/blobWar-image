import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from boxes import get_boxes, get_boxes_precision, get_labels_boxes
from img import get_img
from losses import get_localization_loss, get_classification_loss, get_regression_loss
from models.classifier import Classifier
from models.feature_mapper import FeatureMapper
from models.regr import Regr
from models.roi_pooling import RoiPooling
from models.rpn import Rpn
from preprocess import get_localization_data
from tracking import save_data

def train():
    feature_mapper = FeatureMapper()
    rpn = Rpn()
    roi_pooling = RoiPooling()
    classifier = Classifier()
    regr = Regr()

    feature_mapper.load_weights("./weights/feature_mapper")
    rpn.load_weights("./weights/rpn")
    classifier.load_weights("./weights/classifier")
    regr.load_weights("./weights/regr")

    opt = Adam(learning_rate=5e-5)
    with open("../data/data_detect_local_evaluate_10000.json") as json_file:
        data = json.load(json_file)
    data_index = 0
    while str(data_index) in data:
        raw_data = data[str(data_index)]
        target, bounding_box_target = get_localization_data(raw_data)
        img = get_img("../pictures/pictures_detect_local_evaluate_10000/{}.png".format(data_index))

        def get_loss():
            features = feature_mapper(img)
            rpn_map = rpn(features)

            boxes, probs = get_boxes(rpn_map)
            feature_areas = roi_pooling(features, boxes)

            classification_logits = classifier(feature_areas)
            regression_values = regr(feature_areas)

            labels_boxes = get_labels_boxes(boxes, target)

            localization_loss = get_localization_loss(rpn_map, target)
            regression_loss = get_regression_loss(regression_values, boxes, bounding_box_target, probs)
            classification_loss = get_classification_loss(classification_logits, labels_boxes, probs)

            no_regr_boxes_precision = get_boxes_precision(boxes, np.zeros(regression_values.shape), target)
            final_boxes_precision = get_boxes_precision(boxes, regression_values.numpy(), target)
            save_data(data_index, raw_data, boxes.tolist(), [a.numpy().tolist() for a in classification_logits], labels_boxes, no_regr_boxes_precision, final_boxes_precision, probs.tolist())

            return localization_loss + classification_loss + regression_loss

        opt.minimize(
            get_loss,
            [feature_mapper.trainable_weights, rpn.trainable_weights, classifier.trainable_weights, regr.trainable_weights],
        )

        data_index += 1
        if (data_index % 100 == 99):
            feature_mapper.save_weights("./weights/feature_mapper")
            rpn.save_weights("./weights/rpn")
            classifier.save_weights("./weights/classifier")
            regr.save_weights("./weights/regr")

train()
