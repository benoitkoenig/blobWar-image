import numpy as np
import tensorflow as tf

from boxes import get_boxes, get_final_box
from constants import real_image_height, real_image_width, feature_size
from models.classifier import Classifier
from models.feature_mapper import FeatureMapper
from models.regr import Regr
from models.roi_pooling import RoiPooling
from models.rpn import Rpn
from models.segmentation import Segmentation

feature_mapper = FeatureMapper()
rpn = Rpn()
roi_pooling = RoiPooling()
classifier = Classifier()
regr = Regr()
segmentation = Segmentation()

feature_mapper.load_weights("./weights/feature_mapper")
rpn.load_weights("./weights/rpn")
classifier.load_weights("./weights/classifier")
regr.load_weights("./weights/regr")
segmentation.load_weights("./weights/segmentation")

def get_prediction(img):
    features = feature_mapper(img)
    rpn_map = rpn(features)

    boxes, probs = get_boxes(rpn_map)
    feature_areas = roi_pooling(features, boxes)

    classification_logits = classifier(feature_areas)
    regression_values = regr(feature_areas)

    return boxes, probs, classification_logits.numpy(), regression_values.numpy()

def get_prediction_mask(img):
    features = feature_mapper(img)
    rpn_map = rpn(features)

    boxes, probs = get_boxes(rpn_map)
    feature_areas = roi_pooling(features, boxes)

    regression_values = regr(feature_areas)

    regr_boxes = [get_final_box(boxes[i], regression_values[i].numpy()) for i in range(len(boxes)) if probs[i] > .9]
    regr_feature_areas = roi_pooling(features, regr_boxes)

    predicted_mask = segmentation(regr_feature_areas)

    final_mask = np.zeros([real_image_height, real_image_width])
    for (i, predicted_mask) in enumerate(predicted_mask):
        b = regr_boxes[i]
        x1 = b[0] * real_image_width // feature_size
        y1 = b[1] * real_image_height // feature_size
        x2 = b[2] * real_image_width // feature_size
        y2 = b[3] * real_image_height // feature_size
        predicted_mask = tf.image.resize(predicted_mask, [y2 - y1, x2 - x1]).numpy()
        for y in range(y2 - y1):
            for x in range(x2 - x1):
                if (predicted_mask[y][x][0] > 0):
                    final_mask[y1 + y][x1 + x] = (i + 1)

    return final_mask
