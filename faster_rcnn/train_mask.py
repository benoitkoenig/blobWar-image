import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from boxes import get_boxes, get_final_box, get_box_true_mask
from img import get_img
from losses import get_segmentation_loss
from models.feature_mapper import FeatureMapper
from models.regr import Regr
from models.roi_pooling import RoiPooling
from models.rpn import Rpn
from models.segmentation import Segmentation
from prediction import get_prediction
from preprocess import get_true_mask

def train():
    feature_mapper = FeatureMapper()
    rpn = Rpn()
    roi_pooling = RoiPooling()
    regr = Regr()
    segmentation = Segmentation()

    feature_mapper.load_weights("./weights/feature_mapper")
    rpn.load_weights("./weights/rpn")
    regr.load_weights("./weights/regr")
    segmentation.load_weights("./weights/segmentation")

    opt = Adam(learning_rate=5e-5)
    with open("../data/data_detect_local_evaluate_10000.json") as json_file:
        data = json.load(json_file)
    data_index = 0
    while str(data_index) in data:
        raw_data = data[str(data_index)]
        true_mask = get_true_mask(raw_data)
        img = get_img("../pictures/pictures_detect_local_evaluate_10000/{}.png".format(data_index))

        features = feature_mapper(img)
        rpn_map = rpn(features)
        boxes, probs = get_boxes(rpn_map)
        feature_areas = roi_pooling(features, boxes)
        regression_values = regr(feature_areas)
        regr_boxes = [get_final_box(boxes[i], regression_values[i].numpy()) for i in range(len(boxes)) if probs[i] > .9]
        if len(regr_boxes) > 0:
            regr_feature_areas = roi_pooling(features, regr_boxes)
            box_true_masks = get_box_true_mask(regr_boxes, true_mask)

            def get_loss():
                predicted_masks = segmentation(regr_feature_areas)
                return get_segmentation_loss(predicted_masks, box_true_masks)

            opt.minimize(get_loss, [segmentation.trainable_weights])

        data_index += 1
        if (data_index % 100 == 99):
            print("{} - Weights saved".format(data_index))
            segmentation.save_weights("./weights/segmentation")

train()
