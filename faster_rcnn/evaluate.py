import json
import math
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from boxes import get_boxes, get_final_box
from classifier import Classifier
from constants import feature_size
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

def get_prediction(img):
    features = feature_mapper(img)
    rpn_map = rpn(features)

    boxes, probs = get_boxes(rpn_map)
    feature_areas = roi_pooling(features, boxes)

    classification_logits = classifier(feature_areas)
    regression_values = regr(feature_areas)

    return boxes, probs, classification_logits.numpy(), regression_values.numpy()

statuses = {
    "normal": 0,
    "hat": 1,
    "ghost": 2,
}

def evaluate():
    with open("../data/data_detect_local_evaluate_10000.json") as json_file:
        data = json.load(json_file)
    data_index = 0

    total_match = []
    total_missing_blobs = 0
    total_extra_blobs = 0

    while str(data_index) in data:
        img = get_img("../pictures/pictures_detect_local_evaluate_10000/{}.png".format(data_index))
        boxes, probs, classification_logits, regression_values = get_prediction(img)

        raw_data = data[str(data_index)]
        all_blobs = raw_data["army"] + raw_data["enemy"]

        estimates = []
        for i in range(len(boxes)):
            if probs[i] > .9:
                box = get_final_box(boxes[i], regression_values[i])
                classification_probs = tf.nn.softmax(classification_logits[i]).numpy()
                match = []
                for i, t_blob in enumerate(all_blobs):
                    c_x = (box[0] + box[2]) / (2 * feature_size)
                    c_y = (box[1] + box[3]) / (2 * feature_size)
                    t_x = t_blob["x"] # TODO: I do not not how to include the position between 31 and 32
                    t_y = t_blob["y"] # I think the all code misses +1 on the upper limit
                    m = int(t_blob["alive"])
                    distance = math.sqrt((c_x - t_x) ** 2 + (c_y - t_y) ** 2)
                    m = m / (1 + distance)
                    label = 1 + statuses[t_blob["status"]] + 3 * int(i >= 3)
                    m = m * classification_probs[label]
                    match.append(m)
                estimates.append(match)

        output = []
        while(len(estimates) != 0):
            best_match = np.argmax(estimates)
            predicted_blob = best_match // 6
            target_blob = best_match % 6
            if (estimates[predicted_blob][target_blob] == 0):
                break
            output.append(estimates[predicted_blob][target_blob])
            estimates.pop(predicted_blob)
            for match in estimates:
                match[target_blob] = 0

        alive_blobs = len([1 for b in all_blobs if b["alive"] == True])
        blobs_count_difference = alive_blobs - len(output)

        picture_match = sum(output) / len(output)

        print("\n>>>>>> {}".format(data_index))
        print("{} blobs matching".format(len(output)))
        print("{}% average match".format(int(picture_match * 100)))
        print("Difference nb blobs alive - nb blobs predicted: {}".format(blobs_count_difference))

        total_match.append(picture_match)
        total_missing_blobs += max(0, blobs_count_difference)
        total_extra_blobs += max(0, -blobs_count_difference)

        data_index += 1

    print("\n\n>>>>>> Summary for {} pictures".format(data_index))
    print("Average match: {}%".format(int(sum(total_match) / len(total_match)) * 100))
    print("Total missed blobs: {}".format(total_missing_blobs))
    print("Total extra predictions: {}".format(total_extra_blobs))

evaluate()
