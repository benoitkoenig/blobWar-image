import json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from feature_mapper import FeatureMapper
from img import get_img
from loss import calculate_loss
from preprocess import get_true_mask
from score import Score
from segmentation import Segmentation

def show_evaluation(segmentation, true_masks, id):
    seg = tf.math.sigmoid(segmentation).numpy()
    (pixel_mask, _) = true_masks[0]
    output = [[], []]
    for y in range(594):
        for x in range(742):
            if (pixel_mask[y][x][0] == 1):
                output[1].append(seg[y][x][0])
            else:
                output[0].append(seg[y][x][0])
    avg_zeros = sum(output[0]) / len(output[0])
    avg_ones = sum(output[1]) / len(output[1])
    print("{}: Avg of ones: {} - Avg of zeros: {}".format(id, avg_ones, avg_zeros))

def train():
    feature_mapper = FeatureMapper()
    score = Score()
    segmentation = Segmentation()

    feature_mapper.load_weights("./immutable_weights/feature_mapper")
    # score.load_weights("./weights/score")
    # segmentation.load_weights("./weights/segmentation")

    opt = Adam(learning_rate=5e-5)
    with open("../data/data_classification_train.json") as json_file:
        data = json.load(json_file)
    data_index = 0
    while str(data_index) in data:
        img = get_img("../pictures/pictures_classification_train/{}.png".format(data_index))
        true_masks = get_true_mask(data[str(data_index)])

        features = feature_mapper(img)
        def get_loss():
            segmentation_prediction = segmentation(features)
            score_prediction = score(features)
            show_evaluation(segmentation_prediction, true_masks, data_index)
            return calculate_loss(segmentation_prediction, score_prediction, true_masks)

        opt.minimize(get_loss, [score.trainable_weights, segmentation.trainable_weights])


        if (data_index % 100 == 99):
            score.save_weights("./weights/score")
            segmentation.save_weights("./weights/segmentation")
        data_index += 1

train()
