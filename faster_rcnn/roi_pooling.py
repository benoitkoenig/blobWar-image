import tensorflow as tf
from tensorflow.keras.layers import Layer

from constants import anchor_size, feature_size

class RoiPooling(Layer):
    def call(self, features_map, boxes):
        output = []
        for box in boxes:
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            specific_features = tf.slice(features_map, [0, y1, x1, 0], [-1, y2-y1, x2-x1, -1])
            specific_features = tf.image.resize(specific_features, (anchor_size, anchor_size))
            output.append(specific_features)
        return output
