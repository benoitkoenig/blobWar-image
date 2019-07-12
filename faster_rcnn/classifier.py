import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

from constants import nb_class
from roi_pooling import RoiPooling

class Classifier(Model):
    def __init__(self):
        super(Classifier, self).__init__()
        self.rp1 = RoiPooling()
        self.conv1 = Conv2D(24, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same")
        self.conv2 = Conv2D(24, kernel_size=(3, 3), activation="relu", padding="same")
        self.flat1 = Flatten()
        self.dense1 = Dense(64, activation="relu")
        self.dense_logits = Dense(nb_class, activation="linear")

    def call(self, feature_map, boxes):
        feature_areas = self.rp1(feature_map, boxes)
        output = []
        for fa in feature_areas:
            x = self.conv1(fa)
            x = self.conv2(x)
            x = self.flat1(x)
            x = self.dense1(x)
            logits = self.dense_logits(x)
            output.append(logits)
        output = tf.reshape(output, [len(boxes), nb_class])
        return output
