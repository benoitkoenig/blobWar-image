import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from constants import mask_anchor_size

class Segmentation(Model):
    def __init__(self):
        super(Segmentation, self).__init__()
        self.conv1 = Conv2D(16, kernel_size=(1, 1), activation="relu", padding="same")
        self.flat1 = Flatten()
        self.dense1 = Dense(512, activation="linear") # Tutorial explicitely says not to ReLU
        self.dense2 = Dense(mask_anchor_size ** 2, activation="linear")

    def call(self, feature_areas):
        output = []
        for (i, fa) in enumerate(feature_areas):
            x = self.conv1(fa)
            x = self.flat1(x)
            x = self.dense1(x)
            x = self.dense2(x)
            x = tf.reshape(x, (mask_anchor_size, mask_anchor_size, 1))
            output.append(x)
        return output
