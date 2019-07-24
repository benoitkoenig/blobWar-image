import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from constants import image_height, image_width

class Segmentation(Model):
    def __init__(self):
        super(Segmentation, self).__init__()
        self.conv1 = Conv2D(16, kernel_size=(1, 1), activation="relu", padding="same")
        self.flat1 = Flatten()
        self.dense1 = Dense(512, activation="linear") # Tutorial explicitely says not to ReLU
        self.dense2 = Dense(64 ** 2, activation="linear")

    def call(self, input):
        x = self.conv1(input)
        x = self.flat1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.reshape(x, (64, 64, 1))
        output = tf.image.resize(x, (image_height, image_width))
        return output
