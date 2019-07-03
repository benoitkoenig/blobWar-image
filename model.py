from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

from constants import nb_class

class BlobClassifierModel(Model):
    def __init__(self):
        super(BlobClassifierModel, self).__init__()
        self.conv1 = Conv2D(4, kernel_size=(3, 3), strides=(2, 2), activation="relu")
        self.conv2 = Conv2D(4, kernel_size=(3, 3), strides=(2, 2), activation="relu")
        self.conv3 = Conv2D(4, kernel_size=(3, 3), strides=(2, 2), activation="relu")
        self.flat1 = Flatten()
        self.dense1 = Dense(64, activation="relu")
        self.dense_logits = Dense(nb_class, activation="linear")

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat1(x)
        x = self.dense1(x)
        logits = self.dense_logits(x)
        return logits
