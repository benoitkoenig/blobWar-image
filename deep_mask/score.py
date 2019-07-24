import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D

class Score(Model):
    def __init__(self):
        super(Score, self).__init__()
        self.mp1 = MaxPool2D((2, 2))
        self.flat1 = Flatten()
        self.conv1 = Conv2D(16, kernel_size=(1, 1), activation="relu", padding="same")
        self.dense1 = Dense(64, activation="relu")
        self.dense2 = Dense(128, activation="relu")
        self.dense3 = Dense(1, activation="linear")

    def call(self, input):
        x = self.mp1(input)
        x = self.flat1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.dense3(x)
        return output
