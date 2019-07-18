from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D

from constants import nb_class

class Yolo(Model):
    def __init__(self):
        super(Yolo, self).__init__()
        self.conv1 = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), activation="relu", padding="same")
        self.conv2 = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), activation="relu", padding="same")
        self.conv3 = Conv2D(64, kernel_size=(5, 5), activation="relu", padding="same")
        self.conv_logits = Conv2D(nb_class + 5, kernel_size=(1, 1), activation="linear", padding="same")

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        output = self.conv_logits(x)
        return output
