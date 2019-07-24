from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D

class FeatureMapper(Model):
    def __init__(self):
        super(FeatureMapper, self).__init__()
        self.conv1 = Conv2D(16, kernel_size=(5, 5), strides=(2, 2), activation="relu", padding="same")
        self.conv2 = Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same")
        self.conv3 = Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same")

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        output = self.conv3(x)
        return output
