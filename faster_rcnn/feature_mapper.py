from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D

class FeatureMapper(Model):
    def __init__(self):
        super(FeatureMapper, self).__init__()
        self.conv1 = Conv2D(8, kernel_size=(5, 5), activation="relu", padding="same")
        self.conv2 = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same")
        self.conv3 = Conv2D(8, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same")
        self.mp1 = MaxPool2D((2, 2))

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        output = self.mp1(x)
        return output
