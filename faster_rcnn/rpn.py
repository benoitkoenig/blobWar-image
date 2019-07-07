from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D

from constants import num_anchors

class Rpn(Model):
    def __init__(self):
        super(Rpn, self).__init__()
        self.conv1 = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="normal")
        self.conv_class = Conv2D(num_anchors, (1, 1), activation="linear", kernel_initializer="uniform")
        # self.conv_regr = Conv2D(num_anchors * 4, (1, 1), activation="linear", kernel_initializer="zero")

    def call(self, input):
        x = self.conv1(input)
        x_class = self.conv_class(x)
        # x_regr = self.conv_regr(x) # not yet implemented
        # return (x_class, x_regr)
        return (x_class, None)
