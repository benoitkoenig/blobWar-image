from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Dense, Add, MaxPooling2D, ZeroPadding2D, AveragePooling2D, Flatten, InputLayer

nb_class = 6

class Residual(Model):
    def __init__(self, K, red, input_shape):
        super(Residual, self).__init__()
        self.red = red
        strides = (1, 1)
        self.input1 = InputLayer(input_shape)
        if (red == True):
            strides = (2, 2)
            self.conv_shortcut = Conv2D(K, (1, 1), strides=strides)
        self.bn1 = BatchNormalization()
        self.act1 = Activation("relu")
        self.conv1 = Conv2D(K // 4, (1, 1))
        self.bn2 = BatchNormalization()
        self.act2 = Activation("relu")
        self.conv2 = Conv2D(K // 4, (3, 3), strides=strides, padding="same")
        self.bn3 = BatchNormalization()
        self.act3 = Activation("relu")
        self.conv3 = Conv2D(K, (1, 1))
        self.add1 = Add()

    def call(self, input):
        x = self.input1(input)
        if (self.red == False):
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)
        x = self.bn1(input)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv3(x)
        output = self.add1([x, shortcut])
        return output

class ResNet(Model):
    def __init__(self):
        super(ResNet, self).__init__()
        self.input1 = InputLayer((256, 256, 3))
        self.bn1 = BatchNormalization()
        self.conv1 = Conv2D(64, (5, 5), padding="same")
        self.bn2 = BatchNormalization()
        self.act1 = Activation("relu")
        self.zp1 = ZeroPadding2D((1, 1))
        self.mp1 = MaxPooling2D((3, 3), strides=(2, 2))
        self.res1 = Residual(64, True, (256, 256, 3))
        self.res2 = Residual(64, False, (128, 128, 3))
        self.res3 = Residual(64, False, (128, 128, 3))
        self.res4 = Residual(32, True, (128, 128, 3))
        self.res5 = Residual(32, False, (64, 64, 3))
        self.res6 = Residual(32, False, (64, 64, 3))
        self.res7 = Residual(16, True, (64, 64, 3))
        self.res8 = Residual(16, False, (32, 32, 3))
        self.res9 = Residual(16, False, (32, 32, 3))
        self.bn3 = BatchNormalization()
        self.act2 = Activation("relu")
        self.ap1 = AveragePooling2D((8, 8))
        self.flat1 = Flatten()
        self.dense1 = Dense(nb_class)
        self.act3 = Activation("linear")

    def call(self, input):
        x = self.input1(input)
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.zp1(x)
        x = self.mp1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.ap1(x)
        x = self.flat1(x)
        x = self.dense1(x)
        logits = self.act3(x)
        return logits
