import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

class Regr(Model):
    def __init__(self):
        super(Regr, self).__init__()
        self.conv1 = Conv2D(6, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding="same")
        self.flat1 = Flatten()
        self.dense1 = Dense(32, activation="relu")
        self.dense_regr = Dense(4, activation="linear")

    def call(self, feature_areas):
        output = []
        for fa in feature_areas:
            x = self.conv1(fa)
            x = self.flat1(x)
            x = self.dense1(x)
            logits = self.dense_regr(x)
            output.append(tf.gather(logits, 0))
        output = tf.convert_to_tensor(output)
        return output
