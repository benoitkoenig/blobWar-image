import json
import numpy as np
import tensorflow as tf
from tensorflow.image import decode_jpeg, resize
from tensorflow.io import read_file
from tensorflow.nn import sigmoid_cross_entropy_with_logits, sparse_softmax_cross_entropy_with_logits
from tensorflow.train import AdamOptimizer

tf.enable_eager_execution()
np.set_printoptions(threshold=10000)

from constants import image_size, nb_class, feature_size
from feature_mapper import FeatureMapper
from preprocess import get_localization_data

def get_img(img_path):
    img = read_file(img_path)
    img = decode_jpeg(img, channels=3)
    img = resize(img, [image_size, image_size])
    img = 1 - img/255.0
    return img

def train():
    feature_mapper = FeatureMapper()

    feature_mapper.load_weights("./weights/feature_mapper")

    opt = AdamOptimizer(1e-4)
    with open("../data/data_classification_train.json") as json_file:
        data = json.load(json_file)
    data_index = 0
    while str(data_index) in data:
        img = get_img("../pictures/pictures_classification_train/{}.png".format(data_index))
        img = tf.convert_to_tensor([img])
        labels = get_localization_data(data[str(data_index)])

        def get_loss():
            logits = feature_mapper(img)
            if (data_index % 100 == 99):
                for i in range(feature_size):
                    for j in range(feature_size):
                        label = labels[0, j, i]
                        if (label != 0):
                            specific_logits = tf.gather(tf.gather(tf.gather(logits, 0), j), i)
                            probs = tf.nn.softmax(specific_logits)
                            print(label, probs.numpy())

            loss = sparse_softmax_cross_entropy_with_logits(logits=[logits], labels=[labels])
            loss = tf.reduce_mean(loss)
            return loss

        opt.minimize(get_loss)
        if (data_index % 100 == 99):
            print(data_index)
            feature_mapper.save_weights("./weights/feature_mapper")
        data_index += 1

train()
