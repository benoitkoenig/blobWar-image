import numpy as np
import tensorflow as tf
from tensorflow.image import decode_jpeg, resize
from tensorflow.io import read_file
from tensorflow.nn import sigmoid_cross_entropy_with_logits
from tensorflow.train import AdamOptimizer

tf.enable_eager_execution()

from constants import image_size
from feature_mapper import FeatureMapper
from preprocess import get_localisation_data
from roi import rpn_to_roi
from rpn import Rpn

def get_img(img_path):
    img = read_file(img_path)
    img = decode_jpeg(img, channels=3)
    img = resize(img, [image_size, image_size])
    img = img/255.0
    return img

def train():
    targets = get_localisation_data("../data/data_detect_local_train.json")
    feature_mapper = FeatureMapper()
    rpn = Rpn()
    opt = AdamOptimizer(1e-4)
    count = 0
    for (i, target) in targets:
        img = get_img("../pictures/pictures_detect_local/{}.png".format(i))
        img = tf.convert_to_tensor([img])

        def get_loss():
            features = feature_mapper(img)
            (rpn_class, _) = rpn(features)
            logits = tf.reshape(rpn_class, [-1])
            labels = tf.reshape(tf.convert_to_tensor([target], dtype=np.float32), [-1])
            loss = sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_mean(loss)
            if (count % 100 == 0):
                boxes, probs = rpn_to_roi(rpn_class, _)
                print("\n")
                print("Image id: {}".format(i))
                print("Boxes:")
                print(boxes)
                print("Probs:")
                print(probs)
            return loss
        opt.minimize(get_loss)
        count += 1

train()
