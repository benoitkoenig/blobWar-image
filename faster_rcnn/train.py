import numpy as np
import tensorflow as tf
from tensorflow.image import decode_jpeg, resize
from tensorflow.io import read_file
from tensorflow.nn import sigmoid_cross_entropy_with_logits, sparse_softmax_cross_entropy_with_logits
from tensorflow.train import AdamOptimizer

tf.enable_eager_execution()

from classifier import Classifier
from constants import image_size, nb_class
from feature_mapper import FeatureMapper
from preprocess import get_localisation_data, get_classification_data
from roi_utility import rpn_to_roi, roi_pooling
from rpn import Rpn

def get_img(img_path):
    img = read_file(img_path)
    img = decode_jpeg(img, channels=3)
    img = resize(img, [image_size, image_size])
    img = img/255.0
    return img

def train():
    targets = get_localisation_data("../data/data_classification_train.json")
    labels = get_classification_data("../data/data_classification_train.json")
    feature_mapper = FeatureMapper()
    rpn = Rpn()
    classifier = Classifier()
    opt = AdamOptimizer(1e-4)
    count = 0
    for (i, target) in targets:
        label = -1
        for (j, l) in labels:
            if (i == j):
                label = l
                break
        img = get_img("../pictures/pictures_classification_train/{}.png".format(i))
        img = tf.convert_to_tensor([img])

        def get_loss():
            features = feature_mapper(img)
            (rpn_class, _) = rpn(features)
            boxes, probs = rpn_to_roi(rpn_class, _)
            feature_areas = roi_pooling(features, boxes)
            classification_logits = []
            for fa in feature_areas:
                logits = classifier(fa)
                classification_logits.append(tf.reshape(logits, [nb_class]))

            localization_logits = tf.reshape(rpn_class, [-1])
            localization_labels = tf.reshape(tf.convert_to_tensor([target], dtype=np.float32), [-1])
            localization_loss = sigmoid_cross_entropy_with_logits(labels=localization_labels, logits=localization_logits)
            localization_loss = tf.reduce_mean(localization_loss)

            classification_loss = sparse_softmax_cross_entropy_with_logits(logits=classification_logits, labels=([label] * len(classification_logits)))
            classification_loss = tf.reduce_mean(classification_loss)

            if (count % 100 == 99):
                feature_mapper.save_weights("./weights/feature_mapper")
                rpn.save_weights("./weights/rpn")
                classifier.save_weights("./weights/classifier")
                print("\nWeights saved")
                print(i)
                print(boxes)
                print(probs)
                cl = []
                for c in classification_logits:
                    cl.append(c.numpy().tolist())
                print(cl)

            return localization_loss + classification_loss
        opt.minimize(get_loss)
        count += 1

train()
