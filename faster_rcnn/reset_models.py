import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from classifier import Classifier
from constants import image_size, feature_size, anchor_size
from feature_mapper import FeatureMapper
from rpn import Rpn

def reset_models():
    feature_mapper = FeatureMapper()
    rpn = Rpn()
    classifier = Classifier()

    random_image = tf.convert_to_tensor(np.random.random((1, image_size, image_size, 3)), dtype=np.float32)
    random_features = tf.convert_to_tensor(np.random.random((1, feature_size, feature_size, 8)), dtype=np.float32)
    random_boxes = [[2, 2, 4, 4]]

    _ = feature_mapper(random_image)
    (_, _) = rpn(random_features)
    _ = classifier(random_features, random_boxes)

    feature_mapper.save_weights("./weights/feature_mapper")
    rpn.save_weights("./weights/rpn")
    classifier.save_weights("./weights/classifier")

reset_models()
