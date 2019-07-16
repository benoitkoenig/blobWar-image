import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from constants import image_size
from feature_mapper import FeatureMapper

def reset_model():
    feature_mapper = FeatureMapper()
    random_image = tf.convert_to_tensor(np.random.random((1, image_size, image_size, 3)), dtype=np.float32)
    _ = feature_mapper(random_image)
    feature_mapper.save_weights("./weights/feature_mapper")

reset_model()
