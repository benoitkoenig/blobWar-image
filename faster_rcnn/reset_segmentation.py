import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from constants import anchor_size
from models.segmentation import Segmentation

def reset_segmentation():
    segmentation = Segmentation()

    random_feature_areas = [tf.convert_to_tensor(np.random.random((1, anchor_size, anchor_size, 12)), dtype=np.float32)]

    _ = segmentation(random_feature_areas)

    segmentation.save_weights("./weights/segmentation")

reset_segmentation()
