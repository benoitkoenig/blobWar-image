import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution() # Remove when switching to tf2

from constants import image_size
from yolo import Yolo

def reset_model():
    yolo = Yolo()

    random_image = tf.convert_to_tensor(np.random.random((1, image_size, image_size, 3)), dtype=np.float32)

    _ = yolo(random_image)

    yolo.save_weights("./weights/yolo")

reset_model()
