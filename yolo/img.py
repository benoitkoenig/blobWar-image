import tensorflow as tf
from tensorflow.io import read_file
from tensorflow.image import decode_jpeg, resize

from constants import image_size

def get_img(img_path):
    img = read_file(img_path)
    img = decode_jpeg(img, channels=3)
    img = resize(img, [image_size, image_size])
    img = 1 - img/255. # We would rather have the whole white void area be full of zeros than ones
    img = tf.convert_to_tensor([img])
    return img
