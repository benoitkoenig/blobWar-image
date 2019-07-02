import numpy as np

import tensorflow as tf
from tensorflow.image import decode_jpeg, resize
from tensorflow.io import read_file
from tensorflow.losses import sparse_softmax_cross_entropy
from tensorflow.train import AdamOptimizer

from preprocess_classification import images_data
from resnet import ResNet

tf.enable_eager_execution()

image_size = 256

opt = AdamOptimizer(1e-3)
resnet = ResNet()
# resnet(tf.convert_to_tensor(np.random.random((1, 256, 256, 3))))

def get_img(img_path):
    img = read_file(img_path)
    img = decode_jpeg(img, channels=3)
    img = resize(img, [image_size, image_size])
    img = img/255.0
    return img

for (i, label) in images_data:
    img = get_img("./pictures/{}.png".format(i))
    def get_loss():
        logits = resnet(tf.convert_to_tensor([img]))
        return sparse_softmax_cross_entropy([label], logits)
    opt.minimize(get_loss)
