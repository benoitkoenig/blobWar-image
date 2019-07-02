import numpy as np
import sys
import tensorflow as tf
from tensorflow.image import decode_jpeg, resize
from tensorflow.io import read_file
from tensorflow.losses import sparse_softmax_cross_entropy
from tensorflow.nn import softmax
from tensorflow.train import AdamOptimizer

from constants import image_size
from preprocess import get_classification_data
from resnet import ResNet

tf.enable_eager_execution()

resnet = ResNet()

def get_img(img_path):
    img = read_file(img_path)
    img = decode_jpeg(img, channels=3)
    img = resize(img, [image_size, image_size])
    img = img/255.0
    return img

def train():
    print("Training started")
    opt = AdamOptimizer(1e-3)
    images_data = get_classification_data("data/data_classification_train.json")
    for (i, label) in images_data:
        print("{} of {}".format(i, len(images_data)))
        img = get_img("./pictures/pictures_classification_train/{}.png".format(i))
        def get_loss():
            logits = resnet(tf.convert_to_tensor([img]))
            return sparse_softmax_cross_entropy([label], logits)
        opt.minimize(get_loss)
    resnet.save_weights("./weights/resnet")

def evaluate():
    resnet.load_weights("./weights/resnet")
    images_data = get_classification_data("data/data_classification_evaluate.json")
    for (i, label) in images_data:
        img = get_img("./pictures/pictures_classification_evaluate/{}.png".format(i))
        logits = resnet(tf.convert_to_tensor([img]))
        print(label, logits.numpy())

training = False
evaluating = False
for instruction in sys.argv:
    if (instruction == "train"):
        training = True
    if (instruction == "evaluate"):
        evaluating = True

if training:
    train()
if evaluating:
    evaluate()
if (training == False) & (evaluating == False):
    print("Usage: 'python main_classification.py train' or 'python main_classification.py evaluate'")
