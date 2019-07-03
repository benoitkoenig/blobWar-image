import numpy as np
from random import shuffle
import sys
import tensorflow as tf
from tensorflow.image import decode_jpeg, resize
from tensorflow.io import read_file
from tensorflow.nn import softmax, sparse_softmax_cross_entropy_with_logits
from tensorflow.train import AdamOptimizer

from constants import image_size
from preprocess import get_classification_data
from resnet import ResNet
from tracking import save_data

tf.enable_eager_execution()

def get_resnet():
    resnet = ResNet()
    random_image = tf.convert_to_tensor(np.random.random((1, image_size, image_size, 3)), dtype=np.float32)
    resnet(random_image)
    resnet.load_weights("./weights/resnet")
    return resnet

def get_img(img_path):
    img = read_file(img_path)
    img = decode_jpeg(img, channels=3)
    img = resize(img, [image_size, image_size])
    img = img/255.0
    return img

def train():
    resnet = get_resnet()
    opt = AdamOptimizer(1e-6)
    images_data = get_classification_data("data/data_classification_train.json")
    count = 0
    print("Training started")
    while True: # Until we manually stop it. I dont care about over-fitting for now - I just want to see if any results will come
        shuffle(images_data)
        for (i, label) in images_data:
            img = get_img("./pictures/pictures_classification_train/{}.png".format(i))
            def get_loss():
                img_vector = tf.convert_to_tensor([img], dtype=np.float32)
                logits = resnet(img_vector)
                entropy = sparse_softmax_cross_entropy_with_logits(labels=[label], logits=logits)
                entropy = tf.gather(entropy, 0)
                save_data(label, logits[0].numpy().tolist(), entropy.numpy().tolist())
                return entropy
            opt.minimize(get_loss)
            count += 1
            if (count % 1000 == 0):
                resnet.save_weights("./weights/resnet")
        print("Weights saved")

def evaluate():
    resnet = get_resnet()
    images_data = get_classification_data("data/data_classification_evaluate.json")
    for (i, label) in images_data:
        img = get_img("./pictures/pictures_classification_evaluate/{}.png".format(i))
        img_vector = tf.convert_to_tensor([img], dtype=np.float32)
        logits = resnet(img_vector)
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
