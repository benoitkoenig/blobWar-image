import numpy as np
from random import shuffle
import sys
import tensorflow as tf
from tensorflow.image import decode_jpeg, resize
from tensorflow.io import read_file
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.nn import softmax, sparse_softmax_cross_entropy_with_logits
from tensorflow.train import AdamOptimizer

tf.enable_eager_execution()

from constants import image_size
from model import BlobClassifierModel
from preprocess import get_classification_data
from tracking import save_data

weights_path = "./weights/weights"

def get_model():
    my_model = BlobClassifierModel()
    random_image = tf.convert_to_tensor(np.random.random((1, image_size, image_size, 3)), dtype=np.float32)
    my_model(random_image)
    my_model.load_weights(weights_path)
    return my_model

def reset_model():
    my_model = BlobClassifierModel()
    random_image = tf.convert_to_tensor(np.random.random((1, image_size, image_size, 3)), dtype=np.float32)
    my_model(random_image)
    my_model.save_weights(weights_path)

def get_img(img_path):
    img = read_file(img_path)
    img = decode_jpeg(img, channels=3)
    img = resize(img, [image_size, image_size])
    img = img/255.0
    return img

def train():
    my_model = get_model()
    opt = AdamOptimizer(1e-4)
    images_data = get_classification_data("data/data_classification_train.json")
    count = 0
    print("Training started")
    while True: # Until we manually stop it. I dont care about over-fitting for now - I just want to see if any results will come
        shuffle(images_data)
        for (i, label) in images_data:
            img = get_img("./pictures/pictures_classification_train/{}.png".format(i))
            def get_loss():
                img_vector = tf.convert_to_tensor([img], dtype=np.float32)
                logits = my_model(img_vector)
                entropy = sparse_softmax_cross_entropy_with_logits(labels=[label], logits=logits)
                entropy = tf.gather(entropy, 0)
                save_data(label, logits[0].numpy().tolist(), entropy.numpy().tolist())
                return entropy
            opt.minimize(get_loss)
            count += 1
            if (count % 1000 == 0):
                my_model.save_weights(weights_path)
                print("Weights saved")

def evaluate():
    my_model = get_model()
    images_data = get_classification_data("data/data_classification_evaluate.json")
    count = 0
    for (i, label) in images_data:
        img = get_img("./pictures/pictures_classification_evaluate/{}.png".format(i))
        img_vector = tf.convert_to_tensor([img], dtype=np.float32)
        logits = my_model(img_vector).numpy()[0]
        if (np.argmax(logits) == label):
            count += 1
            print("X {} {}".format(label, logits))
        else:
            print("  {} {}".format(label, logits))
    print("Number of probs where label prob is the max: {}/{}".format(count, len(images_data)))

reseting = False
training = False
evaluating = False
for instruction in sys.argv:
    if (instruction == "reset"):
        reseting = True
    if (instruction == "train"):
        training = True
    if (instruction == "evaluate"):
        evaluating = True

if reseting:
    reset_model()
if training:
    train()
if evaluating:
    evaluate()
if (training == False) & (evaluating == False) & (reseting == False):
    print("Usage: 'python main_classification.py [train, evaluate, reset]'")
