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
    my_model.save_weights(weights_path)
    print("Weights saved")

def evaluate(num):
    my_model = get_model()
    images_data = get_classification_data("data/data_classification_evaluate_{}.json".format(num))
    count = 0
    for (i, label) in images_data:
        img = get_img("./pictures/pictures_classification_evaluate_{}/{}.png".format(num, i))
        img_vector = tf.convert_to_tensor([img], dtype=np.float32)
        logits = my_model(img_vector).numpy()[0]
        if (np.argmax(logits) == label):
            count += 1
            print("  {} {}".format(label, logits.tolist()))
        else:
            print("X {} {}".format(label, logits.tolist()))
    print("Number of probs where label prob is the max: {}/{}".format(count, len(images_data)))

instruction = None
if len(sys.argv) > 1:
    instruction = sys.argv[1]
param = None
if len(sys.argv) > 2:
    param = sys.argv[2]

if (instruction == "reset"):
    reset_model()
elif (instruction == "train"):
    train()
elif (instruction == "evaluate"):
    if (param == "100") | (param == "10000"):
        evaluate(eval(param))
    else:
        print("Usage: 'python main_classification.py evaluate [100, 10000]'")
else:
    print("Usage: 'python main_classification.py [train, evaluate, reset]'")
