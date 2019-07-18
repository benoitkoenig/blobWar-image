import numpy as np
import tensorflow as tf
from tensorflow.nn import sigmoid_cross_entropy_with_logits, sparse_softmax_cross_entropy_with_logits
from tensorflow.losses import huber_loss

from constants import nb_class, feature_size

def smooth_L1_loss(pred, target):
    # target = tf.convert_to_tensor(target, dtype=np.float32)
    x = tf.math.abs(pred - target)
    x_below_1 =  tf.cast(x < 1, dtype=tf.float32)

    return x_below_1 * .5 * tf.math.square(x) + (1 - x_below_1) * (x - .5)

def calculate_loss(preds, true_labels, true_boxes, true_probs):
    boxes = tf.slice(preds, [0, 0, 0, 0], [-1, -1, -1, 4])
    probs = tf.slice(preds, [0, 0, 0, 4], [-1, -1, -1, 1])
    classification_logits = tf.slice(preds, [0, 0, 0, 5], [-1, -1, -1, nb_class])

    boxes_loss = smooth_L1_loss(boxes, true_boxes) # TODO: Use IoU loss instead
    probs_loss = sigmoid_cross_entropy_with_logits(logits=probs, labels=true_probs)
    classification_loss = sparse_softmax_cross_entropy_with_logits(logits=classification_logits, labels=true_labels)

    # boxes_loss = tf.multiply(true_probs.reshape(1, feature_size, feature_size), boxes_loss)
    classification_loss = tf.multiply(true_probs.reshape(1, feature_size, feature_size), classification_loss)

    boxes_loss = tf.reduce_mean(boxes_loss)
    probs_loss = tf.reduce_mean(probs_loss)
    classification_loss = tf.reduce_mean(classification_loss)

    return boxes_loss, probs_loss, classification_loss
