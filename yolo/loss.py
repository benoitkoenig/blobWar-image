import numpy as np
import tensorflow as tf
from tensorflow.nn import sigmoid_cross_entropy_with_logits, sparse_softmax_cross_entropy_with_logits
from tensorflow.losses import huber_loss

from constants import nb_class

def calculate_loss(preds, true_labels, true_boxes, true_probs):
    boxes = tf.slice(preds, [0, 0, 0, 0], [-1, -1, -1, 4])
    probs = tf.slice(preds, [0, 0, 0, 4], [-1, -1, -1, 1])
    classification_logits = tf.slice(preds, [0, 0, 0, 5], [-1, -1, -1, nb_class])

    boxes_loss = huber_loss(predictions=boxes, labels=true_boxes)
    probs_loss = sigmoid_cross_entropy_with_logits(logits=probs, labels=true_probs)
    classification_loss = sparse_softmax_cross_entropy_with_logits(logits=classification_logits, labels=true_labels)

    boxes_loss = tf.reduce_mean(boxes_loss)
    probs_loss = tf.reduce_mean(probs_loss)
    classification_loss = tf.reduce_mean(classification_loss)

    return boxes_loss, probs_loss, classification_loss
