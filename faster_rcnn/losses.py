import numpy as np
import tensorflow as tf
from tensorflow.nn import sigmoid_cross_entropy_with_logits, sparse_softmax_cross_entropy_with_logits

def get_localization_loss(rpn_class, target):
    logits = tf.reshape(rpn_class, [-1])
    labels = np.copy(target) # copy is important due to labels going from 0 to 1
    labels[labels != 0] = 1
    labels = tf.convert_to_tensor(labels, dtype=np.float32)
    labels = tf.reshape(labels, [-1])

    loss = sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)
    return loss

def get_classification_loss(classification_logits, labels_boxes, probs):
    classification_loss = sparse_softmax_cross_entropy_with_logits(logits=classification_logits, labels=labels_boxes)
    classification_loss = tf.multiply(2 * probs ** 5, classification_loss) # First focus on localization - then classification
    classification_loss = tf.reduce_mean(classification_loss)
    return classification_loss
