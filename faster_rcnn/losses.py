import numpy as np
import tensorflow as tf
from tensorflow.nn import sigmoid_cross_entropy_with_logits, sparse_softmax_cross_entropy_with_logits

def get_localization_loss(rpn_class, target):
    localization_logits = tf.reshape(rpn_class, [-1])
    localization_labels = np.reshape(np.copy(target), (-1)) # copy is important due to labels going from 0 to 1
    localization_labels[localization_labels != 0] = 1
    localization_labels = tf.convert_to_tensor(localization_labels, dtype=np.float32)
    localization_loss = sigmoid_cross_entropy_with_logits(labels=localization_labels, logits=localization_logits)
    localization_loss = tf.reduce_mean(localization_loss)
    return localization_loss

def get_regression_loss():
    # TODO
    return 0

def get_classification_loss(classification_logits, labels_boxes):
    classification_loss = sparse_softmax_cross_entropy_with_logits(logits=classification_logits, labels=labels_boxes)
    classification_loss = tf.reduce_mean(classification_loss)
    return classification_loss
