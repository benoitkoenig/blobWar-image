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

def get_regression_loss(regr, target):
    t = tf.convert_to_tensor([target], dtype=np.float32)

    x = regr[:, :, :, 0] - t[:, :, :, 0]
    y = regr[:, :, :, 1] - t[:, :, :, 1]

    is_foreground = tf.cast(t[:, :, :, 2] != 0, dtype=tf.float32)

    x_abs = tf.math.abs(x)
    x_bool = tf.cast(x_abs < 1, dtype=tf.float32)
    x_out = is_foreground * (x_bool * .5 * tf.math.square(x_abs) + (1 - x_bool) * (x_abs - .5))

    y_abs = tf.math.abs(y)
    y_bool = tf.cast(y_abs < 1, dtype=tf.float32)
    y_out = is_foreground * (y_bool * .5 * tf.math.square(y_abs) + (1 - y_bool) * (y_abs - .5))

    w_out = is_foreground * tf.math.log(regr[:, :, :, 2] + 1e-6) - tf.math.log(t[:, :, :, 2] + 1e-6)
    h_out = is_foreground * tf.math.log(regr[:, :, :, 3] + 1e-6) - tf.math.log(t[:, :, :, 3] + 1e-6)

    loss_x = tf.reduce_mean(x_out)
    loss_y = tf.reduce_mean(y_out)
    loss_w = tf.reduce_mean(w_out)
    loss_h = tf.reduce_mean(h_out)

    return loss_x + loss_y + loss_w + loss_h

def get_classification_loss(classification_logits, labels_boxes, probs):
    classification_loss = sparse_softmax_cross_entropy_with_logits(logits=classification_logits, labels=labels_boxes)
    classification_loss = tf.multiply(2 * probs ** 5, classification_loss) # First focus on localization - then classification
    classification_loss = tf.reduce_mean(classification_loss)
    return classification_loss
