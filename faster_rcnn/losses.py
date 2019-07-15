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

def get_regression_loss(values, boxes, bounding_box_target, probs):
    # Values contains the predicted values of the regression: x, y, w, h
    # boxes are the boxes x1, y1, x2, y2
    # boxes is a mapping. For an element at position y, x, it has an array with the location of the real bounding box: center_x, center_y, w, h
    target = []
    is_foreground = []
    for i in range(len(boxes)):
        b = boxes[i]

        center_x = (b[0] + b[2]) // 2
        center_y = (b[1] + b[3]) // 2
        w = (b[2] - b[0])
        h = (b[3] - b[1])

        t = bounding_box_target[center_y, center_x] # t is a list with the actual center_x, center_y, w, h

        if (t[2] == 0): # If the target width is 0, that means the center is actually background
            is_foreground.append([0, 0, 0, 0])
            target.append([0, 0, 0, 0])
        else:
            is_foreground.append([1, 1, 1, 1])
            # The end goal is that actual_x = center_x + value[0], actual_y = center_y + value[1], actual_w = box_w + value[2], actual_h = box_h + value[3]
            # In proper faster-rcnn, w and h are updated via the ratio of the log of the value. This would add unnecessary complication here
            target_x = t[0] - center_x
            target_y = t[1] - center_y
            target_w = t[2] - w
            target_h = t[2] - h
            target.append([target_x, target_y, target_w, target_h])

    probs_formated = [2 * p ** 5 for p in probs]
    probs_formated = [[p, p, p, p] for p in probs_formated]

    target = tf.convert_to_tensor(target, dtype=np.float32)
    x = tf.math.abs(values - target)
    x_below_1 =  tf.cast(x < 1, dtype=tf.float32)

    loss = x_below_1 * .5 * tf.math.square(x) + (1 - x_below_1) * (x - .5)
    loss = tf.multiply(is_foreground, loss) # Regression do not apply on background
    loss = tf.multiply(probs_formated, loss) # First focus on localization - then regression
    loss = tf.reduce_mean(loss)
    return loss
