import tensorflow as tf
from tensorflow.math import log, exp

def calculate_segment_loss(segment, true_mask):
    prod = tf.multiply(segment, true_mask)
    unit_val = log(1 + exp(-prod))
    return tf.reduce_sum(unit_val)

def calculate_score_loss(score, y_k):
    x = tf.gather(tf.gather(score, 0), 0)
    return log(1 + exp(-y_k * x))

def calculate_loss(segment, score, true_masks):
    (true_mask, y_k) = true_masks[0]
    segment_loss = calculate_segment_loss(segment, true_mask)
    score_loss = calculate_score_loss(score, y_k)
    return (1 + y_k) * segment_loss + score_loss
