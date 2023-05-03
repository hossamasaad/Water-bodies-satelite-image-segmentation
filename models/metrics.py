import tensorflow as tf


def dice_loss(y_true, y_pred, SMOOTH=1):
    y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
    nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + SMOOTH
    denominator = tf.reduce_sum( y_pred) + tf.reduce_sum(y_true) + SMOOTH
    result = tf.divide(nominator, denominator)
    return result


def iou_loss(y_true, y_pred, SMOOTH=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + SMOOTH) / (union + SMOOTH)