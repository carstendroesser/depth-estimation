import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops


# nan?!
def silog(y_truth, y_predicted):
    err = K.log(y_predicted) - K.log(y_truth)
    err = K.sqrt(K.mean(err ** 2) - K.mean(err) ** 2) * 100
    return err


# nan?!
def rmse_log(y_truth, y_predicted):
    error = (K.log(y_truth) - K.log(y_predicted)) ** 2
    return K.sqrt(K.mean(error))


def rel_abs(y_truth, y_predicted):
    return K.mean(math_ops.div_no_nan(K.abs(y_truth - y_predicted), y_truth))


def rel_sq(y_truth, y_predicted):
    return K.mean(math_ops.div_no_nan(((y_truth - y_predicted) ** 2), y_truth))
