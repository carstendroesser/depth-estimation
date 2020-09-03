import numpy as np


# absolute relative error
def rel_abs(y_truth, y_predicted):
    return np.mean(np.abs(y_truth - y_predicted) / y_truth)


# squared relative error
def rel_squared(y_truth, y_predicted):
    return np.mean(np.divide(((y_truth - y_predicted) ** 2), y_truth))


# root mean squared error
def rmse(y_truth, y_predicted):
    return np.sqrt(np.mean((y_truth - y_predicted) ** 2))


# average log10 error
def log10(y_truth, y_predicted):
    return (np.abs(np.log10(y_truth) - np.log10(y_predicted))).mean()


# thresholded accuracy
def thresholded_accuracy(y_truth, y_predicted):
    t = np.maximum((y_truth / y_predicted), (y_predicted / y_truth))
    return (t < 1.25).mean(), (t < 1.25 ** 2).mean(), (t < 1.25 ** 3).mean()


# rmse log
def rmse_log(y_truth, y_predicted):
    error = (np.log(y_truth) - np.log(y_predicted)) ** 2
    return np.sqrt(error.mean())


def metrics(y_truth, y_predicted):
    e1 = rel_abs(y_truth, y_predicted)
    e2 = rel_squared(y_truth, y_predicted)
    e3 = rmse(y_truth, y_predicted)
    e4 = rmse_log(y_truth, y_predicted)
    e5 = log10(y_truth, y_predicted)
    e6a, e6b, e6c = thresholded_accuracy(y_truth, y_predicted)
    return [e1, e2, e3, e4, e5, e6a, e6b, e6c]
