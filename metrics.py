import numpy as np


# absolute relative error
def rel_abs(y_truth, y_predicted):
    return np.mean(np.abs(y_truth - y_predicted) / y_truth)


# squared relative error
def rel_squared(y_truth, y_predicted):
    return np.mean(np.divide(((y_truth - y_predicted) ** 2), y_truth))


# root mean squared error
def rmse(y_truth, y_predicted):
    rmse = (y_truth - y_predicted) ** 2
    return np.sqrt(np.mean(rmse))


# average log10 error
def log10(y_truth, y_predicted):
    return (np.abs(np.log10(y_truth) - np.log10(y_predicted))).mean()


# thresholded accuracy
def thresholded_accuracy(y_truth, y_predicted):
    t = np.maximum((y_truth / y_predicted), (y_predicted / y_truth))
    return (t < 1.25).mean(), (t < 1.25 ** 2).mean(), (t < 1.25 ** 3).mean()


# silog
def silog(y_truth, y_predicted):
    error = np.log(y_predicted) - np.log(y_truth)
    return np.sqrt(np.mean(error ** 2) - np.mean(error) ** 2) * 100


def metrics(y_truth, y_predicted):
    e1 = rel_abs(y_truth, y_predicted)
    e2 = rel_squared(y_truth, y_predicted)
    e3 = rmse(y_truth, y_predicted)
    e4 = log10(y_truth, y_predicted)
    e5a, e5b, e5c = thresholded_accuracy(y_truth, y_predicted)
    e6 = silog(y_truth, y_predicted)
    return [e1, e2, e3, e4, e5a, e5b, e5c, e6]
