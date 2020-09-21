import numpy
import tensorflow as tf
import tensorflow.keras.backend as K


def loss_fn(y_truth, y_predicted):
    # gets called per depthmap and not per batch
    max_depth_value = K.max(y_truth)

    # pointwise depth error
    loss_depth = K.mean(K.abs(y_predicted - y_truth), axis=-1)

    # image gradients
    dy_truth, dx_truth = tf.image.image_gradients(y_truth)
    dy_predicted, dx_predicted = tf.image.image_gradients(y_predicted)
    loss_gradients = K.mean(K.abs(dy_predicted - dy_truth) + K.abs(dx_predicted - dx_truth), axis=-1)

    # Structural similarity (SSIM) index lim -> 0
    # ssim can be in range [-1, 1] so it is clipped to 1
    loss_similarity = K.clip((1 - tf.image.ssim(y_truth, y_predicted, max_depth_value)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = 0.1

    loss = (w1 * loss_similarity) + (w2 * K.mean(loss_gradients)) + (w3 * K.mean(loss_depth))
    return loss
