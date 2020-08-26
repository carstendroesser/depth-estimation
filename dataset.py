import os
import random

import cv2
import numpy as np
import tensorflow as tf


def iterator(dictionary, max_depth, shape_input, shape_depthmap):
    while True:
        for entry in dictionary:
            # image
            img_array = cv2.imread(entry[0].decode("utf-8"))
            resized_img_array = cv2.resize(img_array, (shape_input[1], shape_input[0]), interpolation=cv2.INTER_NEAREST)
            resized_img_array = cv2.cvtColor(resized_img_array, cv2.COLOR_BGR2RGB)
            resized_img_array = tf.cast(resized_img_array, tf.float32) * (1. / 255.0)

            # depthmap
            yaml_file = cv2.FileStorage(entry[1].decode("utf-8"), cv2.FILE_STORAGE_READ)
            depthmap = yaml_file.getNode(
                ("depthmap_" + os.path.splitext(os.path.basename(entry[1].decode("utf-8")))[0])).mat()
            yaml_file.release()
            min_depth = 0.5
            depthmap = np.clip(depthmap, min_depth, max_depth)
            resized_depthmap = cv2.resize(depthmap, (shape_depthmap[1], shape_depthmap[0]),
                                          interpolation=cv2.INTER_NEAREST)
            resized_depthmap = resized_depthmap / np.amax(resized_depthmap) * np.amax(depthmap)
            resized_depthmap = tf.expand_dims(resized_depthmap, -1)

            yield resized_img_array, resized_depthmap


def get_dataset(images_path, yamls_path, max_depth, shape_input, shape_depthmap, batch_size, validation_split):
    dictionary = []
    for single_image_name in os.listdir(images_path):
        single_image_path = os.path.join(images_path, single_image_name)
        single_yaml_path = os.path.join(yamls_path, os.path.splitext(single_image_name)[0] + ".yml")
        dictionary.append([single_image_path, single_yaml_path])

    # shuffle data, but always shuffle the same way
    shuffled_dictionary = dictionary[:]
    random.seed(27)
    random.shuffle(shuffled_dictionary)

    # split data into validation and train set
    validation_start_index = int(round((1.0 - validation_split) * len(shuffled_dictionary)))

    # [0 - validation_start_index]
    train_dictionary = shuffled_dictionary[:validation_start_index]
    # [validation_start_index - end]
    validation_dictionary = shuffled_dictionary[validation_start_index:]

    train_dataset = tf.data.Dataset.from_generator(iterator,
                                                   args=(train_dictionary, max_depth, shape_input, shape_depthmap),
                                                   output_types=(tf.float32, tf.float32),
                                                   output_shapes=(shape_input, shape_depthmap))

    validation_dataset = tf.data.Dataset.from_generator(iterator,
                                                        args=(
                                                            validation_dictionary, max_depth, shape_input,
                                                            shape_depthmap),
                                                        output_types=(tf.float32, tf.float32),
                                                        output_shapes=(shape_input, shape_depthmap))

    train_dataset = train_dataset.batch(batch_size)
    train_count = len(train_dictionary)
    validation_dataset = validation_dataset.batch(batch_size)
    validation_count = len(validation_dictionary)

    return train_dataset, train_count, validation_dataset, validation_count
