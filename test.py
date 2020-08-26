import os
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import utils
from config_reader import read_config_file
from model import get_model

# ask for paths
path_config = filedialog.askopenfilename()
path_ckpt = filedialog.askopenfilename()
path_ckpt = "".join(os.path.splitext(path_ckpt)[:-1])
path_img = filedialog.askopenfilename()

# read params out of config file
params = read_config_file(path_config)
shape_input = [int(params[0]), int(params[1]), int(params[2])]
shape_depthmap = [int(params[0]), int(params[1]), 1]
base_encoder = params[3]
multi_scale_extractor = params[4]
dilation_rates = [int(params[5]), int(params[6]), int(params[7])]
skip_connections = params[8]
decoder_steps = params[9]
images_path = params[10]
yamls_path = params[11]
max_depth = float(params[12])
epochs = int(params[13])
batch_size = int(params[14])
learning_rate = float(params[15])
regularization = params[16]

# create model name out of params
model_name = utils.concatenate_model_name(params)

img_array = cv2.imread(path_img)
resized_img_array = cv2.resize(img_array, (shape_input[1], shape_input[0]), interpolation=cv2.INTER_NEAREST)
resized_img_array = cv2.cvtColor(resized_img_array, cv2.COLOR_BGR2RGB)
resized_img_array = tf.cast(resized_img_array, tf.float32) * (1. / 255.0)
resized_img_array = tf.expand_dims(resized_img_array, axis=0)

# create model
model = get_model(shape_input=shape_input, base_encoder=base_encoder, multi_scale_extractor=multi_scale_extractor,
                  dilation_rates=dilation_rates, skip_connections=skip_connections, decoder_steps=decoder_steps,
                  weight_regularization=regularization)

# load checkpoint
model.load_weights(path_ckpt)

prediction = model.predict(resized_img_array)
prediction = prediction * max_depth
np.abs(prediction)
prediction = np.squeeze(prediction)
prediction = np.clip(prediction, 0, max_depth)
print("prediction ", prediction.shape)
plt.imshow(prediction, cmap='Wistia', vmin=0, vmax=max_depth)
# plt.imshow(img_array)
plt.show()
