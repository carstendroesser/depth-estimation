import os
from tkinter import filedialog

import numpy as np

import utils
from config_reader import read_config_file
from dataset import get_dataset
from metrics import metrics
from model import get_model

# ask for paths
path_config = filedialog.askopenfilename()
path_ckpt = filedialog.askopenfilename()
path_ckpt = "".join(os.path.splitext(path_ckpt)[:-1])

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

# create model
model = get_model(shape_input=shape_input, base_encoder=base_encoder, multi_scale_extractor=multi_scale_extractor,
                  dilation_rates=dilation_rates, skip_connections=skip_connections, decoder_steps=decoder_steps,
                  weight_regularization=regularization)

# load checkpoint
model.load_weights(path_ckpt)

# load validation dataset
validation_dataset, validation_count = get_dataset(images_path=images_path,
                                                   yamls_path=yamls_path,
                                                   max_depth=max_depth,
                                                   shape_input=shape_input,
                                                   shape_depthmap=shape_depthmap,
                                                   batch_size=batch_size,
                                                   validation_split=0.2)[-2:]

i = 0
print("validation count: ", validation_count)

errors = []

for element in validation_dataset:
    if not i < 3:
        break

    i = i + 1

    print("shape of gt: ", element[1].shape)

    # predict n = batch_size images
    predictions = model.predict(element[0], batch_size=batch_size)
    predictions = predictions * max_depth
    predictions = np.abs(predictions)
    predictions = np.clip(predictions, 0, max_depth)
    errors.append(metrics(element[1], predictions))

mean = np.mean(np.stack(errors, axis=0), axis=0)
print("MEAN errors: ", mean)

# for single prediction use only:
# path_img = filedialog.askopenfilename()
# image = cv2.imread(path_img)
# image = prepare_image(image, shape_input)
# image = tf.expand_dims(image, axis=0)
# prediction = model.predict(image)
# prediction = prediction * max_depth
# prediction = np.abs(prediction)
# prediction = np.squeeze(prediction)
# prediction = np.clip(prediction, 0, max_depth)
# plt.imshow(prediction, cmap='Wistia', vmin=0, vmax=max_depth)
# plt.show()
