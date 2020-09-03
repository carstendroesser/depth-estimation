import os
import time
from tkinter import filedialog

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K

import utils
from config_reader import read_config_file
from dataset import get_dataset
from metrics import metrics
from model import get_model

# increase default dpi for plots
mpl.rcParams['figure.dpi'] = 300

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


# track time
time_start = time.time()

test = True
errors = []

for element in validation_dataset:
    predictions = model.predict(element[0], batch_size=batch_size)
    predictions = max_depth / predictions
    predictions = np.clip(predictions, 0.5, max_depth)

    y_truth = max_depth / element[1]
    errors.append(metrics(y_truth, predictions))

    if test is True:
        break

time_finish = time.time()
time_elapsed = time_finish - time_start

errors = np.stack(errors, axis=0)
errors = np.mean(errors, axis=0)
errors = np.round(errors, 3)

# count params, ctrl+c/v from source: https://stackoverflow.com/a/58894981/1478054
count_trainable = np.sum([K.count_params(w) for w in model.trainable_weights])
count_non_trainable = np.sum([K.count_params(w) for w in model.non_trainable_weights])

print('----------------')
print('Summary for ', utils.concatenate_model_name(params))
print('Evaluation of', validation_count * batch_size, 'images took',
      time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))
print('---- Params: ----------------')
print('total: {:,}'.format(count_trainable + count_non_trainable))
print('trainable: {:,}'.format(count_trainable))
print('non-trainable: {:,}'.format(count_non_trainable))
print('---- Errors: ----------------')
print("rel_abs:", '{:.3f}'.format(errors[0]))
print("rel_squared:", '{:.3f}'.format(errors[1]))
print("rmse:", '{:.3f}'.format(errors[2]))
print("rmse_log:", '{:.3f}'.format(errors[3]))
print("log10:", '{:.3f}'.format(errors[4]))
print("acc1:", '{:.3f}'.format(errors[5]))
print("acc2:", '{:.3f}'.format(errors[6]))
print("acc3:", '{:.3f}'.format(errors[7]))

# single image use only:
# prediction = model.predict(resized_img_array)
# prediction = (max_depth)/prediction
# prediction = np.squeeze(prediction)
# prediction = np.clip(prediction, 0.5, max_depth)
# print("aha", prediction)
# print("aha", np.amax(prediction))
# plt.imshow(prediction, cmap='plasma', vmin=0, vmax=max_depth)
##plt.imshow(prediction, cmap='Greys', vmin=0, vmax=np.amax(prediction))
# plt.colorbar()
# plt.title(model_name, fontdict = {'fontsize' : 8})
##plt.imshow(img_array)
# plt.show()
