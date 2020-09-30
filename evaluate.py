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
min_depth = float(params[17])

# create identifying model_name
model_name = utils.concatenate_model_name(params)

# create folder to save summary and plots
if not os.path.exists(model_name):
    os.mkdir(model_name)

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
                                                   min_depth=min_depth,
                                                   shape_input=shape_input,
                                                   shape_depthmap=shape_depthmap,
                                                   batch_size=batch_size,
                                                   validation_split=0.2)[-2:]

# track time
time_start = time.time()

batches_to_plot = [12, 32, 62, 81]
errors = []

i = 0
j = 0

for element in validation_dataset:
    print("Processing batch", str(i), "of", str(validation_count // batch_size))
    predictions = model.predict(element[0], batch_size=batch_size)

    # prevent division by zero
    predictions = max_depth / predictions
    predictions = np.clip(predictions, min_depth, max_depth)

    y_truth = max_depth / element[1]

    # crop
    images = utils.crop_center(element[0], (shape_input[0] // 8) * 5, (shape_input[1] // 8) * 5)
    predictions = utils.crop_center(predictions, (shape_input[0] // 8) * 5, (shape_input[1] // 8) * 5)
    y_truth = utils.crop_center(y_truth, (shape_input[0] // 8) * 5, (shape_input[1] // 8) * 5)

    errors.append(metrics(y_truth, predictions))

    if i in batches_to_plot:
        figure, axarr = plt.subplots(batch_size, 3)

        axarr[0, 0].set_title('Image', fontdict={'fontsize': 8})
        axarr[0, 1].set_title('Ground Truth', fontdict={'fontsize': 8})
        axarr[0, 2].set_title('Predicted', fontdict={'fontsize': 8})

        k = 0
        for image, pred, gt in zip(images, predictions, y_truth):
            axarr[k, 0].imshow(np.squeeze(image))
            axarr[k, 0].axis('off')
            axarr[k, 1].imshow(np.squeeze(gt), cmap='plasma', vmin=min_depth, vmax=max_depth)
            axarr[k, 1].axis('off')
            axarr[k, 2].imshow(np.squeeze(pred), cmap='plasma', vmin=min_depth, vmax=max_depth)
            axarr[k, 2].axis('off')
            k = k + 1

        figure.suptitle(model_name, fontsize=8)
        plt.savefig('{}/batched_comparison_{}.png'.format(model_name, j), format='png', dpi=300)
        plt.show(dpi=300)
        j = j + 1

    if i < (validation_count // batch_size):
        i = i + 1
    else:
        break

time_finish = time.time()
time_elapsed = time_finish - time_start

errors = np.stack(errors, axis=0)
errors = np.mean(errors, axis=0)
errors = np.round(errors, 3)

# count params, ctrl+c/v from source: https://stackoverflow.com/a/58894981/1478054
count_trainable = np.sum([K.count_params(w) for w in model.trainable_weights])
count_non_trainable = np.sum([K.count_params(w) for w in model.non_trainable_weights])

file_summary = open(model_name + '/summary.txt', "w")

file_summary.write('\nModel: ' + utils.concatenate_model_name(params))
file_summary.write('\nevaluated predictions: ' + str(validation_count))
file_summary.write('\nelapsed time: ' + time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))
file_summary.write('\n\nParams:')
file_summary.write('\ntotal: {:,}'.format(count_trainable + count_non_trainable))
file_summary.write('\ntrainable: {:,}'.format(count_trainable))
file_summary.write('\nnon-trainable: {:,}'.format(count_non_trainable))
file_summary.write('\n\nErrors:')
file_summary.write('\nrel_abs: ' + str('{:.3f}'.format(errors[0])))
file_summary.write('\nrel_squared: ' + str('{:.3f}'.format(errors[1])))
file_summary.write('\nrmse: ' + str('{:.3f}'.format(errors[2])))
file_summary.write('\nrmse_log: ' + str('{:.3f}'.format(errors[3])))
file_summary.write('\nlog10: ' + str('{:.3f}'.format(errors[4])))
file_summary.write('\nacc1: ' + str('{:.3f}'.format(errors[5])))
file_summary.write('\nacc2: ' + str('{:.3f}'.format(errors[6])))
file_summary.write('\nacc3: ' + str('{:.3f}'.format(errors[7])))
file_summary.close()
