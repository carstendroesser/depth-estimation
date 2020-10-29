import os
import shutil
from tkinter import filedialog

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.callbacks as callbacks

import utils
from config_reader import read_config_file
from dataset import get_dataset
from losses import loss_fn
from model import get_model

# train from scratch or continue training of an existing model?
CONTINUE_TRAINING = False

path_config = 'model.cfg'

if CONTINUE_TRAINING:
    path_config = filedialog.askopenfilename()
    path_ckpt = filedialog.askopenfilename()
    path_ckpt = "".join(os.path.splitext(path_ckpt)[:-1])

# input password for email-updates
mail_password = input("pw:")

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

# create model name out of params
if CONTINUE_TRAINING:
    model_name = utils.concatenate_model_name(params) + '_continued'
else:
    model_name = utils.concatenate_model_name(params)

# notify start
utils.send_update("Started training "
                  + model_name,
                  'carsten.droesser@gmail.com',
                  mail_password,
                  'carsten.droesser@gmail.com')

# create dataset
train_dataset, train_count, validation_dataset, validation_count = get_dataset(images_path=images_path,
                                                                               yamls_path=yamls_path,
                                                                               max_depth=max_depth,
                                                                               min_depth=min_depth,
                                                                               shape_input=shape_input,
                                                                               shape_depthmap=shape_depthmap,
                                                                               batch_size=batch_size,
                                                                               validation_split=0.2)

# create model
model = get_model(shape_input=shape_input, base_encoder=base_encoder, multi_scale_extractor=multi_scale_extractor,
                  dilation_rates=dilation_rates, skip_connections=skip_connections, decoder_steps=decoder_steps,
                  weight_regularization=regularization)

# create optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)

# create early-stopping callback to auto-detect overfitting
# removed because of undesired behavior
# cb_early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, mode='auto')

# create checkpoint callback to auto-save weights
path_checkpoint = model_name + "/cp-{epoch:04d}.ckpt"
cb_checkpoint = callbacks.ModelCheckpoint(filepath=path_checkpoint, verbose=1, save_weights_only=True,
                                          mode='auto', save_freq='epoch')

# compile model
model.compile(optimizer=optimizer, loss=loss_fn)

# continue training?
if CONTINUE_TRAINING:
    model.load_weights(path_ckpt)

# plot model
if not os.path.exists(path=model_name):
    os.mkdir(path=model_name)
    # use on high-end machines only:
    #tf.keras.utils.plot_model(model, model_name + '/model.png', show_shapes=True)
else:
    raise Exception("Path does already exist")

# copy config-file
shutil.copyfile(src=path_config, dst=model_name + "/model.cfg")

# start training
history = model.fit(x=train_dataset, epochs=epochs, steps_per_epoch=train_count // batch_size,
                    validation_data=validation_dataset, validation_steps=validation_count // batch_size,
                    callbacks=[cb_checkpoint])

# plot history
matplotlib.use('Agg')
plt.plot(history.history['loss'], label='train', color='tab:blue')
plt.plot(history.history['val_loss'], label='validation', color='tab:orange')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title(model_name, fontdict={'fontsize': 8})
plt.savefig(model_name + '/train_and_val_loss.png', format='png')

# notify when finished
utils.send_update("Training model "
                  + model_name
                  + " finished. ",
                  'carsten.droesser@gmail.com',
                  mail_password,
                  'carsten.droesser@gmail.com')
