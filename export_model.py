import os
from tkinter import filedialog

from config_reader import read_config_file
from model import get_model


path_cfg = filedialog.askopenfilename()
path_ckpt = filedialog.askopenfilename()
path_ckpt = "".join(os.path.splitext(path_ckpt)[:-1])

# read params out of config file
params = read_config_file(path_cfg)
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

model = get_model(shape_input=shape_input, base_encoder=base_encoder, multi_scale_extractor=multi_scale_extractor,
                  dilation_rates=dilation_rates, skip_connections=skip_connections, decoder_steps=decoder_steps,
                  weight_regularization=regularization)

model.load_weights(path_ckpt)

model.save(filepath='exported')
