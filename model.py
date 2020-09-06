import tensorflow as tf
from tensorflow.keras import layers

from modules import densenet_module, resnet_module, vgg_module, aspp_module, wasp_module, ilp_module, upsampling


def get_model(shape_input, base_encoder, multi_scale_extractor, dilation_rates, skip_connections, decoder_steps,
              weight_regularization):
    # input layer
    inputs = tf.keras.Input(shape=(shape_input[0], shape_input[1], shape_input[2]), name="image_input")

    # base-encoder: densenet_module or resnet_module
    encoder = None
    if base_encoder == 'resnet':
        encoder = resnet_module(append_to=inputs)
    elif base_encoder == 'vgg':
        encoder = vgg_module(append_to=inputs)
    elif base_encoder == 'densenet':
        encoder = densenet_module(append_to=inputs)
    else:
        raise Exception("No valid argument for base_encoder. base_encoder has to be one of [resnet, densenet, vgg].")

    # local features: aspp_module or wasp_module
    local_branch = None
    if multi_scale_extractor == 'aspp':
        local_branch = aspp_module(filters=encoder.shape[-1], dilation_rates=dilation_rates,
                                   append_to=encoder)
    elif multi_scale_extractor == 'wasp':
        local_branch = wasp_module(filters=encoder.shape[-1], dilation_rates=dilation_rates, append_to=encoder)
    else:
        raise Exception("No valid argument for local_branch. local_branch has to be one of [wasp, aspp].")

    # global features
    global_branch = ilp_module(append_to=encoder)
    global_branch = upsampling(factor_h=shape_input[0] // 8, factor_w=shape_input[1] // 8,
                               filtercount=global_branch.shape[-1] / 2,
                               append_to=global_branch)

    if skip_connections == 'skipped':
        local_branch = layers.concatenate([encoder, local_branch])
        global_branch = layers.concatenate([encoder, global_branch])  # skip connection skips ilp module

    # branch merging
    decoder = layers.concatenate([local_branch, global_branch])

    # upsamling
    if decoder_steps == 'single_step':
        decoder = upsampling(factor_h=8, factor_w=8, filtercount=decoder.shape[-1] / 4, append_to=decoder)
    elif decoder_steps == 'double_step':
        decoder = upsampling(factor_h=2, factor_w=2, filtercount=decoder.shape[-1] / 2, append_to=decoder)
        decoder = upsampling(factor_h=4, factor_w=4, filtercount=decoder.shape[-1] / 2, append_to=decoder)
    elif decoder_steps == 'triple_step':
        decoder = upsampling(factor_h=2, factor_w=2, filtercount=decoder.shape[-1] / 2, append_to=decoder)
        decoder = upsampling(factor_h=2, factor_w=2, filtercount=decoder.shape[-1] / 2, append_to=decoder)
        decoder = upsampling(factor_h=2, factor_w=2, filtercount=decoder.shape[-1] / 2, append_to=decoder)
    else:
        raise Exception(
            "No valid argument for decoder_steps. decoder_steps has to be one of [single_step, double_step, triple_step]")

    # depth output layer
    decoder = layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(decoder)

    # model
    model = tf.keras.Model(inputs=inputs, outputs=decoder)

    if weight_regularization == 'regularized':
        for layer in model.layers:
            layer.kernel_regularizer = tf.keras.regularizers.l2(l=0.001)
    elif weight_regularization != 'unregularized':
        raise Exception(
            "No valid argument for regularization. regularization has to be one of [regularized, unregularized]")

    return model
