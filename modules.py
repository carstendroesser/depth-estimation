import tensorflow as tf


def vgg_module(append_to):
    out = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(append_to)
    out = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(out)
    out = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(out)
    out = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(out)
    out = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(out)
    out = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(out)
    out = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(out)
    out = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(out)
    out = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(out)
    out = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(out)
    # out = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(out)
    # out = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(out)
    # out = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(out)
    return out


def densenet_bottleneck(append_to):
        out = tf.keras.layers.BatchNormalization()(append_to)
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)(out)
        out = tf.keras.layers.Dropout(rate=0.2)(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.ReLU()(out)
        out = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same')(out)
        return tf.keras.layers.Dropout(rate=0.2)(out)


def densenet_transition(append_to, downsample):
    append_to = tf.keras.layers.BatchNormalization()(append_to)
    append_to = tf.keras.layers.ReLU()(append_to)
    append_to = tf.keras.layers.Conv2D(filters=append_to.shape[-1]//2, kernel_size=(1, 1), strides=1, padding='same')(
        append_to)
    append_to = tf.keras.layers.Dropout(rate=0.2)(append_to)

    if downsample is True:
        return tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(append_to)
    else:
        return append_to



def densenet_block(append_to, count_layers):
    layers_to_concat = list()
    layers_to_concat.append(append_to)

    out = densenet_bottleneck(append_to=append_to)
    layers_to_concat.append(out)

    for i in range(count_layers-1):
        out = tf.concat(layers_to_concat, axis=3)
        out = densenet_bottleneck(out)
        layers_to_concat.append(out)

    return tf.concat(layers_to_concat, axis=3)


def densenet_module(append_to):
    # header
    encoder = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(append_to)
    encoder = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2)(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.ReLU()(encoder)
    #encoder = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(encoder)
    #encoder = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(encoder)

    # denseblock 1
    encoder = densenet_block(append_to=encoder, count_layers=6)
    encoder = densenet_transition(append_to=encoder, downsample=True)

    #denseblock 2
    encoder = densenet_block(append_to=encoder, count_layers=12)
    encoder = densenet_transition(append_to=encoder, downsample=True)

    #denseblock 3
    encoder = densenet_block(append_to=encoder, count_layers=32)
    encoder = densenet_transition(append_to=encoder, downsample=False)

    #-> input downsampled to 1/8

    return encoder


def create_resnet_block(append_to, output_filter_count, parallel_conv):
    left = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(append_to)
    left = tf.keras.layers.BatchNormalization()(left)
    left = tf.keras.layers.ReLU()(left)
    left = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(left)
    left = tf.keras.layers.BatchNormalization()(left)
    left = tf.keras.layers.ReLU()(left)
    left = tf.keras.layers.Conv2D(filters=output_filter_count, kernel_size=(3, 3), strides=1, padding='same')(left)
    left = tf.keras.layers.BatchNormalization()(left)

    if parallel_conv:
        append_to = tf.keras.layers.Conv2D(filters=output_filter_count, kernel_size=(3, 3), strides=1, padding='same')(
            append_to)
        append_to = tf.keras.layers.BatchNormalization()(append_to)

    output = tf.keras.layers.add([left, append_to])
    output = tf.keras.layers.Activation(activation='relu')(output)

    return output


def resnet_module(append_to):
    # head
    encoder = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(append_to)
    encoder = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.ReLU()(encoder)
    encoder = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(encoder)
    encoder = tf.keras.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding='valid')(encoder)

    # block1
    encoder = create_resnet_block(encoder, output_filter_count=256, parallel_conv=True)
    encoder = create_resnet_block(encoder, output_filter_count=256, parallel_conv=False)
    encoder = create_resnet_block(encoder, output_filter_count=256, parallel_conv=False)

    # block2
    left = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(encoder)
    left = tf.keras.layers.BatchNormalization()(left)
    left = tf.keras.layers.ReLU()(left)
    left = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(left)
    left = tf.keras.layers.BatchNormalization()(left)
    left = tf.keras.layers.ReLU()(left)
    left = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(left)
    left = tf.keras.layers.BatchNormalization()(left)

    right = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')(encoder)
    right = tf.keras.layers.BatchNormalization()(right)

    encoder = tf.keras.layers.add([left, right])
    encoder = tf.keras.layers.ReLU()(encoder)

    # downscaling to 128 filters. not part of usual ResNet
    encoder = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(encoder)

    return encoder


def asp_block(filtercount, kernel_size, rate, append_to, double_singled_conv):
    asp = tf.keras.layers.Conv2D(filters=filtercount, kernel_size=kernel_size, strides=1, dilation_rate=rate,
                                 padding='same')(append_to)
    # 1x1 to reduce param count and dimensionality
    asp = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same')(asp)
    asp = tf.keras.layers.LeakyReLU()(asp)
    asp = tf.keras.layers.BatchNormalization()(asp)
    asp = tf.keras.layers.Dropout(0.2)(asp)
    if double_singled_conv:
        asp = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same')(asp)
        asp = tf.keras.layers.LeakyReLU()(asp)
        asp = tf.keras.layers.BatchNormalization()(asp)
        asp = tf.keras.layers.Dropout(0.2)(asp)
    return asp


def aspp_module(filters, dilation_rates, append_to):
    aspp0 = asp_block(filtercount=filters, kernel_size=1, rate=1, append_to=append_to, double_singled_conv=False)
    aspp1 = asp_block(filtercount=filters, kernel_size=3, rate=dilation_rates[0], append_to=append_to,
                      double_singled_conv=False)
    aspp2 = asp_block(filtercount=filters, kernel_size=3, rate=dilation_rates[1], append_to=append_to,
                      double_singled_conv=False)
    aspp3 = asp_block(filtercount=filters, kernel_size=3, rate=dilation_rates[2], append_to=append_to,
                      double_singled_conv=False)
    concatenated = tf.keras.layers.concatenate([aspp0, aspp1, aspp2, aspp3])
    return concatenated


def wasp_module(filters, dilation_rates, append_to):
    wasp0 = asp_block(filtercount=filters, kernel_size=1, rate=1, append_to=append_to, double_singled_conv=True)
    wasp1 = asp_block(filtercount=filters, kernel_size=3, rate=dilation_rates[0], append_to=wasp0,
                      double_singled_conv=True)
    wasp2 = asp_block(filtercount=filters, kernel_size=3, rate=dilation_rates[1], append_to=wasp1,
                      double_singled_conv=True)
    wasp3 = asp_block(filtercount=filters, kernel_size=3, rate=dilation_rates[2], append_to=wasp2,
                      double_singled_conv=True)
    concatenated = tf.keras.layers.concatenate([wasp0, wasp1, wasp2, wasp3])
    return concatenated


def ilp_module(append_to):
    # avg -> reshape -> 1x1 conv
    ilp = tf.reduce_mean(append_to, axis=[1, 2])
    ilp = tf.reshape(ilp, (tf.shape(ilp)[0], 1, 1, ilp.shape[1]))  # (x) -> (none, 1, 1, x), alternative: tf.stack
    ilp = tf.keras.layers.Conv2D(ilp.shape[-1], kernel_size=1, strides=1, padding='same')(ilp)
    return ilp


def upsampling(factor_h, factor_w, filtercount, append_to):
    up = tf.keras.layers.UpSampling2D(size=(factor_h, factor_w), interpolation='bilinear')(append_to)
    up = tf.keras.layers.Conv2D(filtercount, kernel_size=(3, 3), strides=1, padding='same')(up)
    up = tf.keras.layers.LeakyReLU(alpha=0.2)(up)
    up = tf.keras.layers.BatchNormalization()(up)
    return up
