import tensorflow as tf


def vgg_module(append_to):
    # according to VGG16, but without dense blocks -> 'VGG13'
    vgg = vgg_unit(append_to=append_to, count_layers=2, kernel_sizes=(3, 3), filters=64, downsample=True)
    vgg = vgg_unit(append_to=vgg, count_layers=2, kernel_sizes=(3, 3), filters=128, downsample=True)
    vgg = vgg_unit(append_to=vgg, count_layers=3, kernel_sizes=(3, 3, 1), filters=256, downsample=True)
    vgg = vgg_unit(append_to=vgg, count_layers=3, kernel_sizes=(3, 3, 1), filters=512, downsample=False)
    vgg = vgg_unit(append_to=vgg, count_layers=3, kernel_sizes=(3, 3, 1), filters=512, downsample=False)
    return vgg


def vgg_unit(append_to, count_layers, kernel_sizes, filters, downsample):
    for i in range(count_layers):
        append_to = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_sizes[i], strides=(1, 1),
                                           padding='same')(append_to)
        append_to = tf.keras.layers.ReLU()(append_to)

    if downsample is True:
        return tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(append_to)
    else:
        return append_to


def densenet_unit(append_to):
    # DropOut was removed because of no benefits
    out = tf.keras.layers.BatchNormalization()(append_to)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    return tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same')(out)


def densenet_transition(append_to, downsample):
    # DropOut was removed because of no benefits
    append_to = tf.keras.layers.BatchNormalization()(append_to)
    append_to = tf.keras.layers.ReLU()(append_to)
    append_to = tf.keras.layers.Conv2D(filters=append_to.shape[-1] // 2, kernel_size=(1, 1), strides=1, padding='same')(
        append_to)

    if downsample is True:
        return tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(append_to)
    else:
        return append_to


def densenet_block(append_to, count_layers):
    layers_to_concat = list()
    layers_to_concat.append(append_to)

    out = densenet_unit(append_to=append_to)
    layers_to_concat.append(out)

    for i in range(count_layers - 1):
        out = tf.concat(layers_to_concat, axis=3)
        out = densenet_unit(out)
        layers_to_concat.append(out)

    return tf.concat(layers_to_concat, axis=3)


def densenet_module(append_to):
    # header
    encoder = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(append_to)
    encoder = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2)(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.ReLU()(encoder)

    # removed to prevent too much downsampling in the beginning:
    # encoder = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(encoder)
    # encoder = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(encoder)

    # denseblock 1
    encoder = densenet_block(append_to=encoder, count_layers=6)
    encoder = densenet_transition(append_to=encoder, downsample=True)

    # denseblock 2
    encoder = densenet_block(append_to=encoder, count_layers=12)
    encoder = densenet_transition(append_to=encoder, downsample=True)

    # denseblock 3
    encoder = densenet_block(append_to=encoder, count_layers=48)
    encoder = densenet_transition(append_to=encoder, downsample=False)

    # -> input downsampled to 1/8
    return encoder


def resnet_unit(append_to, filters, downsample=False):
    if downsample:
        stride = 2
        identity_kernel_size = 3
    else:
        identity_kernel_size = 1
        stride = 1

    left = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=stride, padding='same')(append_to)
    left = tf.keras.layers.BatchNormalization()(left)
    left = tf.keras.layers.ReLU()(left)
    left = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3), strides=1, padding='same')(left)
    left = tf.keras.layers.BatchNormalization()(left)
    left = tf.keras.layers.ReLU()(left)
    left = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(1, 1), strides=1, padding='same')(left)
    left = tf.keras.layers.BatchNormalization()(left)
    left = tf.keras.layers.ReLU()(left)

    append_to = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(identity_kernel_size, identity_kernel_size),
                                       strides=stride, padding='same')(append_to)
    append_to = tf.keras.layers.BatchNormalization()(append_to)
    append_to = tf.keras.layers.ReLU()(append_to)

    output = tf.keras.layers.add([left, append_to])
    output = tf.keras.layers.Activation(activation='relu')(output)

    return output


def resnet_block(append_to, count_layers, filters, downsample):
    for i in range(count_layers):
        if i == 0 and downsample is True:
            downsample = True
        else:
            downsample = False
        append_to = resnet_unit(append_to=append_to, filters=filters, downsample=(downsample))

    return append_to


def resnet_module(append_to):
    # according resnet-152
    # head
    encoder = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(append_to)
    encoder = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.ReLU()(encoder)
    # removed to prevent to much downsampling in the beginning
    # encoder = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(encoder)
    # encoder = tf.keras.layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding='valid')(encoder)

    # block 1
    encoder = resnet_block(append_to=encoder, count_layers=3, filters=(64, 256), downsample=False)
    # block 2
    encoder = resnet_block(append_to=encoder, count_layers=8, filters=(128, 512), downsample=True)
    # block 3: in resnet_152, the 3rd block has 36 layers. To have similar count of parameters in comparison with densenet,
    # we experimentally reduce the count of layers in the 3rd block to 30 layers and the last filtercount to 512
    # see: https://pytorch.org/hub/pytorch_vision_resnet/
    encoder = resnet_block(append_to=encoder, count_layers=30, filters=(256, 512), downsample=True)

    return encoder


def asp_block(filtercount, kernel_size, rate, append_to, double_singled_conv):
    asp = tf.keras.layers.Conv2D(filters=filtercount, kernel_size=kernel_size, strides=1, dilation_rate=rate,
                                 padding='same')(append_to)
    # 1x1 to reduce dimensionality and param count
    asp = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same')(asp)
    asp = tf.keras.layers.LeakyReLU()(asp)
    asp = tf.keras.layers.BatchNormalization()(asp)
    # removed, because of no benefits:
    # asp = tf.keras.layers.Dropout(0.2)(asp)
    if double_singled_conv:
        asp = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same')(asp)
        asp = tf.keras.layers.LeakyReLU()(asp)
        asp = tf.keras.layers.BatchNormalization()(asp)
        # removed, because of no benefits:
        # asp = tf.keras.layers.Dropout(0.2)(asp)
    return asp


def aspp_module(filters, dilation_rates, append_to):
    # 3 branches ASPP module with 1x1 additional module
    aspp0 = asp_block(filtercount=filters, kernel_size=1, rate=1, append_to=append_to, double_singled_conv=False)
    aspp1 = asp_block(filtercount=filters, kernel_size=3, rate=dilation_rates[0], append_to=append_to,
                      double_singled_conv=False)
    aspp2 = asp_block(filtercount=filters, kernel_size=3, rate=dilation_rates[1], append_to=append_to,
                      double_singled_conv=False)
    aspp3 = asp_block(filtercount=filters, kernel_size=3, rate=dilation_rates[2], append_to=append_to,
                      double_singled_conv=False)
    concatenated = tf.keras.layers.concatenate([aspp0, aspp1, aspp2, aspp3])
    return concatenated


def wasp_block(filtercount, kernel_size, rate, append_to):
    root = tf.keras.layers.Conv2D(filters=filtercount, kernel_size=kernel_size, strides=1, dilation_rate=rate,
                                  padding='same')(append_to)
    root = tf.keras.layers.LeakyReLU()(root)
    root = tf.keras.layers.BatchNormalization()(root)

    branch = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same')(root)
    branch = tf.keras.layers.LeakyReLU()(branch)
    branch = tf.keras.layers.BatchNormalization()(branch)

    return root, branch


def wasp_module(filters, dilation_rates, append_to):
    w0_root, w0_branch = wasp_block(filtercount=filters, kernel_size=1, rate=1, append_to=append_to)
    w1_root, w1_branch = wasp_block(filtercount=filters, kernel_size=3, rate=dilation_rates[0], append_to=w0_root)
    w2_root, w2_branch = wasp_block(filtercount=filters, kernel_size=3, rate=dilation_rates[1], append_to=w1_root)
    w3_branch = wasp_block(filtercount=filters, kernel_size=3, rate=dilation_rates[2], append_to=w2_root)[1]

    return tf.keras.layers.concatenate([w0_branch, w1_branch, w2_branch, w3_branch])


def ilp_module(append_to):
    # according to DORN's Full-Image-Encoder:
    ilp = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(append_to)
    ilp = tf.keras.layers.Flatten()(ilp)
    # units were reduced to 256 (instead of 512)
    ilp = tf.keras.layers.Dense(units=256)(ilp)
    ilp = tf.reshape(ilp, (tf.shape(ilp)[0], 1, 1, ilp.shape[1]))
    ilp = tf.keras.layers.Conv2D(filters=ilp.shape[3], kernel_size=1, strides=(1, 1))(ilp)
    ilp = tf.keras.layers.ReLU()(ilp)
    ilp = tf.keras.layers.BatchNormalization()(ilp)
    enlarge = tf.constant([1, append_to.shape[1], append_to.shape[2], 1])
    ilp = tf.tile(ilp, enlarge)
    return ilp


def upsampling(factor_h, factor_w, filtercount, append_to):
    up = tf.keras.layers.UpSampling2D(size=(factor_h, factor_w), interpolation='bilinear')(append_to)
    up = tf.keras.layers.Conv2D(filtercount, kernel_size=(3, 3), strides=1, padding='same')(up)
    up = tf.keras.layers.LeakyReLU(alpha=0.2)(up)
    up = tf.keras.layers.BatchNormalization()(up)
    up = tf.keras.layers.Conv2D(filtercount, kernel_size=(3, 3), strides=1, padding='same')(up)
    up = tf.keras.layers.LeakyReLU(alpha=0.2)(up)
    up = tf.keras.layers.BatchNormalization()(up)
    return up
