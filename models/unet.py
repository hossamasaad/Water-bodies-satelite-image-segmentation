from keras.models import Model
from keras.layers import (
    Conv2D,
    Dropout,
    Input,
    Activation,
    concatenate,
    MaxPooling2D,
    Conv2DTranspose,
)


def encoder_block(inputs, n_filters, dropout_rate):
    x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = Activation('relu')(x)
    f = x
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=dropout_rate)(x)

    return x, f


def encoder(inputs):
    x, f1 = encoder_block(inputs, n_filters=64, dropout_rate=0.3)
    x, f2 = encoder_block(x, n_filters=128, dropout_rate=0.3)
    x, f3 = encoder_block(x, n_filters=256, dropout_rate=0.3)
    x, f4 = encoder_block(x, n_filters=512, dropout_rate=0.3)
    return x, (f1, f2, f3, f4)


def bottleneck(inputs):
    x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(x)
    x = Activation('relu')(x)
    return x


def decoder_block(inputs, f, n_filters, dropout_rate):
    x = Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=2, padding='same')(inputs)
    x = concatenate([x, f])
    x = Dropout(rate=dropout_rate)(x)
    x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(x)
    x = Activation('relu')(x)
    return x


def decoder(inputs, convs):
    f1, f2, f3, f4 = convs
    x = decoder_block(inputs, f4, n_filters=512, dropout_rate=0.3)
    x = decoder_block(x, f3, n_filters=256, dropout_rate=0.3)
    x = decoder_block(x, f2, n_filters=128, dropout_rate=0.3)
    x = decoder_block(x, f1, n_filters=64, dropout_rate=0.3)

    outputs = Conv2D(filters=3, kernel_size=(1, 1), activation='softmax')(x)

    return outputs


def Unet():
    inputs = Input(shape=(256, 256, 3,))

    encoder_output, convs = encoder(inputs)
    bottle_neck = bottleneck(encoder_output)
    outputs = decoder(bottle_neck, convs)

    model = Model(inputs, outputs)

    return model