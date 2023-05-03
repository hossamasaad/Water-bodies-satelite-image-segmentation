from keras.models import Model
from keras.layers import (
    add,
    Input,
    Conv2D,
    Dropout, 
    Activation,
    Cropping2D,
    MaxPooling2D,
    Conv2DTranspose, 
    BatchNormalization
)


def block_base(inputs, n_filters, stride):
    shortcut = inputs

    x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same", strides=stride)(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same", strides=1)(x)
    x = BatchNormalization()(x)

    if stride != 1:
        shortcut = Conv2D(filters=n_filters, kernel_size=(1, 1), strides=stride)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x


def res_block(inputs, n_filters, stride):
    x = block_base(inputs, n_filters, stride)
    x = block_base(x, n_filters, stride)
    return x


def ResNet18_backbone(inputs):

    x = Conv2D(64, (7, 7), strides=2, padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=2)(x)

    b1 = res_block(x , n_filters=64 , stride=1)
    b2 = res_block(b1, n_filters=128, stride=2)
    b3 = res_block(b2, n_filters=256, stride=2)
    b4 = res_block(b3, n_filters=512, stride=2)

    return b1, b2, b3, b4


def s_block(vertical_input, lateral_input, n_filters):

    shared = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same')

    inputs = add([vertical_input, lateral_input])

    x = shared(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = shared(x)
    x = BatchNormalization()(x)

    output = add([inputs, x])
    output = Activation('relu')(output)

    return output


def conv1x1_block(inputs, n_filters):
    b = Conv2D(filters=n_filters, kernel_size=(1, 1), padding="same")(inputs)
    b = BatchNormalization()(b)
    b = Activation("relu")(b)
    return b


def conv_trans(inputs, n_filters):
    d = Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=2, padding='same')(inputs)
    d = Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=2, padding='same')(d)
    return d


def conv_stride(inputs, n_filters):
    en = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=2, padding="same")(inputs)
    en = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=2, padding="same")(en)
    return en


def ShelfNet():
    # Inputs
    inputs = Input(shape=(224, 224, 3,))

    # Backbone
    b1, b2, b3, b4 = ResNet18_backbone(inputs)
    f_input = b4

    # 1 - 1*1 Convs
    b1 = conv1x1_block(b1, n_filters=64)
    b2 = conv1x1_block(b2, n_filters=128)
    b3 = conv1x1_block(b3, n_filters=256)
    b4 = conv1x1_block(b4, n_filters=512)

    # 2 - Decoder
    d2 = s_block(vertical_input=f_input, lateral_input=b4, n_filters=512)

    d2 = conv_trans(d2, n_filters=256)
    d2 = s_block(vertical_input=d2, lateral_input=b3, n_filters=256)

    shortcut_d_1 = d2

    d2 = conv_trans(d2, n_filters=128)
    d2 = Cropping2D(cropping=(1, 1))(d2)
    d2 = s_block(vertical_input=d2, lateral_input=b2, n_filters=128)

    shortcut_d_2 = d2

    d2 = conv_trans(d2, n_filters=64)
    d2 = Cropping2D(cropping=(1, 1))(d2)
    d2 = s_block(vertical_input=d2, lateral_input=b1, n_filters=64)

    shortcut_d_3 = d2

    # 3 - Encoder

    en3 = s_block(vertical_input=d2, lateral_input=shortcut_d_3, n_filters=64)

    shortcut_e_1 = en3

    en3 = conv_stride(en3, n_filters=128)
    en3 = s_block(vertical_input=en3, lateral_input=shortcut_d_2, n_filters=128)

    shortcut_e_2 = en3

    en3 = conv_stride(en3, n_filters=256)
    en3 = s_block(vertical_input=en3, lateral_input=shortcut_d_1, n_filters=256)

    shortcut_e_3 = en3

    en3 = conv_stride(en3, n_filters=512)

    # 4 - Decoder
    d4 = s_block(vertical_input=en3, lateral_input=en3, n_filters=512)

    d4 = conv_trans(d4, 256)
    d4 = s_block(vertical_input=d4, lateral_input=shortcut_e_3, n_filters=256)

    d4 = conv_trans(d4, 128)
    d4 = Cropping2D(cropping=(1, 1))(d4)
    d4 = s_block(vertical_input=d4, lateral_input=shortcut_e_2, n_filters=128)

    d4 = conv_trans(d4, 64)
    d4 = Cropping2D(cropping=(1, 1))(d4)
    d4 = s_block(vertical_input=d4, lateral_input=shortcut_e_1, n_filters=64)

    # output
    output = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(d4)

    model = Model(inputs=inputs, outputs=output)

    return model