from keras.models import Model
from keras.layers import Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, Cropping2D, Activation, Add, Input, Activation

def encoder(input):

    # Block-1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x

    # Block-2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block-3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # Block-4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # Block-5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x

    return f1 , f2 , f3 , f4 , f5


def decoder(n_classes, f5, f4, f3):
 
    o = f5
    o = Conv2D(4096, (7, 7), activation='relu', padding='same')(o)
    o = Dropout(0.3)(o)
    o = Conv2D(4096, (1, 1), activation='relu', padding='same')(o)
    o = Dropout(0.3)(o)

    o = Conv2D(n_classes, (1,1), activation='relu')(o)
    o = Conv2DTranspose(n_classes, kernel_size=(4,4), strides=(2,2), use_bias=False)(o)
    o = Cropping2D(((1, 1), (1, 1)))(o)

    o2 = f4
    o2 = Conv2D(n_classes, (1, 1), activation='relu')(o2)
	
    o = Add()([o, o2])
    o = Conv2DTranspose(n_classes, kernel_size=(4,4), strides=(2,2), use_bias=False)(o)
    o = Cropping2D(((1, 1), (1, 1)))(o)
  
    o2 = f3 
    o2 =Conv2D(n_classes, (1, 1), activation='relu')(o2)
  
    o  = Add()([o2, o])
    o = Conv2DTranspose(n_classes, kernel_size=(16,16), strides=(8,8), use_bias=False)(o)
    o = Cropping2D(((4, 4), (4, 4)))(o)
    o = Activation("sigmoid")(o)

    return o


def FCN8():
    input = Input(shape=(256, 256, 3))
    f1, f2, f3, f4, f5 = encoder(input)
    output = decoder(1, f5, f4, f3)
    model = Model(input, output)
    return model                                                                                                                                          