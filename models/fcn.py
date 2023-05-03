from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D, Activation, Add, Input


class FCN8(Model):
    def __init__(self, n_classes):
        super(FCN8, self).__init__(name='FCN-8')

        # Encoder
        self.conv1a = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv1b = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv2a = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv2b = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv3a = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv3b = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv3c = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv4a = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv4b = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv4c = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.max_pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.conv5a = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv5b = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv5c = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.max_pool5 = MaxPooling2D(pool_size=(2, 2), strides=2)

        self.extra1 = Conv2D(filters=4096, kernel_size=(7, 7), activation="relu", padding="same", name="conv6")
        self.extra2 = Conv2D(filters=4096, kernel_size=(1, 1), activation="relu", padding="same", name="conv7")

        # Decoder
        # path 1.1
        self.convT1 = Conv2DTranspose(filters=n_classes, kernel_size=(4, 4), strides=2, use_bias=False)
        self.cropT1 = Cropping2D(cropping=(1, 1))

        # path 1.2
        self.conv1d1 = Conv2D(filters=n_classes, kernel_size=(1, 1), padding='same', activation='relu')

        # ADD-1
        self.add1 = Add()

        # path 2.1
        self.convT2 = Conv2DTranspose(filters=n_classes, kernel_size=(4, 4), strides=2, use_bias=False)
        self.cropT2 = Cropping2D(cropping=(1, 1))

        # path 2.2
        self.conv1d2 = Conv2D(filters=n_classes, kernel_size=(1, 1), padding='same', activation='relu')

        # ADD-2
        self.add2 = Add()

        # output
        self.convT3 = Conv2DTranspose(filters=n_classes, kernel_size=(8, 8), strides=8, use_bias=False)
        self.softmax = Activation("softmax")

        self.output_layer = Conv2D(filters=1, kernel_size=1, strides=1, activation='sigmoid', padding='same', name="OutputLayer")

    def call(self, inputs):
        # Encoder #
        # Block 1
        x = self.conv1a(inputs)
        x = self.conv1b(x)
        x = self.max_pool1(x)

        # Block 2
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.max_pool2(x)

        # Block 3
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.conv3c(x)
        x = self.max_pool3(x)
        pool3 = x

        # Block 4
        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.conv4c(x)
        x = self.max_pool4(x)
        pool4 = x

        # Block 5
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.conv5c(x)
        x = self.max_pool5(x)

        x = self.extra1(x)
        x = self.extra2(x)

        pool5 = x

        # Decoder #
        # Path 1.1
        path11 = self.convT1(pool5)
        path11 = self.cropT1(path11)

        # Path 1.2
        path12 = self.conv1d1(pool4)

        # Path 2.1 = path1.1 + path1.2
        path21 = self.add1([path11, path12])
        path21 = self.convT2(path21)
        path21 = self.cropT2(path21)

        # path 2.2
        path22 = self.conv1d2(pool3)

        # Add
        add = self.add2([path21, path22])

        # output
        output = self.convT3(add)
        output = self.output_layer(output)

        return output