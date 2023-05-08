import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, AveragePooling2D, Concatenate, UpSampling2D, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50



IMAGE_SIZE = 256



class ConvBlock(tf.keras.layers.Layer):
    
    def __init__(self, filters=256, kernel_size=3, dilation_rate=1, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        
        self.net = Sequential([
            Conv2D(filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate, use_bias=False, kernel_initializer='he_normal'),
            BatchNormalization(), 
            ReLU()
        ])
    
    def call(self, X):
        return self.net(X)
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters":self.filters,
            "kernel_size":self.kernel_size,
            "dilation_rate":self.dilation_rate,
        }

def AtrousSpatialPyramidPooling(X):
    B, H, W, C = X.shape
    
    # Image Pooling
    image_pool = AveragePooling2D(pool_size=(H, W), name="ASPP-AvgPool")(X)
    image_pool = ConvBlock(kernel_size=1, name="ASPP-ImagePool-CB")(image_pool)
    image_pool = UpSampling2D(size=(H//image_pool.shape[1], W//image_pool.shape[2]), name="ASPP-ImagePool-UpSample")(image_pool)
    
    # Atrous Oprtations
    conv_1  = ConvBlock(kernel_size=1, dilation_rate=1, name="ASPP-CB-1")(X)
    conv_6  = ConvBlock(kernel_size=3, dilation_rate=6, name="ASPP-CB-6")(X)
    conv_12 = ConvBlock(kernel_size=3, dilation_rate=12, name="ASPP-CB-12")(X)
    conv_18 = ConvBlock(kernel_size=3, dilation_rate=18, name="ASPP-CB-18")(X)
    
    # Combine All
    combined = Concatenate(name="ASPP-Combine")([image_pool, conv_1, conv_6, conv_12, conv_18])
    processed = ConvBlock(kernel_size=1, name="ASPP-Net")(combined)
    
    # Final Output
    return processed


def DeepLab():

    # Input
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # Base Mode
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)


    # ASPP Phase
    DCNN = resnet50.get_layer('conv4_block6_2_relu').output
    ASPP = AtrousSpatialPyramidPooling(DCNN)
    ASPP = UpSampling2D(size=(IMAGE_SIZE//4//ASPP.shape[1], IMAGE_SIZE//4//ASPP.shape[2]))(ASPP)


    # LLF Phase
    LLF = resnet50.get_layer('conv2_block3_2_relu').output
    LLF = ConvBlock(filters=48, kernel_size=1)(LLF)


    # Combined
    combined = Concatenate(axis=-1)([ASPP, LLF])
    features = ConvBlock()(combined)
    features = ConvBlock()(features)
    upsample = UpSampling2D(size=(IMAGE_SIZE//features.shape[1], IMAGE_SIZE//features.shape[1]), interpolation='bilinear')(features)


    # Output Mask
    PredMask = Conv2D(1, kernel_size=3, strides=1, padding='same', activation='sigmoid', use_bias=False)(upsample)

    # DeelLabV3+ Model
    model = Model(inputs, PredMask, name="DeepLabV3-Plus")

    return model