from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dropout, Flatten, SpatialDropout2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D


def get_conv_model(initial_channels=32):
    inputs = Input(shape=(None, None, 1))

    x = Conv2D(initial_channels, (5, 5), strides=(2, 2), activation='relu')(inputs)
    x = SpatialDropout2D(.1)(x)
    x = Conv2D(initial_channels, (3, 3), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(.1)(x)
    x = Conv2D(initial_channels, (3, 3), strides=(2, 2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(.1)(x)
    x = Conv2D(initial_channels * 2**1, (3, 3), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout2D(.1)(x)
    x = Conv2D(initial_channels * 2**1, (3, 3), strides=(2, 2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(initial_channels * 2**2, (3, 3), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(initial_channels * 2**3, (3, 3), strides=(2, 2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(initial_channels * 2**3, (3, 3), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(initial_channels * 2**3, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Dropout(.5)(x)
    x = Conv2D(4, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)

    return model


def get_train_model(initial_channels=32, model_factory=get_conv_model):
    conv_model = model_factory(initial_channels)

    inputs = Input(shape=(128, 128, 1))
    x = conv_model(inputs)
    x = Flatten()(x)

    model = Model(inputs=inputs, outputs=x)

    return model