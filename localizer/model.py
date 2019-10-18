from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, SpatialDropout2D, BatchNormalization, Conv2D


def get_conv_model(initial_channels=32):
    inputs = Input(shape=(None, None, 1))

    x_0 = Conv2D(initial_channels, (3, 3), strides=(2, 2), activation='relu')(inputs)
    x_0 = SpatialDropout2D(.1)(x_0)

    x_1 = Conv2D(initial_channels * 2 ** 1, (3, 3), strides=(2, 2), activation='relu')(x_0)
    x_1 = BatchNormalization()(x_1)
    x_1 = SpatialDropout2D(.1)(x_1)

    x_2 = Conv2D(initial_channels * 2 ** 2, (3, 3), strides=(2, 2), activation='relu')(x_1)
    x_2 = BatchNormalization()(x_2)
    x_2 = SpatialDropout2D(.1)(x_2)

    x_2 = Conv2D(initial_channels * 2 ** 2, (3, 3), strides=(1, 1), activation='relu')(x_2)
    x_2 = BatchNormalization()(x_2)
    x_2 = SpatialDropout2D(.1)(x_2)

    x_2 = Conv2D(initial_channels * 2 ** 2, (3, 3), strides=(1, 1), activation='relu')(x_2)
    x_2 = BatchNormalization()(x_2)
    x_2 = SpatialDropout2D(.1)(x_2)

    x_2 = Conv2D(initial_channels * 2 ** 2, (3, 3), strides=(1, 1), activation='relu')(x_2)
    x_2 = BatchNormalization()(x_2)
    x_2 = SpatialDropout2D(.1)(x_2)

    x_2 = Conv2D(initial_channels * 2 ** 2, (3, 3), strides=(1, 1), activation='relu')(x_2)
    x_2 = BatchNormalization()(x_2)

    x_3 = Conv2D(initial_channels * 2 ** 3, (3, 3), strides=(1, 1), activation='relu')(x_2)
    x_3 = BatchNormalization()(x_3)

    xu_3 = Conv2D(initial_channels, (1, 1), strides=(1, 1), activation='relu')(x_3)
    xu_3 = Conv2D(4, (1, 1), activation='sigmoid')(xu_3)

    model = Model(inputs=inputs, outputs=xu_3)

    return model


def get_train_model(initial_channels=32, model_factory=get_conv_model):
    conv_model = model_factory(initial_channels)

    inputs = Input(shape=(128, 128, 1))
    x = conv_model(inputs)
    x = Lambda(lambda t: t[:, 2, 2, :])(x)

    model = Model(inputs=inputs, outputs=x)

    return model
