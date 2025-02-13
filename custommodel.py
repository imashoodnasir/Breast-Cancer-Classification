import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Input, Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    """ Residual block with skip connections """
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, kernel_size, strides=1, padding="same")(x)
    x = BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU()(x)
    
    return x

def multi_scale_feature_extraction(x):
    """ Multi-scale convolution layers """
    x1 = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x2 = Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x3 = Conv2D(64, (7, 7), padding="same", activation="relu")(x)

    x = Add()([x1, x2, x3])
    return x

def build_32_layer_cnn(input_shape=(224, 224, 3), num_classes=2):
    """ Build 32-layer CNN model inspired by YOLO, U-Net, and ResNet """
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)

    x = multi_scale_feature_extraction(x)

    for _ in range(6):
        x = residual_block(x, 64)

    x = Conv2D(128, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    for _ in range(8):
        x = residual_block(x, 128)

    x = Conv2D(256, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    for _ in range(8):
        x = residual_block(x, 256)

    x = Conv2D(512, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    for _ in range(8):
        x = residual_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model

# Example usage
model = build_32_layer_cnn()
model.summary()
