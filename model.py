"""
auto encoder.
"""
from typing import Tuple
from tensorflow import keras

def make_autoencoder(image_size: Tuple[int, int] = (28, 28)) -> keras.Model:
    """
    Creates an autoencoder that receives 3 channels images, and downsamples three times.
    Args:
        image_size: The size of the image.
    Returns:
        The autoencoder. which is compiled with the adam optimizer and binary crossentropy loss.
    """
    input_layer = keras.layers.Input(shape=(image_size[0], image_size[1], 3))

    # Encoder
    encoder = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    encoder = keras.layers.MaxPooling2D((2, 2), padding='same')(encoder)
    encoder = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
    encoder = keras.layers.MaxPooling2D((2, 2), padding='same')(encoder)
    encoder = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
    encoder = keras.layers.MaxPooling2D((2, 2), padding='same')(encoder)
    encoder = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
    encoder = keras.layers.MaxPooling2D((2, 2), padding='same')(encoder)
    encoder = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
    encoder = keras.layers.MaxPooling2D((2, 2), padding='same')(encoder)

    # Decoder
    decoder = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder)
    decoder = keras.layers.Dense(32, activation='relu')(decoder)
    decoder = keras.layers.UpSampling2D((2, 2))(decoder)
    decoder = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(decoder)
    decoder = keras.layers.Dense(32, activation='relu')(decoder)
    decoder = keras.layers.UpSampling2D((2, 2))(decoder)
    decoder = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(decoder)
    decoder = keras.layers.Dense(32, activation='relu')(decoder)
    decoder = keras.layers.UpSampling2D((2, 2))(decoder)
    decoder = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(decoder)
    decoder = keras.layers.Dense(32, activation='relu')(decoder)
    decoder = keras.layers.UpSampling2D((2, 2))(decoder)
    decoder = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(decoder)
    decoder = keras.layers.Dense(32, activation='relu')(decoder)
    decoder = keras.layers.UpSampling2D((2, 2))(decoder)

    decoder = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(decoder)
    decoder = keras.layers.Dense(32, activation='relu')(decoder)

    decoder = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoder)

    autoencoder = keras.Model(input_layer, decoder)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder
