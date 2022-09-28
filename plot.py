"""
plot functions
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def plot_images(images: list[tf.Tensor]) -> None:
    """
    Plot the images.
    Args:
        images: The images to plot.
    Returns:
        None.
    """
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(keras.preprocessing.image.array_to_img(image))
    plt.show()
