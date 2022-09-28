"""
view saved model
"""
import os
import sys
import tensorflow as tf
from tensorflow import keras
import plot
import noise

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if len(sys.argv) < 2:
    print("Usage: python3 view_model.py <path to model>")
    sys.exit(1)

model_path: str = sys.argv[1]
model: keras.Model = keras.models.load_model(model_path)

input_size = model.input.shape[1:3]
#make noise image
noise_image = noise.make_noise_image(input_size)
#predict
prediction = model.predict(tf.expand_dims(noise_image, axis=0))
prediction = tf.squeeze(prediction, axis=0)

plot.plot_images([noise_image, prediction])
