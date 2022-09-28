"""
auto encoder which y data is locked
"""
import os
import sys
import tensorflow as tf
import model
import plot
import noise

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

max_image_size: int = 128
train_data_size: int = 1024

if len(sys.argv) < 2:
    print("Usage: python3 main.py <path to target>")
    sys.exit(1)

# Load target image
target: str = sys.argv[1]
target: tf.Tensor = tf.io.read_file(target)
target: tf.Tensor = tf.image.decode_png(target, channels=3)
target: tf.Tensor = tf.image.convert_image_dtype(target, tf.float32)
target = target / 255.0

# If image size is bigger than 128x128, downsample it with same ratio
if target.shape[0] > max_image_size or target.shape[1] > max_image_size:
    if target.shape[0] > target.shape[1]:
        target = tf.image.resize(
            target, (max_image_size, int(target.shape[1] * max_image_size / target.shape[0])))
    else:
        target = tf.image.resize(
            target, (int(target.shape[0] * max_image_size / target.shape[1]), max_image_size))
    print("target image downsampled: ", target.shape)

# Compute size that nearist multiple of 2**5 = 32
size_x = target.shape[0]
size_y = target.shape[1]
while size_x % 32 != 0:
    size_x += 1
while size_y % 32 != 0:
    size_y += 1

# Padding
padding_x = size_x - target.shape[0]
padding_y = size_y - target.shape[1]
target = tf.image.pad_to_bounding_box(target, 0, 0, size_x, size_y)

# make train data
train_X: tf.Tensor = tf.stack(
    [noise.make_noise_image(target.shape[:2]) for _ in range(train_data_size)])
train_Y: tf.Tensor = tf.stack(
    [target for _ in range(train_data_size)])

# Create the autoencoder with the target image size
autoencoder = model.make_autoencoder(image_size=(target.shape[0], target.shape[1]))

autoencoder.fit(
    x=train_X,
    y=train_Y,
    epochs=32,
    batch_size=128,
    validation_data=(train_X, train_Y)
)

# Save model
model_save_path = sys.argv[1].split("/")[-1].split(".")[0] + ".h5"
autoencoder.save(model_save_path)
print("model saved: ", model_save_path)

prediction = autoencoder.predict(tf.expand_dims(train_X[0], axis=0))
prediction = tf.squeeze(prediction, axis=0)

plot.plot_images([
    tf.image.crop_to_bounding_box(
        target, 0, 0, target.shape[0] - padding_x, target.shape[1] - padding_y),
    tf.image.crop_to_bounding_box(
        train_X[0], 0, 0, target.shape[0] - padding_x, target.shape[1] - padding_y),
    tf.image.crop_to_bounding_box(
        prediction, 0, 0, prediction.shape[0] - padding_x, prediction.shape[1] - padding_y)
])
