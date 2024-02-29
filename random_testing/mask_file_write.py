import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import cv2


def normalize(input_mask):  # normalize input and convert mask format
    # print("start: " + str(input_mask))
    # # input_mask = input_mask[:, :, :1] - 1
    # # normalize
    # input_image = tf.cast(input_image, tf.float32) / 255.0
    # # print("end: " + str(input_mask))
    # return input_image, input_mask
    return input_mask


def read_mask(file):
    mask = tf.io.read_file(mask_dir + file)
    mask = tf.image.decode_image(mask, channels=3, expand_animations=False)
    mask = tf.convert_to_tensor(mask)
    mask = tf.image.resize(mask, (20, 20), method=ResizeMethod.NEAREST_NEIGHBOR)

    return mask


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


#
# TRAIN_LENGTH = 800  # info.splits['train'].num_examples
# VALIDATION_LENGTH = 200
# BATCH_SIZE = 50
# BUFFER_SIZE = 1000
# STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
#
mask_dir = '../Images/data2/train/seg/'
# print("hi")
# image_paths = os.listdir(image_dir)
# mask_paths = os.listdir(mask_dir)
# image_paths.sort()
# mask_paths.sort()
#
# image_train_paths = image_paths[:VALIDATION_LENGTH]
# image_val_paths = image_paths[VALIDATION_LENGTH:]
# mask_train_paths = mask_paths[:VALIDATION_LENGTH]
# mask_val_paths = mask_paths[VALIDATION_LENGTH:]
#
# train_images = tf.data.Dataset.from_tensor_slices((image_train_paths, mask_train_paths))
# train_images = train_images.map(read_image).map(normalize)
# val_images = tf.data.Dataset.from_tensor_slices((image_val_paths, mask_val_paths))
# val_images = val_images.map(read_image).map(normalize)
#
# train_batches = (
#     train_images
#     .cache()
#     .batch(BATCH_SIZE)
#     .repeat()
#     .prefetch(buffer_size=tf.data.AUTOTUNE))
#
# val_batches = val_images.batch(BATCH_SIZE)
# np.random.seed(0)

np.set_printoptions(threshold=np.inf)

cv_mask = cv2.imread('../Images/data2/train/seg/seg2.png')
tf.io.write_file('cv_mask.txt', str(cv_mask))
print(cv_mask.max())
print(cv_mask.min())
sample_mask = normalize(read_mask('seg2.png'))
tf.io.write_file('sample_mask.txt', str(sample_mask.numpy()[:, :, 0]))
# print(sample_mask.numpy())
display([sample_mask])

# CARLA converted
# Building - 3
# Sky - 11
# Road - 1
# Flag/Sign - 22
# Road Line - 24
# Vegetation - 9
# Fence/Post - 20
# Sidewalk - 25
#
#
#
#
#
#
#
#
