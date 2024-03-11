from sklearn.metrics import classification_report
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

if True:
    def normalize(input_image, input_mask):  # normalize input and convert mask format
        # print("start: " + str(input_mask))
        # convert color
        input_mask = input_mask[:, :, :1]
        # normalize

        input_image = tf.cast(input_image, tf.float32) / 255.0
        # print("end: " + str(input_mask))
        return input_image, input_mask


    def read_test_image(image_file, mask_file):
        image = tf.io.read_file(test_image_dir + image_file)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        mask = tf.io.read_file(test_mask_dir + mask_file)
        mask = tf.image.decode_image(mask, channels=3, expand_animations=False)
        image = tf.convert_to_tensor(image)
        mask = tf.convert_to_tensor(mask)
        image = tf.image.resize(image, (224, 224), method=ResizeMethod.NEAREST_NEIGHBOR)
        mask = tf.image.resize(mask, (224, 224), method=ResizeMethod.NEAREST_NEIGHBOR)
        # print("start: " + str(mask))
        return image, mask


    def display(display_list):
        plt.figure(figsize=(15, 15))

        title = ['Input Image', 'True Mask', 'Clear Data', 'Clear Augmented', 'Weather Data']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()


    def create_mask(pred_mask):
        pred_mask = tf.math.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]


    test_image_dir = ''
    test_mask_dir = ''


    def init_dataset(path):
        global test_mask_dir, test_image_dir
        test_image_dir = path + 'rgb/'
        test_mask_dir = path + 'seg/'
        image_test_paths = os.listdir(test_image_dir)
        mask_test_paths = os.listdir(test_mask_dir)
        image_test_paths.sort()
        # image_test_paths.remove('.DS_Store')
        mask_test_paths.sort()
        test_images = tf.data.Dataset.from_tensor_slices((image_test_paths, mask_test_paths))
        test_images = test_images.map(read_test_image, num_parallel_calls=tf.data.AUTOTUNE).map(normalize)
        test_batches = test_images.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return test_batches


    def full_eval(name):
        print(name)
        test_set = init_dataset('Images/test_data/' + name + '/')
        model.evaluate(test_set)


    ################################################################################

    BUFFER_SIZE = 1000
    BATCH_SIZE = 50

    # model1 = tf.keras.models.load_model('models/Towns1-7,10/50epochs.keras')
    # images, true_masks = tuple(zip(*test_images))

    model1 = tf.keras.models.load_model('models/Towns1-7,10/50epochs.keras')
    model2 = tf.keras.models.load_model('models/Towns1-7,10/50epochs_albumentation.keras')
    model3 = tf.keras.models.load_model('models/Towns1-7,10/50epochs_weather.keras')
    test_set = init_dataset('Images/test_data/TNW/')
    for image, mask in test_set.take(10):
        pred_mask1 = model1.predict(image)
        pred_mask2 = model2.predict(image)
        pred_mask3 = model3.predict(image)
        display([image[0], mask[0], create_mask(pred_mask1), create_mask(pred_mask2), create_mask(pred_mask3)])

    # model = tf.keras.models.load_model('models/Towns1-7,10/50epochs_albumentation.keras')
    # print('ALBUMENTATION CLEAR WEATHER DATA')
    # full_eval('TDC')
    # full_eval('TDR')
    # full_eval('TDW')
    # full_eval('TNC')
    # full_eval('TNR')
    # full_eval('TNW')
    # full_eval('TW')

    # # Predict the masks
    # predicted_masks = model1.predict(test_batches)
    # print(predicted_masks.shape)
    # display([create_mask(predicted_masks[0])])
    #
    # # Convert predicted masks to class labels
    # predicted_labels = [np.argmax(mask, axis=-1).flatten() for mask in predicted_masks]
    #
    # # Flatten true masks
    # true_labels_flat = np.concatenate([mask.numpy().flatten() for mask in true_masks])
    #
    # # Flatten predicted labels
    # predicted_labels_flat = np.concatenate(predicted_labels)
    #
    # # Print classification report
    # print(classification_report(true_labels_flat, predicted_labels_flat))
