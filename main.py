from sympy import Integer
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import albumentations as A
import tensorflow as tf
import numpy as np
from tensorflow_examples.models.pix2pix import pix2pix
import os
from IPython.display import clear_output
import matplotlib.pyplot as plt
import random

# try:
if True:
    def normalize_augment(input_image, input_mask):  # normalize input and convert mask format
        # print("start: " + str(input_mask))
        # convert color
        input_mask = input_mask[:, :, :1]
        # normalize

        input_image = tf.numpy_function(augment_func, inp=[input_image], Tout=tf.float32)
        input_image = tf.cast(input_image, tf.float32) / 255.0
        # print("end: " + str(input_mask))
        return input_image, input_mask


    def normalize(input_image, input_mask):  # normalize input and convert mask format
        # print("start: " + str(input_mask))
        # convert color
        input_mask = input_mask[:, :, :1]
        # normalize

        input_image = tf.cast(input_image, tf.float32) / 255.0
        # print("end: " + str(input_mask))
        return input_image, input_mask


    def read_image(image_file, mask_file):
        image = tf.io.read_file(image_dir + image_file)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        mask = tf.io.read_file(mask_dir + mask_file)
        mask = tf.image.decode_image(mask, channels=3, expand_animations=False)
        image = tf.convert_to_tensor(image)
        mask = tf.convert_to_tensor(mask)
        image = tf.image.resize(image, (224, 224), method=ResizeMethod.NEAREST_NEIGHBOR)
        mask = tf.image.resize(mask, (224, 224), method=ResizeMethod.NEAREST_NEIGHBOR)
        # print("start: " + str(mask))
        return image, mask


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


    def augment_func(img):
        transform = A.Compose(
            [A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=0.5),
             A.RandomSunFlare(src_radius=150, flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.5),
             A.RandomFog(fog_coef_lower=0.4, fog_coef_upper=0.5, alpha_coef=0.1, p=0.5),
             A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
             A.RandomGamma(gamma_limit=(80, 500), p=0.5)
             ])
        img = transform(image=img)['image']
        return img.astype(np.float32)


    def augment(img, mask):
        img = tf.image.random_brightness(img, max_delta=0.3)
        img = tf.image.random_hue(img, 0.4)
        return img, mask


    def unet_model(output_channels: int):
        inputs = tf.keras.layers.Input(shape=[224, 224, 3])

        # Downsampling through the model
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            filters=output_channels, kernel_size=3, strides=2,
            padding='same')  # 128x128 -> 256x256

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)


    def create_mask(pred_mask):
        pred_mask = tf.math.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]


    def show_predictions(dataset=None, num=5):
        if dataset:
            for image, mask in dataset.take(num):
                pred_mask = model.predict(image)
                display([image[0], mask[0], create_mask(pred_mask)])


    class DisplayCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            clear_output(wait=True)
            show_predictions()
            print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


    TRAIN_LENGTH = 1000
    VALIDATION_LENGTH = 200
    CV_SPLITS = 10
    SPLIT_LENGTH = int((VALIDATION_LENGTH + TRAIN_LENGTH) / CV_SPLITS)
    BATCH_SIZE = 50
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    image_dir = 'Images/diverse_weather_data/rgb/'
    mask_dir = 'Images/diverse_weather_data/seg/'
    print("hi")
    image_paths = os.listdir(image_dir)
    mask_paths = os.listdir(mask_dir)
    # image_paths.remove('.DS_Store')
    image_paths.sort()
    mask_paths.sort()
    print(image_paths)
    print(mask_paths)
    path_pairs = []  # all pairs of (image, mask)
    TOTAL_PAIRS = len(mask_paths)
    for i in range(TOTAL_PAIRS):
        path_pairs.append([image_paths[i], mask_paths[i]])
    random.shuffle(path_pairs)

    ####################################################################################
    # 10 cross validation
    for i in range(CV_SPLITS):
        val_path_pairs = np.array(path_pairs[i * SPLIT_LENGTH: (i + 1) * SPLIT_LENGTH])
        train0 = np.array(path_pairs[:i * SPLIT_LENGTH])
        train1 = np.array(path_pairs[(i + 1) * SPLIT_LENGTH:])
        if i == 0:
            train_path_pairs = train1
        elif i == CV_SPLITS - 1:
            train_path_pairs = train0
        else:
            train_path_pairs = np.concatenate((train0, train1))

        train_images = tf.data.Dataset.from_tensor_slices((train_path_pairs[:, 0], train_path_pairs[:, 1]))
        train_images = train_images.map(read_image, num_parallel_calls=tf.data.AUTOTUNE).map(
            normalize)  # only train images are augmented
        val_images = tf.data.Dataset.from_tensor_slices((val_path_pairs[:, 0], val_path_pairs[:, 1]))
        val_images = val_images.map(read_image, num_parallel_calls=tf.data.AUTOTUNE).map(normalize)

        train_batches = (
            train_images
            .cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .repeat()
            .prefetch(buffer_size=tf.data.AUTOTUNE))

        val_batches = val_images.batch(BATCH_SIZE).repeat()
        print("train images: " + str(len(train_images)))
        print("val images: " + str(len(val_images)))

        # tf.random.set_seed(3)
        # for images, masks in train_batches.take(3):
        #     sample_image, sample_mask = images[0], masks[0]
        #     display([sample_image, sample_mask])

        # MAKING THE MODEL
        base_model = tf.keras.applications.MobileNetV2(input_shape=[224, 224, 3], include_top=False)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

        down_stack.trainable = False

        up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),  # 32x32 -> 64x64
            pix2pix.upsample(32, 3),  # 16x16 -> 32x32
        ]

        OUTPUT_CLASSES = 29

        model = unet_model(output_channels=OUTPUT_CLASSES)

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # tf.keras.utils.plot_model(model, show_shapes=True)

        EPOCHS = 20
        VALIDATION_STEPS = VALIDATION_LENGTH // BATCH_SIZE
        # show_predictions()
        model_history = model.fit(train_batches, epochs=EPOCHS,
                                  steps_per_epoch=TRAIN_LENGTH // BATCH_SIZE,
                                  validation_steps=VALIDATION_STEPS,
                                  validation_data=val_batches)
        # callbacks=[DisplayCallback()])
        model.save('models/Towns1-7,10/20epochs_weatherCV' + str(i) + '.keras')

        loss = model_history.history['loss']
        val_loss = model_history.history['val_loss']

    # plt.figure()
    # plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    # plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss Value')
    # plt.ylim([0, 1])
    # plt.legend()
    # plt.show()
# except Exception as e:
#     with open('log.txt', 'a') as f:
#         f.write(str(e))
#         f.write(traceback.format_exc())
