from tensorflow.python.estimator import keras
from tensorflow.python.ops.image_ops_impl import ResizeMethod

# try:
if True:
    import tensorflow as tf
    import numpy as np
    import tensorflow_datasets as tfds
    from tensorflow_examples.models.pix2pix import pix2pix
    import cv2
    import os
    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    import torch
    import traceback


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

        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()


    TRAIN_LENGTH = 800  # info.splits['train'].num_examples
    VALIDATION_LENGTH = 200
    BATCH_SIZE = 50
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    image_dir = 'Images/data2/train/rgb/'
    mask_dir = 'Images/data2/train/seg/'
    test_image_dir = 'Images/preset_weather_data/rgb/'
    test_mask_dir = 'Images/preset_weather_data/seg/'
    print("hi")
    image_paths = os.listdir(image_dir)
    mask_paths = os.listdir(mask_dir)
    image_paths.sort()
    mask_paths.sort()

    image_train_paths = image_paths[:VALIDATION_LENGTH]
    image_val_paths = image_paths[VALIDATION_LENGTH:]
    mask_train_paths = mask_paths[:VALIDATION_LENGTH]
    mask_val_paths = mask_paths[VALIDATION_LENGTH:]

    train_images = tf.data.Dataset.from_tensor_slices((image_train_paths, mask_train_paths))
    train_images = train_images.map(read_image, num_parallel_calls=tf.data.AUTOTUNE).map(normalize)
    val_images = tf.data.Dataset.from_tensor_slices((image_val_paths, mask_val_paths))
    val_images = val_images.map(read_image, num_parallel_calls=tf.data.AUTOTUNE).map(normalize)

    image_test_paths = os.listdir(test_image_dir)
    mask_test_paths = os.listdir(test_mask_dir)
    image_test_paths.sort()
    mask_test_paths.sort()
    test_images = tf.data.Dataset.from_tensor_slices((image_test_paths, mask_test_paths))
    test_images = test_images.map(read_test_image, num_parallel_calls=tf.data.AUTOTUNE).map(normalize)

    train_batches = (
        train_images
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .repeat()
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    val_batches = val_images.batch(BATCH_SIZE)
    test_batches = test_images.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # for images, masks in train_batches.take(1):
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


    OUTPUT_CLASSES = 28

    model = unet_model(output_channels=OUTPUT_CLASSES)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    tf.keras.utils.plot_model(model, show_shapes=True)


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


    EPOCHS = 30
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = len(image_val_paths) // BATCH_SIZE // VAL_SUBSPLITS

    # model_history = model.fit(train_batches, epochs=EPOCHS,
    #                           steps_per_epoch=STEPS_PER_EPOCH,
    #                           validation_steps=VALIDATION_STEPS,
    #                           validation_data=val_batches)
    #                             # callbacks=[DisplayCallback()])
    # model.save('models/30epochs.keras')
    model = tf.keras.models.load_model('models/30epochs.keras')
    # show_predictions()
    for image, mask in test_batches.take(10):
        pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])

    # loss = model_history.history['loss']
    # val_loss = model_history.history['val_loss']
    #
    # plt.figure()
    # plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    # plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and fValidation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
# except Exception as e:
#     with open('log.txt', 'a') as f:
#         f.write(str(e))
#         f.write(traceback.format_exc())
