# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:04:40 2020

@author: Jeff
"""

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output



AUTOTUNE = tf.data.experimental.AUTOTUNE

distribute_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:1")
#distribute_strategy = tf.distribute.MirroredStrategy()

with distribute_strategy.scope():
    from tensorflow_examples.models.pix2pix import pix2pix
    


##########################
#Loading the Raw Data
##########################

def parse_image(img_path: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    # For one Image path:
    # .../trainset/images/training/ADE_train_00000001.jpg
    # Its corresponding annotation path is:
    # .../trainset/annotations/training/ADE_train_00000001.png
    mask_path = tf.strings.regex_replace(img_path, "Images_Categorical", "Labels_Categorical")
    mask_path = tf.strings.regex_replace(mask_path, ".jpg", ".png")
    mask = tf.io.read_file(mask_path)
    # The masks contain a class index for each pixels
    mask = tf.image.decode_png(mask, channels=1)
    # In scene parsing, "not labeled" = 255
    # But it will mess up with our N_CLASS = 150
    # Since 255 means the 255th class
    # Which doesn't exist
    mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
    # Note that we have to convert the new value (0)
    # With the same dtype than the tensor itself

    return {'image': image, 'segmentation_mask': mask}



train_imgs = tf.data.Dataset.list_files("Z:/Unreal_Datasets/unrealmeadow/Images_Categorical/*.jpg")
val_imgs = tf.data.Dataset.list_files("Z:/Unreal_Datasets/unrealmeadow/Images_Categorical/*.jpg")

train_set = train_imgs.map(parse_image)

test_set = val_imgs.map(parse_image)

dataset = {"train": train_set, "test": test_set}

dataset

##########################
#Data Loader with Normalization and Augmentations
##########################


def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (SIZE_HEIGHT, SIZE_WIDTH))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (SIZE_HEIGHT, SIZE_WIDTH))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (SIZE_HEIGHT, SIZE_WIDTH))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (SIZE_HEIGHT, SIZE_WIDTH))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

train_imgs = glob("Z:/Unreal_Datasets/unrealmeadow/Images_Categorical/*.jpg")
TRAIN_LENGTH = len(train_imgs)

BATCH_SIZE = 4
BUFFER_SIZE = BATCH_SIZE * 10
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE // 10

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

train

##########################
#Visualizing the Loaded Dataset
##########################

def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

for image, mask in train.take(4):
    sample_image, sample_mask = image, mask
display_sample([sample_image, sample_mask])

##########################
#Developing the Model
##########################


with distribute_strategy.scope():
    OUTPUT_CHANNELS = N_CLASSES
    
    base_model = tf.keras.applications.MobileNetV2(input_shape=[SIZE_HEIGHT, SIZE_WIDTH, 3], include_top=False)
    
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]
    
    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    
    down_stack.trainable = True
    
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]



def unet_model(output_channels):
    with distribute_strategy.scope():
        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            output_channels, 3, strides=2,
            padding='same', activation='softmax')  #64x64 -> 128x128
    
        inputs = tf.keras.layers.Input(shape=[SIZE_HEIGHT, SIZE_WIDTH, 3])
        x = inputs
    
        # Downsampling through the model
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])
    
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])
    
        x = last(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=x) 
        return model

with distribute_strategy.scope():
    model = unet_model(OUTPUT_CHANNELS)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """Return a filter mask with the top 1 predictions
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [SIZE, SIZE, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [SIZE, SIZE, 1] mask with top 1 predictions
        for each pixels.
    """
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    """Show a sample prediction.

    Parameters
    ----------
    dataset : [type], optional
        [Input dataset, by default None
    num : int, optional
        Number of sample to show, by default 1
    """
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display_sample([image[0], mask[0], create_mask(pred_mask)])
    else:
        # The model is expecting a tensor of the size
        # [BATCH_SIZE, SIZE, SIZE, 3] for the images
        # and
        # [BATCH_SIZE, SIZE, SIZE, 1] for the annotations
        # -> sample_image is [SIZE, SIZE, 1]
        # -> sample_image[tf.newaxis, ...] is [BATCH_SIZE, SIZE, SIZE, 1]
        display_sample([sample_image, sample_mask,
                        create_mask(model.predict(sample_image[tf.newaxis, ...]))])

#show_predictions()

##########################
#Training
##########################

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        #show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 20
VAL_SUBSPLITS = 2
VALIDATION_STEPS = 2000//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, 
                          epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])

##########################
#Results
##########################
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()