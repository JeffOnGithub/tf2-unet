import tensorflow as tf
import matplotlib.pyplot as plt
from configuration import CONFIG

##########################
#Model Utilities
##########################

    
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

def show_predictions(model, dataset, num, main_title, n_classes):
    """Show a sample prediction.

    Parameters
    ----------
    dataset : [type], optional
        [Input dataset, by default None
    num : int, optional
        Number of sample to show, by default 1
    """
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        display_sample([image[0], mask[0], create_mask(pred_mask)], main_title, n_classes)

def display_sample(display_list, main_title, n_classes):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    
    plt.figure(figsize=(18, 9))

    title = ['Input Image', 'True Mask', 'Predicted Mask', 'Prediction Error']

    for i in range(len(display_list)):
        
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        if i == 0:
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else: #Special treatement on label maps to insure the same scale
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i], scale=False),
                       #cmap='cividis',
                       vmin=0,
                       vmax=n_classes - 1)
        plt.axis('off')
    
    plt.suptitle(main_title)
    plt.show()