import tensorflow as tf
import numpy as np
from glob import glob
from configuration import CONFIG
import model_helper


class ImgSegDataSet:
    
    def __init__(self,
                 native_height,
                 native_width,
                 size_height,
                 size_width,
                 n_classes,
                 batch_size,
                 batch_factor,
                 train_imgs_folder,
                 test_imgs_folder,
                 img_ext,
                 lbl_ext,
                 img_folder_path,
                 lbl_folder_path):
        
        #Direct
        self.native_height = native_height
        self.native_width = native_width
        self.size_height = size_height
        self.size_width = size_width
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.batch_factor = batch_factor
        self.train_imgs_folder = train_imgs_folder
        self.test_imgs_folder = test_imgs_folder
        self.img_ext = img_ext
        self.lbl_ext = lbl_ext
        self.img_folder_path = img_folder_path
        self.lbl_folder_path = lbl_folder_path
        
        #Calculated
        self.buffer_size = self.batch_size * 20
        
        train_imgs = tf.data.Dataset.list_files(train_imgs_folder)
        test_imgs = tf.data.Dataset.list_files(test_imgs_folder)
        
        train_set = train_imgs.map(self.parse_image)
        
        test_set = test_imgs.map(self.parse_image)
        
        self.dataset = {"train": train_set, "test": test_set}
    
    ##########################
    #Loading the Raw Data
    ##########################
    def parse_image(self, img_path: str) -> dict:
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
        mask_path = tf.strings.regex_replace(img_path, self.img_folder_path, self.lbl_folder_path)
        mask_path = tf.strings.regex_replace(mask_path, self.img_ext, self.lbl_ext)
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
    
    ##########################
    #Data Loader with Normalization and Augmentations
    ##########################
    def normalize(self, input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
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
    
    def crop_and_resize(self, input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
        
        #Get a random size for the crop region
        crop_dimension = tf.random.uniform((), 
                                           minval=self.size_height, 
                                           maxval=self.native_height)
        
        #Get random top left corner that fits with the crop region size
        crop_y = tf.random.uniform((), 
                                   minval=0, 
                                   maxval=self.native_height - crop_dimension)
        crop_x = tf.random.uniform((), 
                                   minval=0, 
                                   maxval=self.native_height - crop_dimension)
        
        crop_y = int(crop_y)
        crop_x = int(crop_x)
        crop_dimension = int(crop_dimension)
        
        #Crop
        input_image = tf.image.crop_to_bounding_box(input_image, crop_y, crop_x, crop_dimension, crop_dimension)
        input_mask = tf.image.crop_to_bounding_box(input_mask, crop_y, crop_x, crop_dimension, crop_dimension)
        
        #Resize
        input_image = tf.image.resize(input_image, (self.size_height, self.size_width))
        input_mask = tf.image.resize(input_mask, (self.size_height, self.size_width))
        
        return input_image, input_mask
    
    @tf.function
    def load_image_train(self, datapoint: dict) -> tuple:
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
        input_image, input_mask = self.crop_and_resize(datapoint['image'], datapoint['segmentation_mask'])
        
        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)
    
        input_image, input_mask = self.normalize(input_image, input_mask)
    
        return input_image, input_mask
    
    @tf.function
    def load_image_test(self, datapoint: dict) -> tuple:
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
        input_image, input_mask = self.crop_and_resize(datapoint['image'], datapoint['segmentation_mask'])
    
        input_image, input_mask = self.normalize(input_image, input_mask)
    
        return input_image, input_mask
    
    def get_train_length(self):
        imgs = glob(self.train_imgs_folder)
        return len(imgs)

    def get_test_length(self):
        imgs = glob(self.test_imgs_folder)
        return len(imgs)

    def get_train_set(self):
        train = self.dataset['train'].map(self.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train.shuffle(self.buffer_size).batch(self.batch_size).repeat()
        
        #train_dataset = train.shuffle(self.buffer_size).batch(self.batch_size).cache().repeat()
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        return train_dataset

    def get_test_set(self):
        test = self.dataset['test'].map(self.load_image_test)
        test_dataset = test.shuffle(self.buffer_size).batch(self.batch_size).repeat()
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return test_dataset
    
    
if __name__ == '__main__':
    ##########################
    #Test the dataset
    ##########################
    dataset_data = CONFIG['DATASET_REAL']
    example_set = ImgSegDataSet(native_height = dataset_data['NATIVE_HEIGHT'],
                                native_width = dataset_data['NATIVE_WIDTH'],
                                size_height = dataset_data['SIZE_HEIGHT'],
                                size_width = dataset_data['SIZE_WIDTH'],
                                n_classes = dataset_data['N_CLASSES'],
                                batch_size = dataset_data['BATCH_SIZE'],
                                batch_factor = dataset_data['BATCH_FACTOR'],
                                train_imgs_folder = dataset_data['train_imgs_folder'],
                                test_imgs_folder = dataset_data['test_imgs_folder'],
                                img_ext = dataset_data['img_ext'],
                                lbl_ext = dataset_data['lbl_ext'],
                                img_folder_path = dataset_data['img_folder_path'],
                                lbl_folder_path = dataset_data['lbl_folder_path'])
    
    for image, mask in example_set.get_train_set().take(1):
        sample_image, sample_mask = image, mask
    
    for i in range(len(sample_image)):
        model_helper.display_sample([sample_image[i], sample_mask[i]], 'Train sample example ' + str(i), dataset_data['N_CLASSES'])
        
        
    for image, mask in example_set.get_test_set().take(1):
        sample_image, sample_mask = image, mask
    
    for i in range(len(sample_image)):
        model_helper.display_sample([sample_image[i], sample_mask[i]], 'Test sample example ' + str(i), dataset_data['N_CLASSES'])