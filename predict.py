import tensorflow as tf
import numpy as np
from configuration import CONFIG
from dataset import ImgSegDataSet
import model
import model_helper

##########################
#Training
##########################

#dataset_data = CONFIG['DATASET_REAL']
dataset_data = CONFIG['DATASET_REAL_TEST']
#dataset_data = CONFIG['DATASET_SOYA_CORN']
#dataset_data = CONFIG['DATASET_GI_AGRICUTURE']


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

#train_dataset = example_set.get_train_set()
test_dataset = example_set.get_test_set()

TRAIN_LENGTH = example_set.get_train_length()

distribute_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:1")
with distribute_strategy.scope():
    reference_model = model.UnetModel(output_channels = dataset_data['N_CLASSES'],
                                      size_height = dataset_data['SIZE_HEIGHT'],
                                      size_width = dataset_data['SIZE_WIDTH'])
    
    keras_model = reference_model.get_model()
    keras_model.compile(optimizer=tf.keras.optimizers.RMSprop(), 
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=['sparse_categorical_accuracy'])


#Load existing weights
#keras_model.load_weights(filepath='./weights/weight_epoch-50_valloss-0.12152234464883804.h5')
#keras_model.load_weights(filepath='./weights/weight_epoch-25_valloss-0.08790893852710724.h5')
keras_model.load_weights(filepath='./weights/weight_epoch-38_valloss-0.037506891414523125.h5')

number_of_prediction = 100
running_total = 0
for image, mask in test_dataset.take(number_of_prediction):
    pred_mask = keras_model.predict(image)
    error_mask =  np.equal(mask[0], model_helper.create_mask(pred_mask))
    
    prediction_success = 1 - ((error_mask.size - np.sum(error_mask)) / error_mask.size)
    running_total = running_total + prediction_success
    
    model_helper.display_sample([image[0], mask[0], model_helper.create_mask(pred_mask), error_mask * example_set.n_classes],
                                'Sample Prediction, Success = ' + str(prediction_success),
                                dataset_data['N_CLASSES'])
    
print("Average prediction rate: " + str(running_total / number_of_prediction))
