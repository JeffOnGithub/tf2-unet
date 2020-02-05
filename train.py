import tensorflow as tf
import matplotlib.pyplot as plt
from configuration import CONFIG
from dataset import ImgSegDataSet
import model
import model_helper

##########################
#Training
##########################

#dataset_data = CONFIG['DATASET_VIRTUAL']
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

train_dataset = example_set.get_train_set()
test_dataset = example_set.get_test_set()

TRAIN_LENGTH = example_set.get_train_length()


EPOCHS = CONFIG['TRAINING']['EPOCHS']
VAL_SUBSPLITS = CONFIG['TRAINING']['VAL_SUBSPLITS']
BATCH_SIZE = dataset_data['BATCH_SIZE']

STEPS_PER_EPOCH = 100
#STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE // dataset_data['BATCH_FACTOR']
VALIDATION_STEPS = STEPS_PER_EPOCH // 50

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        model_helper.show_predictions(model=keras_model, 
                                      dataset=test_dataset, 
                                      num=2, 
                                      main_title='Sample Prediction after epoch {}'.format(epoch+1),
                                      n_classes=dataset_data['N_CLASSES'])

saveWeights = tf.keras.callbacks.ModelCheckpoint(filepath='./weights/weight_epoch-{epoch}_valloss-{val_loss}.h5',
                                                 #save_best_only=True,
                                                 #monitor='val_loss',
                                                 verbose=1)
    
distribute_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:1")
#distribute_strategy = tf.distribute.MirroredStrategy() #No NCCL on Windows!
#distribute_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
#distribute_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice()) #Too slow

with distribute_strategy.scope():
    reference_model = model.UnetModel(output_channels = dataset_data['N_CLASSES'],
                                      size_height = dataset_data['SIZE_HEIGHT'],
                                      size_width = dataset_data['SIZE_WIDTH'])
    
    keras_model = reference_model.get_model()
    keras_model.compile(optimizer=tf.keras.optimizers.RMSprop(), 
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=['sparse_categorical_accuracy'])


#Load existing weights
keras_model.load_weights(filepath='./weights/weights-virtual-training.h5')

#Unfrozen training
model_history = keras_model.fit(train_dataset, 
                                epochs=EPOCHS,
                                steps_per_epoch=STEPS_PER_EPOCH,
                                validation_steps=VALIDATION_STEPS,
                                validation_data=test_dataset,
                                callbacks=[DisplayCallback(), saveWeights])

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
plt.ylim([0, 2])
plt.legend()
plt.show()