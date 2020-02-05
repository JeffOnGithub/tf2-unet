import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from configuration import CONFIG

##########################
#Developing the Model
##########################
class UnetModel:
    
    def __init__(self, output_channels, size_height, size_width):
        self.output_channels = output_channels
        self.size_height = size_height
        self.size_width = size_width

        base_model = tf.keras.applications.MobileNetV2(input_shape=[self.size_height, self.size_width, 3], include_top=False)
        
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
        self.down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
        
        self.down_stack.trainable = True
        
        self.up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]
            
    def get_model(self):
        return self.unet_model(self.output_channels)
    
    def unet_model(self, output_channels):
        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            output_channels, 3, strides=2,
            padding='same', activation='softmax')  #64x64 -> 128x128
    
        inputs = tf.keras.layers.Input(shape=[self.size_height, self.size_width, 3])
        x = inputs
    
        # Downsampling through the model
        skips = self.down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])
    
        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])
    
        x = last(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=x) 
        return model


if __name__ == '__main__':
##########################
#Test the Model
##########################
    model = UnetModel(output_channels = CONFIG['DATASET']['N_CLASSES'],
                      size_height = CONFIG['DATASET']['SIZE_HEIGHT'],
                      size_width = CONFIG['DATASET']['SIZE_WIDTH']).get_model()
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])