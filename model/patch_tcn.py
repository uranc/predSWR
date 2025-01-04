import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, Dense, Input, TimeDistributed
from tcn import TCN

class PatchTCN(Layer):
    def __init__(self, patch_size, stride, tcn_params):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.tcn_params = tcn_params
        
    def build(self, input_shape):
        # Create TCN layer
        self.tcn = TCN(**self.tcn_params)
        
    def create_patches(self, inputs):
        # inputs: [batch, time, channels]
        patches = tf.image.extract_patches(
            images=tf.expand_dims(inputs, axis=-1),
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.stride, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        # Reshape to [batch, num_patches, patch_size, channels]
        patches = tf.reshape(patches, 
            (tf.shape(inputs)[0], -1, self.patch_size, tf.shape(inputs)[-1]))
        return patches

    def call(self, inputs):
        # Method 1: Using TimeDistributed (processes each patch independently)
        patches = self.create_patches(inputs)
        # Reshape for TimeDistributed: [batch, num_patches, patch_size, channels]
        batch_size = tf.shape(patches)[0]
        num_patches = tf.shape(patches)[1]
        patch_size = self.patch_size
        channels = tf.shape(patches)[-1]
        
        # Reshape patches to [batch * num_patches, patch_size, channels]
        reshaped_patches = tf.reshape(patches, [-1, patch_size, channels])
        
        # Apply TCN
        processed = self.tcn(reshaped_patches)
        
        # Reshape back to [batch, num_patches, patch_size, channels]
        output = tf.reshape(processed, [batch_size, num_patches, -1, self.tcn_params['nb_filters']])
        return output

class DirectPatchTCN(Layer):
    def __init__(self, patch_size, stride, tcn_params):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.tcn_params = tcn_params
        
    def build(self, input_shape):
        # Method 2: Direct convolution approach
        # Adjust TCN to handle the entire sequence at once
        self.tcn = TCN(
            nb_filters=self.tcn_params['nb_filters'],
            kernel_size=self.patch_size,  # Kernel size matches patch size
            dilations=self.tcn_params['dilations'],
            padding='causal',
            return_sequences=True
        )
        
    def call(self, inputs):
        # Process entire sequence directly
        return self.tcn(inputs)

def build_patch_tcn_model(input_shape, patch_size=50, stride=25):
    inputs = Input(shape=input_shape)
    
    # Define TCN parameters
    tcn_params = {
        'nb_filters': 32,
        'kernel_size': 3,
        'dilations': [1, 2, 4],
        'activation': 'relu',
        'return_sequences': True
    }
    
    # Choose one of the implementations:
    # 1. Patch-based with independent processing
    patch_tcn = PatchTCN(patch_size, stride, tcn_params)(inputs)
    
    # 2. Direct convolution approach
    # direct_tcn = DirectPatchTCN(patch_size, stride, tcn_params)(inputs)
    
    # Add final layers as needed
    outputs = Dense(input_shape[-1])(patch_tcn)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Example usage:
# model = build_patch_tcn_model(input_shape=(None, 8))
