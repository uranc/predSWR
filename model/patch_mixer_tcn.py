import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Conv1D, Input, Concatenate, GlobalAveragePooling1D
from tcn import TCN

class PatchTokenizer(Layer):
    def __init__(self, patch_size, stride, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.proj = Conv1D(embed_dim, kernel_size=patch_size, strides=stride)
        
    def call(self, x):
        # Project patches to embedding dimension
        return self.proj(x)

class ChannelMixer(Layer):
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        self.norm = LayerNormalization()
        self.fc1 = Dense(dim * expansion_factor)
        self.gelu = tf.keras.activations.gelu
        self.fc2 = Dense(dim)
        
    def call(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class TemporalMixer(Layer):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNormalization()
        # Temporal mixing using 1D convolutions
        self.temporal_mix = Conv1D(dim, kernel_size=3, padding='same', groups=dim)
        
    def call(self, x):
        x = self.norm(x)
        return self.temporal_mix(x)

class MultiScalePatchMixer(Layer):
    def __init__(self, dim, num_scales=3):
        super().__init__()
        self.scales = [2**i for i in range(num_scales)]
        self.convs = [Conv1D(dim//2, kernel_size=scale, padding='same') 
                     for scale in self.scales]
        self.proj = Dense(dim)
        
    def call(self, x):
        # Multi-scale feature extraction
        features = [conv(x) for conv in self.convs]
        # Concatenate features from different scales
        x = Concatenate(axis=-1)(features)
        return self.proj(x)

class TCNMixerBlock(Layer):
    def __init__(self, dim, tcn_params):
        super().__init__()
        self.channel_mixer = ChannelMixer(dim)
        self.temporal_mixer = TemporalMixer(dim)
        self.multiscale_mixer = MultiScalePatchMixer(dim)
        self.tcn = TCN(**tcn_params)
        self.norm = LayerNormalization()
        
    def call(self, x):
        # Channel mixing
        x1 = x + self.channel_mixer(x)
        # Temporal mixing
        x2 = x1 + self.temporal_mixer(x1)
        # Multi-scale mixing
        x3 = x2 + self.multiscale_mixer(x2)
        # TCN processing
        x4 = self.norm(x3)
        x4 = self.tcn(x4)
        return x3 + x4  # Residual connection

class MemoryQueue:
    def __init__(self, max_size, feature_dim):
        self.max_size = max_size
        self.queue = tf.Variable(
            tf.zeros([max_size, feature_dim]), 
            trainable=False
        )
        self.ptr = tf.Variable(0, trainable=False)
        
    @tf.function
    def enqueue(self, features):
        # Enqueue new features, overwriting oldest if full
        batch_size = tf.shape(features)[0]
        ptr = self.ptr
        
        # Handle wraparound
        indices = tf.math.mod(tf.range(ptr, ptr + batch_size), self.max_size)
        self.queue.scatter_nd_update(
            tf.expand_dims(indices, 1),
            features
        )
        self.ptr.assign(tf.math.mod(ptr + batch_size, self.max_size))
        
    def get_memories(self):
        return self.queue

class HistoryAwareTCNMixerBlock(TCNMixerBlock):
    def __init__(self, dim, tcn_params, memory_size=1024):
        super().__init__(dim, tcn_params)
        self.memory = MemoryQueue(memory_size, dim)
        self.memory_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=8, 
            key_dim=dim
        )
        
    def call(self, x, training=True):
        # Regular mixing operations
        x = super().call(x)
        
        # Memory operations
        if training:
            # Store features in memory
            self.memory.enqueue(x[:,-1,:])  # store last timestep features
            
        # Query memory for relevant patterns
        memories = self.memory.get_memories()
        mem_context = self.memory_attn(x, memories, memories)
        
        return x + mem_context

class PatchMixerTCN(tf.keras.Model):
    def __init__(self, 
                 patch_size=32,
                 stride=16,
                 embed_dim=128,
                 num_blocks=4,
                 memory_size=1024,
                 tcn_params=None):
        super().__init__()
        
        if tcn_params is None:
            tcn_params = {
                'nb_filters': embed_dim,
                'kernel_size': 3,
                'dilations': [1, 2, 4, 8],
                'return_sequences': True
            }
        
        self.patch_tokenizer = PatchTokenizer(patch_size, stride, embed_dim)
        
        # Replace regular blocks with history-aware blocks
        self.blocks = [HistoryAwareTCNMixerBlock(
            embed_dim, 
            tcn_params,
            memory_size=memory_size
        ) for _ in range(num_blocks)]
                      
        # Prediction head
        self.pool = GlobalAveragePooling1D()
        self.fc = Dense(1)
        
    def call(self, inputs, training=True):
        x = self.patch_tokenizer(inputs)
        
        # Process through blocks with memory
        for block in self.blocks:
            x = block(x, training=training)
            
        x = self.pool(x)
        return self.fc(x)

def build_model(input_shape, patch_size=32, stride=16):
    inputs = Input(shape=input_shape)
    
    # Model parameters
    params = {
        'patch_size': patch_size,
        'stride': stride,
        'embed_dim': 128,
        'num_blocks': 4,
        'memory_size': 1024,
        'tcn_params': {
            'nb_filters': 128,
            'kernel_size': 3,
            'dilations': [1, 2, 4, 8],
            'return_sequences': True
        }
    }
    
    model = PatchMixerTCN(**params)
    return model(inputs)

# Example usage:
# model = build_model(input_shape=(None, 8))
