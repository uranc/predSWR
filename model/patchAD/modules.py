import tensorflow as tf

class MLPMixer(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(dim * 4)
        self.fc2 = tf.keras.layers.Dense(dim)
        self.norm = tf.keras.layers.LayerNormalization()
        
    def call(self, x, training=False):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = tf.nn.gelu(x)
        x = self.fc2(x)
        return x + residual

class PatchMixerLayer(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.channel_mixer = MLPMixer(dim)
        self.inter_mixer = MLPMixer(dim)
        self.intra_mixer = MLPMixer(dim)
        self.mixrep_mixer = MLPMixer(dim)
        
    def call(self, x_inter, x_intra, training=False):
        # Channel mixing
        x_inter = self.channel_mixer(x_inter, training=training)
        x_intra = self.channel_mixer(x_intra, training=training)
        
        # Inter-patch mixing
        x_inter = self.inter_mixer(x_inter, training=training)
        
        # Intra-patch mixing
        x_intra = self.intra_mixer(x_intra, training=training)
        
        # Mix representations
        x_inter = self.mixrep_mixer(x_inter, training=training)
        x_intra = self.mixrep_mixer(x_intra, training=training)
        
        return x_inter, x_intra

class ProjectHead(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(dim)
        self.fc2 = tf.keras.layers.Dense(dim)
        
    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
