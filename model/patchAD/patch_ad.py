import tensorflow as tf

class PatchAD(tf.keras.Model):
    def __init__(self, input_channels, seq_length, patch_size, hidden_dim=256, num_layers=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_length // patch_size
        self.hidden_dim = hidden_dim
        
        # Positional embedding
        self.pos_embedding = self.add_weight(
            "pos_embedding",
            shape=(1, self.num_patches, hidden_dim),
            initializer="random_normal"
        )
        
        # Value embeddings
        self.value_embed_inter = tf.keras.layers.Dense(hidden_dim)
        self.value_embed_intra = tf.keras.layers.Dense(hidden_dim)
        
        # Encoders
        self.encoders = [PatchMixerLayer(hidden_dim) for _ in range(num_layers)]
        
        # Project heads
        self.proj_head_inter = ProjectHead(hidden_dim)
        self.proj_head_intra = ProjectHead(hidden_dim)
        
        # Reconstruction heads
        self.rec_head_inter = tf.keras.layers.Dense(patch_size)
        self.rec_head_intra = tf.keras.layers.Dense(patch_size)
        
        # Layer weights
        self.alpha = self.add_weight(
            "alpha",
            shape=(num_layers,),
            initializer="ones"
        )
        self.beta = self.add_weight(
            "beta",
            shape=(num_layers,),
            initializer="ones"
        )

    def call(self, x, training=False):
        # Multi-scale patching
        x_patched = self.make_patches(x)
        
        # Value embeddings
        x_inter = self.value_embed_inter(x_patched)
        x_intra = self.value_embed_intra(tf.transpose(x_patched, perm=[0, 1, 3, 2]))
        
        # Add positional embeddings
        x_inter = x_inter + self.pos_embedding
        x_intra = x_intra + self.pos_embedding
        
        # Store intermediate representations
        inter_reps = []
        intra_reps = []
        
        # Pass through encoder layers
        for encoder in self.encoders:
            x_inter, x_intra = encoder(x_inter, x_intra, training=training)
            inter_reps.append(x_inter)
            intra_reps.append(x_intra)
        
        # Apply reweighting
        alpha = tf.nn.softmax(self.alpha)
        beta = tf.nn.softmax(self.beta)
        
        x_inter = tf.add_n([a * rep for a, rep in zip(alpha, inter_reps)])
        x_intra = tf.add_n([b * rep for b, rep in zip(beta, intra_reps)])
        
        # Project representations
        proj_inter = self.proj_head_inter(x_inter)
        proj_intra = self.proj_head_intra(x_intra)
        
        # Reconstruction
        rec_inter = self.rec_head_inter(x_inter)
        rec_intra = self.rec_head_intra(tf.transpose(x_intra, perm=[0, 1, 3, 2]))
        
        return {
            'inter': x_inter,
            'intra': x_intra,
            'proj_inter': proj_inter,
            'proj_intra': proj_intra,
            'rec_inter': rec_inter,
            'rec_intra': rec_intra
        }
    
    def make_patches(self, x):
        patch_size = self.patch_size
        stride = patch_size  # Assuming stride is equal to patch_size
        patches = tf.image.extract_patches(
            images=tf.expand_dims(x, axis=-1),
            sizes=[1, patch_size, 1, 1],
            strides=[1, stride, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(patches, (tf.shape(x)[0], -1, patch_size, tf.shape(x)[-1]))
        return patches
