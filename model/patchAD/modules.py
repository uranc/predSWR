import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Layer, Activation
from tensorflow_addons.activations import gelu # Assuming gelu is used
from tensorflow.keras.initializers import GlorotUniform


def get_activation_tf(activ_str):
    if activ_str == "gelu":
        return gelu
    elif activ_str == "relu":
        return tf.nn.relu
    else:
        return tf.identity

def get_norm_tf(norm_str, features_dim, name=None):
    if norm_str == "ln":
        return LayerNormalization(epsilon=1e-5, name=name)
    else:
        return lambda x, training=None: x  # Return identity function instead of tf.identity


class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, dim, in_features, hid_features, out_features, 
                 activ="gelu", drop=0.0, jump_conn="proj", norm="ln", 
                 name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim = dim  # Dimension to apply MLP over
        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = out_features
        self.activ_name = activ
        self.drop_rate = drop
        self.jump_conn_type = jump_conn
        self.norm_type = norm
        
        # Initialize all layers with proper initializers
        kernel_init = tf.keras.initializers.GlorotUniform()
        
        # Core MLP layers
        self.norm1 = get_norm_tf(norm, in_features, name=f"{name}_norm1" if name else None)
        self.fc1 = Dense(hid_features, 
                        kernel_initializer=kernel_init,
                        use_bias=True,
                        name=f"{name}_fc1" if name else None)
        
        self.norm2 = get_norm_tf(norm, hid_features, name=f"{name}_norm2" if name else None)
        self.fc2 = Dense(out_features,
                        kernel_initializer=kernel_init,
                        use_bias=True,
                        name=f"{name}_fc2" if name else None)
        
        # Activation and dropout
        # Use Activation layer for serialization compatibility
        self.activation_layer = Activation(get_activation_tf(activ), name=f"{name}_activ" if name else None)
        # self.dropout1 = Dropout(drop)
        # self.dropout2 = Dropout(drop) # Added second dropout after fc2 as in call
        
        # Skip connection handling
        self.skip_proj = None # Initialize skip_proj
        if jump_conn == "proj":
            if in_features != out_features:
                self.skip_proj = Dense(out_features,
                                     kernel_initializer=kernel_init,
                                     use_bias=False,
                                     name=f"{name}_skip" if name else None)
            # else: # Implicitly handle identity skip if not proj needed
            #     self.skip_proj = lambda x: x # Lambdas are not easily serializable
        elif jump_conn == "trunc":
            if in_features != out_features:
                raise ValueError("jump_conn='trunc' requires in_features == out_features")
            # self.skip_proj = lambda x: x # Lambdas are not easily serializable
        # else: # No skip connection or identity
            # self.skip_proj = lambda x: x # Lambdas are not easily serializable

    def call(self, x, training=False):
        # Save original input for skip connection
        identity = x
        
        # Handle dimension permutation if needed
        perm = None
        if self.dim != -1 and self.dim < len(x.shape) -1 : # Check if dim is valid and not the last one
            perm = list(range(len(x.shape)))
            perm[self.dim], perm[-1] = perm[-1], perm[self.dim]
            x = tf.transpose(x, perm)
        
        # Main forward pass
        x = self.norm1(x, training=training)
        x = self.fc1(x)
        x = self.activation_layer(x) # Use Activation layer
        # x = self.dropout1(x, training=training)
        x = self.norm2(x, training=training)
        x = self.fc2(x)
        # x = self.dropout2(x, training=training) # Use second dropout
        
        # Apply skip connection
        if self.jump_conn_type == "proj" and self.skip_proj is not None:
             if perm: # Transpose identity if needed before projection
                 identity = tf.transpose(identity, perm)
             identity = self.skip_proj(identity)
        elif self.jump_conn_type == "proj" and self.skip_proj is None: # Identity skip for proj when dims match
             if perm:
                 identity = tf.transpose(identity, perm)
        elif self.jump_conn_type == "trunc": # Identity skip for trunc
             if perm:
                 identity = tf.transpose(identity, perm)
        else: # No skip or identity if jump_conn is not proj or trunc
             identity = 0 # Or handle as needed, maybe identity if dims match? Assuming additive skip

        x = x + identity
        
        # Restore original dimension order if needed
        if perm:
            x = tf.transpose(x, perm)
            
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "in_features": self.in_features,
            "hid_features": self.hid_features,
            "out_features": self.out_features,
            "activ": self.activ_name,
            "drop": self.drop_rate,
            "jump_conn": self.jump_conn_type,
            "norm": self.norm_type,
        })
        # Manually serialize skip_proj if it's a Dense layer
        if isinstance(self.skip_proj, Dense):
             config['skip_proj_config'] = serialize_keras_object(self.skip_proj)
        else:
             config['skip_proj_config'] = None # Or handle other cases if necessary
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize skip_proj if present
        skip_proj_config = config.pop('skip_proj_config', None)
        instance = cls(**config)
        if skip_proj_config:
            instance.skip_proj = deserialize_keras_object(skip_proj_config)
        return instance
    
    
##############################
# PatchMixerLayer            #
##############################

class PatchMixerLayer(tf.keras.layers.Layer):
    def __init__(self, in_len, hid_len, in_chn, hid_chn, out_chn,
                 patch_size, hid_pch, d_model, norm="ln", activ="gelu", 
                 drop=0.0, jump_conn="proj", name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Validate inputs
        if in_len % patch_size != 0:
            raise ValueError(f"in_len ({in_len}) must be divisible by patch_size ({patch_size})")
        
        # Store parameters needed for get_config
        self.in_len = in_len
        self.hid_len = hid_len
        self.in_chn = in_chn
        self.hid_chn = hid_chn
        self.out_chn = out_chn
        self.patch_size = patch_size
        self.hid_pch = hid_pch
        self.d_model = d_model
        self.norm_type = norm
        self.activ_name = activ
        self.drop_rate = drop
        self.jump_conn_type = jump_conn
        
        self.num_patches = in_len // patch_size
        
        # Initialize kernel properly for all layers
        kernel_init = tf.keras.initializers.GlorotUniform()
        
        # Channel mixing layers (dim=1)
        self.ch_mixing1 = MLPBlock(
            dim=1, 
            in_features=in_chn,
            hid_features=hid_chn,
            out_features=out_chn,
            activ=activ,
            drop=drop,
            jump_conn=jump_conn,
            norm=norm,
            name=f"{name}_ch_mix1" if name else None
        )
        
        # Patch number mixing (dim=2)
        self.patch_num_mix = MLPBlock(
            dim=2,
            in_features=self.num_patches,
            hid_features=hid_len,
            out_features=self.num_patches,
            activ=activ,
            drop=drop,
            jump_conn=jump_conn,
            norm=norm,
            name=f"{name}_num_mix" if name else None
        )
        
        # Patch size mixing (dim=2)
        self.patch_size_mix = MLPBlock(
            dim=2,
            in_features=patch_size,
            hid_features=hid_pch,
            out_features=patch_size,
            activ=activ,
            drop=drop,
            jump_conn=jump_conn,
            norm=norm,
            name=f"{name}_size_mix" if name else None
        )
        
        # Feature dimension mixing (dim=3)
        self.d_mixing1 = MLPBlock(
            dim=3,
            in_features=d_model,
            hid_features=d_model, # Assuming hidden dim is same as input/output for d_mixing
            out_features=d_model,
            activ=activ,
            drop=drop,
            jump_conn=jump_conn,
            norm=norm,
            name=f"{name}_d_mix1" if name else None
        )

        # Layer normalization layers
        self.norm1_num = get_norm_tf(norm, in_chn, name=f"{name}_norm1_num")
        self.norm1_size = get_norm_tf(norm, in_chn, name=f"{name}_norm1_size")
        self.norm2_num = get_norm_tf(norm, out_chn, name=f"{name}_norm2_num")
        self.norm2_size = get_norm_tf(norm, out_chn, name=f"{name}_norm2_size")
        self.norm3_num = get_norm_tf(norm, out_chn, name=f"{name}_norm3_num")
        self.norm3_size = get_norm_tf(norm, out_chn, name=f"{name}_norm3_size")

    @tf.function
    def call(self, x_patch_num, x_patch_size, training=False):
        # ... (call method remains the same) ...
        # Shape validation
        tf.debugging.assert_rank(x_patch_num, 4, "x_patch_num must have rank 4")
        tf.debugging.assert_rank(x_patch_size, 4, "x_patch_size must have rank 4")
        
        # Number path
        x_num = self.norm1_num(x_patch_num, training=training)
        x_num = self.ch_mixing1(x_num, training=training)
        x_num = self.norm2_num(x_num, training=training)
        x_num = self.patch_num_mix(x_num, training=training)
        x_num = self.norm3_num(x_num, training=training)
        out_num = self.d_mixing1(x_num, training=training)

        # Size path
        x_size = self.norm1_size(x_patch_size, training=training)
        x_size = self.ch_mixing1(x_size, training=training) # Re-use ch_mixing1
        x_size = self.norm2_size(x_size, training=training)
        x_size = self.patch_size_mix(x_size, training=training)
        x_size = self.norm3_size(x_size, training=training)
        out_size = self.d_mixing1(x_size, training=training) # Re-use d_mixing1

        return out_num, out_size

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_len": self.in_len,
            "hid_len": self.hid_len,
            "in_chn": self.in_chn,
            "hid_chn": self.hid_chn,
            "out_chn": self.out_chn,
            "patch_size": self.patch_size,
            "hid_pch": self.hid_pch,
            "d_model": self.d_model,
            "norm": self.norm_type,
            "activ": self.activ_name,
            "drop": self.drop_rate,
            "jump_conn": self.jump_conn_type,
        })
        # Note: Sub-layers (MLPBlocks, norms) will be reconstructed based on these params
        return config

class EncoderEnsemble(tf.keras.layers.Layer):
    def __init__(self, encoder_layers, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        # Ensure encoder_layers is a list of Layer instances for serialization
        if not isinstance(encoder_layers, list) or not all(isinstance(layer, Layer) for layer in encoder_layers):
             raise TypeError("encoder_layers must be a list of Keras Layer instances.")
        self.encoder_layers = encoder_layers # List of PatchMixerLayer instances
        num_layers = len(encoder_layers)
        
        # Initialize ensemble weights with proper initializer
        weight_init = tf.keras.initializers.Constant(1.0)  # Initialize to ones
        
        # Learnable weights for ensemble block
        self.ensemble_weights_num = self.add_weight(
            name='ensemble_weights_num',
            shape=(num_layers,),
            initializer=weight_init,
            trainable=True,
            dtype=tf.float32
        )
        self.ensemble_weights_size = self.add_weight(
            name='ensemble_weights_size',
            shape=(num_layers,),
            initializer=weight_init,
            trainable=True,
            dtype=tf.float32
        )

    @tf.function
    def call(self, x_patch_num, x_patch_size, training=False):
        # ... (call method remains the same) ...
        # Input validation
        tf.debugging.assert_rank(x_patch_num, 4, "x_patch_num must have rank 4")
        tf.debugging.assert_rank(x_patch_size, 4, "x_patch_size must have rank 4")
        
        # Lists to store intermediate results
        all_logits_num = []
        all_logits_size = []
        dist_num_list = []
        dist_size_list = []

        # Forward pass through encoder layers
        current_num = x_patch_num
        current_size = x_patch_size

        for i, layer in enumerate(self.encoder_layers):
            # Get outputs and logits from the layer
            out_num, out_size = layer(
                current_num, current_size, 
                training=training
            )

            # Store raw logits
            all_logits_num.append(out_num)
            all_logits_size.append(out_size)

            # Compute distributions with numerically stable softmax
            dist_num = tf.nn.softmax(out_num, axis=-1)  # Over feature dimension D
            dist_size = tf.nn.softmax(out_size, axis=-1)  # Over feature dimension D

            # Average over channel dimension (axis=1)
            reduced_dist_num = tf.reduce_mean(dist_num, axis=1)   # [B, N, D]
            reduced_dist_size = tf.reduce_mean(dist_size, axis=1) # [B, P, D]

            # Store reduced distributions
            dist_num_list.append(reduced_dist_num)
            dist_size_list.append(reduced_dist_size)

            # Update for next layer
            current_num = out_num
            current_size = out_size

        # Ensemble Block
        if dist_num_list:  # Check if we have any distributions
            # Stack distributions: [B, N, D, L] and [B, P, D, L]
            dist_num_stack = tf.stack(dist_num_list, axis=-1)
            dist_size_stack = tf.stack(dist_size_list, axis=-1)

            # Apply numerically stable softmax to ensemble weights
            weights_num = tf.nn.softmax(self.ensemble_weights_num)
            weights_size = tf.nn.softmax(self.ensemble_weights_size)

            # Reshape weights for broadcasting
            weights_num_b = tf.reshape(weights_num, [1, 1, 1, -1])  # [1, 1, 1, L]
            weights_size_b = tf.reshape(weights_size, [1, 1, 1, -1])  # [1, 1, 1, L]

            # Apply weights and sum across layers
            weighted_dist_num = dist_num_stack * weights_num_b
            weighted_dist_size = dist_size_stack * weights_size_b

            # Final weighted distributions (unstack back into a list)
            final_dist_num = tf.unstack(weighted_dist_num, axis=-1)
            final_dist_size = tf.unstack(weighted_dist_size, axis=-1)
        else:
            final_dist_num = []
            final_dist_size = []

        return final_dist_num, final_dist_size, all_logits_num, all_logits_size

    def get_config(self):
        config = super().get_config()
        # Serialize the list of encoder layers
        config.update({
            "encoder_layers": [serialize_keras_object(layer) for layer in self.encoder_layers]
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize the list of encoder layers
        encoder_layers_config = config.pop("encoder_layers")
        encoder_layers = [deserialize_keras_object(layer_config) for layer_config in encoder_layers_config]
        # Pass the deserialized layers list to the constructor
        return cls(encoder_layers=encoder_layers, **config)

class ProjectHead(tf.keras.layers.Layer):
    def __init__(self, dim, hidden_dim=None, dropout=0.1, activation='gelu', name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        
        # If hidden_dim not specified, use 2x input dim
        if hidden_dim is None:
            hidden_dim = dim * 2
            
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.activation_name = activation

        # Proper initialization for all layers
        kernel_init = tf.keras.initializers.GlorotUniform()
        
        # Define individual layers
        self.fc1 = Dense(hidden_dim, 
                         kernel_initializer=kernel_init,
                         use_bias=True,
                         name=f"{name}_fc1" if name else None)
        self.norm1 = LayerNormalization(epsilon=1e-5, 
                                        name=f"{name}_norm1" if name else None)
        # Use Activation layer to wrap the activation function
        self.activation_layer = Activation(get_activation_tf(activation), 
                                           name=f"{name}_activ" if name else None)
        self.dropout_layer = Dropout(dropout)
        self.fc2 = Dense(dim,
                         kernel_initializer=kernel_init,
                         use_bias=True,
                         name=f"{name}_fc2" if name else None)
        self.norm2 = LayerNormalization(epsilon=1e-5,
                                        name=f"{name}_norm2" if name else None)

    @tf.function
    def call(self, x, training=False):
        """
        Projects input features through MLP projection head.
        
        Args:
            x: Input tensor with shape [B, ..., D] where D is feature dimension
            training: Boolean indicating training mode
            
        Returns:
            Projected features with same shape as input [B, ..., D]
        """
        # Shape validation
        tf.debugging.assert_rank_at_least(x, 2, 
            message="Input must have at least 2 dimensions: batch and features")
        
        # Apply layers sequentially
        x = self.fc1(x)
        x = self.norm1(x, training=training) # Pass training flag if applicable
        x = self.activation_layer(x)
        x = self.dropout_layer(x, training=training)
        x = self.fc2(x)
        projected = self.norm2(x, training=training) # Pass training flag if applicable
        
        # Ensure output shape matches input shape (except for the feature dimension which is projected to self.dim)
        # We cannot assert the full shape equality if dim != input feature dim.
        # Let's assert the rank and the final dimension.
        tf.debugging.assert_rank(projected, tf.rank(x),
            message="Output rank must match input rank")
        tf.debugging.assert_equal(tf.shape(projected)[-1], self.dim,
            message=f"Output feature dimension must be {self.dim}")

        return projected

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_rate,
            "activation": self.activation_name
        })
        return config