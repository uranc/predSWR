import tensorflow as tf
from tensorflow.keras.layers import (Dense, Dropout, Layer, LayerNormalization, 
                                   Reshape, Permute, Softmax, Activation)
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import GlorotUniform
from tensorflow_addons.activations import gelu
from .modules import PatchMixerLayer, MLPBlock, EncoderEnsemble, ProjectHead
import pdb

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, max_len=5000, name="positional_embedding", **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Create position and dimension indices
        position = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]  # [max_len, 1]
        
        # Calculate number of dimensions for sin/cos terms
        half_dim = input_dim // 2
        div_term = tf.exp(
            tf.range(0, half_dim, dtype=tf.float32) * (-tf.math.log(10000.0) / half_dim)
        )
        
        # Initialize PE matrix
        pe = tf.zeros((max_len, input_dim))
        
        # Calculate sin terms for even indices
        angles = tf.matmul(position, div_term[tf.newaxis, :])  # [max_len, half_dim]
        sin_terms = tf.sin(angles)
        
        # Update even indices (0, 2, 4, ...)
        even_indices = tf.range(0, input_dim, 2)
        pe = tf.tensor_scatter_nd_update(
            pe,
            tf.stack([
                tf.repeat(tf.range(max_len), len(even_indices)),
                tf.tile(even_indices, [max_len])
            ], axis=1),
            tf.reshape(sin_terms, [-1])
        )
        
        # Calculate cos terms for odd indices
        if input_dim % 2 != 0:  # Handle odd input_dim
            div_term = div_term[:-1]  # Remove last element
        cos_terms = tf.cos(angles[:, :input_dim//2])  # Only use needed terms
        
        # Update odd indices (1, 3, 5, ...)
        odd_indices = tf.range(1, input_dim, 2)
        pe = tf.tensor_scatter_nd_update(
            pe,
            tf.stack([
                tf.repeat(tf.range(max_len), len(odd_indices)),
                tf.tile(odd_indices, [max_len])
            ], axis=1),
            tf.reshape(cos_terms, [-1])
        )
        
        # Store as non-trainable constant
        self.pe = tf.constant(pe[tf.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

class PatchAD(tf.keras.Model):
    def __init__(self,
                input_channels,
                seq_length,
                patch_sizes,
                d_model=50,
                e_layer=3,
                dropout=0.0,
                activation="elu",
                norm="ln",
                mlp_hid_len_ratio=0.7,
                mlp_hid_chn_ratio=1.2,
                mlp_out_chn_ratio=1.0,
                mlp_hid_pch_ratio=1.2,
                mixer_mlp_ratio=0.5,
                name="patch_ad_model",
                **kwargs):
        
        super().__init__(name=name, **kwargs)
        # Store model parameters
        self.patch_sizes = patch_sizes
        self.seq_length = seq_length
        self.d_model = d_model
        self.input_channels = input_channels
        self.e_layer = e_layer
        
        # Initialize activation function
        self.activation_fn = gelu if activation == "gelu" else tf.keras.activations.get(activation)
        
        # Kernel initializer for all Dense layers
        # self.kernel_init = GlorotUniform()
        
        # 1. Positional Embedding
        self.win_emb = PositionalEmbedding(input_channels, max_len=seq_length+100)
        
        # 2. Patch Embeddings (with proper initialization)
        self.patch_num_embeddings = []
        self.patch_size_embeddings = []
        for p in patch_sizes:
            self.patch_num_embeddings.append(
                Dense(d_model, 
                      kernel_initializer=tf.keras.initializers.GlorotUniform(),
                      name=f"patch_num_emb_{p}")
            )
            self.patch_size_embeddings.append(
                Dense(d_model,
                      kernel_initializer=tf.keras.initializers.GlorotUniform(),
                      name=f"patch_size_emb_{p}")
            )
        
        # 3. Encoders
        self.patch_encoders = []
        for i, p in enumerate(patch_sizes):
            patch_size = p
            patch_num = seq_length // patch_size
            if seq_length % patch_size != 0:
                raise ValueError(f"seq_length ({seq_length}) must be divisible by patch_size ({patch_size})")
                
            # Calculate hidden dimensions
            hid_len = max(32, int(patch_num * mlp_hid_len_ratio))
            hid_chn = max(1, int(input_channels * mlp_hid_chn_ratio))
            out_chn = max(1, int(input_channels * mlp_out_chn_ratio))
            hid_pch = max(32, int(patch_size * mlp_hid_pch_ratio))
            
            # Create encoder layers
            enc_layers = [
                PatchMixerLayer(
                    in_len=seq_length,
                    hid_len=hid_len,
                    in_chn=input_channels,
                    hid_chn=hid_chn,
                    out_chn=out_chn,
                    patch_size=patch_size,
                    hid_pch=hid_pch,
                    d_model=d_model,
                    norm=norm,
                    activ=activation,
                    drop=dropout,
                    jump_conn='proj',
                    name=f"p{p}_layer{j}"
                )
                for j in range(e_layer)
            ]
            self.patch_encoders.append(
                EncoderEnsemble(encoder_layers=enc_layers, name=f"p{p}_encoder")
            )
        

        # 4. Mixers (directly built instead of using Sequential)
        mixer_hidden_dim = max(32, int(d_model * mixer_mlp_ratio))
        
        # Num mixer layers
        self.num_mixer_mlp = MLPBlock(
            dim=2,
            in_features=d_model,
            hid_features=mixer_hidden_dim,
            out_features=d_model,
            activ=self.activation_fn,
            drop=dropout,
            jump_conn='trunc',
            norm=norm,
            name="num_mixer_mlp"
        )
        self.num_mixer_softmax = Softmax(axis=-1, name="num_mixer_softmax")
        
        # Size mixer layers
        self.size_mixer_mlp = MLPBlock(
            dim=2,
            in_features=d_model,
            hid_features=mixer_hidden_dim,
            out_features=d_model,
            activ=self.activation_fn,
            drop=dropout,
            jump_conn='trunc',
            norm=norm,
            name="size_mixer_mlp"
        )
        self.size_mixer_softmax = Softmax(axis=-1, name="size_mixer_softmax")

        # 5. Projection Heads (NEW)
        self.proj_num_heads = []
        self.proj_size_heads = []
        for i, p in enumerate(patch_sizes):
             self.proj_num_heads.append(
                 ProjectHead(dim=d_model, name=f"p{p}_proj_num_head")
             )
             self.proj_size_heads.append(
                 ProjectHead(dim=d_model, name=f"p{p}_proj_size_head")
             )
        
        # 6. Reconstruction Heads
        self.recons_num_heads = []
        self.recons_size_heads = []
        self.rec_alpha = []
        
        for i, p in enumerate(patch_sizes):
            patch_size = p
            patch_num = seq_length // patch_size
            
            # Build num reconstruction head layers
            num_head = {
                'reshape': Reshape((-1, patch_num * d_model),
                                input_shape=(self.input_channels, patch_num, d_model),
                                name=f"p{p}_rec_num_flatten"),
                'norm1': LayerNormalization(epsilon=1e-5, name=f"p{p}_rec_num_norm1"),
                'dense1': Dense(d_model, kernel_initializer=tf.keras.initializers.GlorotUniform(),
                            name=f"p{p}_rec_num_dense1"),
                'activ1': Activation(self.activation_fn, name=f"p{p}_rec_num_activ1"),
                'norm2': LayerNormalization(epsilon=1e-5, name=f"p{p}_rec_num_norm2"),
                'dense2': Dense(seq_length, kernel_initializer=tf.keras.initializers.GlorotUniform(),
                            name=f"p{p}_rec_num_dense2"),
                'permute': Permute((2, 1), name=f"p{p}_rec_num_permute")
            }
            self.recons_num_heads.append(num_head)
            
            # Build size reconstruction head layers
            size_head = {
                'reshape': Reshape((-1, patch_size * d_model),
                                input_shape=(self.input_channels, patch_size, d_model),
                                name=f"p{p}_rec_size_flatten"),
                'norm1': LayerNormalization(epsilon=1e-5, name=f"p{p}_rec_size_norm1"),
                'dense1': Dense(d_model, kernel_initializer=tf.keras.initializers.GlorotUniform(),
                            name=f"p{p}_rec_size_dense1"),
                'activ1': Activation(self.activation_fn, name=f"p{p}_rec_size_activ1"),
                'norm2': LayerNormalization(epsilon=1e-5, name=f"p{p}_rec_size_norm2"),
                'dense2': Dense(seq_length, kernel_initializer=tf.keras.initializers.GlorotUniform(),
                            name=f"p{p}_rec_size_dense2"),
                'permute': Permute((2, 1), name=f"p{p}_rec_size_permute")
            }
            self.recons_size_heads.append(size_head)
            
            # Initialize reconstruction alpha weights
            self.rec_alpha.append(
                self.add_weight(
                    name=f'rec_alpha_{p}',
                    shape=(),
                    initializer=tf.keras.initializers.Constant(0.5),
                    trainable=True
                )
            )

    @tf.function
    def call(self, x, training=False):
        # Shape validation
        input_shape = tf.shape(x)
        B, L, C = input_shape[0], input_shape[1], input_shape[2]
        
        tf.debugging.assert_equal(
            L, self.seq_length,
            message=f"Input sequence length {L} does not match model's seq_length {self.seq_length}"
        )
        tf.debugging.assert_equal(
            C, self.input_channels,
            message=f"Input channel count {C} does not match model's input_channels {self.input_channels}"
        )
        
        # Apply positional embedding
        x_emb = self.win_emb(x, training=training)
        
        # Initialize output collections
        all_patch_num_dists = []
        all_patch_size_dists = []
        all_proj_num = []      # NEW: For projection head outputs
        all_proj_size = []     # NEW: For projection head outputs
        all_scale_recons = []
        all_patch_num_mx = []  # Initialize mixer output list
        all_patch_size_mx = [] # Initialize mixer output list
        
        # Process each patch size
        for patch_idx, patch_size in enumerate(self.patch_sizes):
            patch_num = self.seq_length // patch_size
            
            # Prepare patches
            x_patch_num = tf.reshape(x_emb, [B, patch_num, patch_size, C])
            x_patch_num = tf.transpose(x_patch_num, perm=[0, 3, 1, 2])
            
            x_patch_size = tf.reshape(x_emb, [B, patch_size, patch_num, C])
            x_patch_size = tf.transpose(x_patch_size, perm=[0, 3, 1, 2])
            
            # Apply embeddings
            x_patch_num_emb = self.patch_num_embeddings[patch_idx](x_patch_num)
            x_patch_size_emb = self.patch_size_embeddings[patch_idx](x_patch_size)
            
            # Process through encoder
            final_dists_num, final_dists_size, layer_logits_num, layer_logits_size = (
                self.patch_encoders[patch_idx](
                    x_patch_num_emb, x_patch_size_emb,
                    training=training
                )
            )
            
            # Collect distributions
            all_patch_num_dists.extend(final_dists_num)
            all_patch_size_dists.extend(final_dists_size)
            
            # Process each layer
            layer_recons = []
            for layer_idx in range(len(layer_logits_num)):
                logits_num = layer_logits_num[layer_idx]
                logits_size = layer_logits_size[layer_idx]
                
                # Calculate mixers
                mean_logits_num = tf.reduce_mean(logits_num, axis=1)#, keepdims=True)  # [B, C, N, D] → [B, 1, N, D]
                mean_logits_size = tf.reduce_mean(logits_size, axis=1)#, keepdims=True)  # [B, C, P, D] → [B, 1, P, D]
                
                # Apply Projection Heads (NEW)
                proj_num = self.proj_num_heads[patch_idx](mean_logits_num, training=training)
                proj_size = self.proj_size_heads[patch_idx](mean_logits_size, training=training)
                patch_num_mx = self.num_mixer_softmax(self.num_mixer_mlp(proj_num, training=training))
                patch_size_mx = self.size_mixer_softmax(self.size_mixer_mlp(proj_size, training=training))
                
                
                all_patch_num_mx.append(patch_num_mx)
                all_patch_size_mx.append(patch_size_mx)
                
                # Calculate reconstructions
                def apply_recons_head(x, head_layers):
                    x = head_layers['reshape'](x)
                    x = head_layers['norm1'](x)
                    x = head_layers['dense1'](x)
                    x = head_layers['activ1'](x)
                    x = head_layers['norm2'](x)
                    x = head_layers['dense2'](x)
                    x = head_layers['permute'](x)
                    return x
                
                rec1 = apply_recons_head(logits_num, self.recons_num_heads[patch_idx])
                rec2 = apply_recons_head(logits_size, self.recons_size_heads[patch_idx])                

                # Combine reconstructions with learned alpha
                alpha = tf.nn.sigmoid(self.rec_alpha[patch_idx])
                rec_combined = rec1 * alpha + rec2 * (1.0 - alpha)
                layer_recons.append(rec_combined)
            
            # Average reconstructions for this scale
            if layer_recons:
                scale_recons_avg = tf.reduce_mean(tf.stack(layer_recons, axis=0), axis=0)
                all_scale_recons.append(scale_recons_avg)
        
        # Final reconstruction
        if all_scale_recons:
            final_rec_x = tf.reduce_mean(tf.stack(all_scale_recons, axis=0), axis=0)
        else:
            tf.print("Warning: No reconstructions generated, returning zeros")
            final_rec_x = tf.zeros_like(x)
        
        return all_patch_num_dists, all_patch_size_dists, all_patch_num_mx, all_patch_size_mx, final_rec_x