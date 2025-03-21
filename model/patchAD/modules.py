import tensorflow as tf


class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, dim, in_features, hid_features, out_features,
                 activ="gelu", drop=0.0, jump_conn="proj", **kwargs):
        """
        Args:
          dim: integer indicating which dimension to transpose to last for mixing.
          in_features: size of the feature dimension to mix.
          hid_features: hidden units.
          out_features: output units.
          activ: activation name.
          drop: dropout rate.
          jump_conn: either "proj" (learn a projection) or "trunc" (identity).
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = out_features
        self.jump_conn = jump_conn
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.fc1 = tf.keras.layers.Dense(hid_features)
        self.act = tf.keras.activations.get(activ)
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.fc2 = tf.keras.layers.Dense(out_features)
        # self.dropout = tf.keras.layers.Dropout(drop)
        if jump_conn == "proj":
            self.jump = tf.keras.layers.Dense(out_features)
        elif jump_conn == "trunc":
            self.jump = tf.keras.layers.Lambda(lambda x: x)
        else:
            raise ValueError("jump_conn must be 'proj' or 'trunc'")
    def call(self, x, training=False):
        # We want to mix along the dimension specified by dim (counted from the end).
        # Save original shape.
        original_shape = tf.shape(x)
        
        # Create a permutation that puts the target dimension at the end
        rank = len(x.shape)
        # dim is 1-indexed from the end, so dim=1 means the last dimension
        target_axis = rank - self.dim
        
        # Build permutation that moves target_axis to the end
        perm = list(range(rank))
        perm.pop(target_axis)
        perm.append(target_axis)
        
        # Build inverse permutation to restore original order
        inv_perm = list(range(rank))
        for i, p in enumerate(perm):
            inv_perm[p] = i
            
        # Apply permutation
        x_perm = tf.transpose(x, perm=perm)
        
        # Apply the MLP operations
        jump_out = self.jump(x_perm)
        out = self.norm1(x_perm)
        out = self.fc1(out)
        out = self.act(out)
        out = self.norm2(out)
        out = self.fc2(out)
        # out = self.dropout(out, training=training)
        out = jump_out + out
        
        # Restore original dimension order
        out = tf.transpose(out, perm=inv_perm)
        return out

##############################
# PatchMLP_layer_TF          #
##############################
class PatchMixerLayer(tf.keras.layers.Layer):
    def __init__(self, in_len, hid_len, in_chn, hid_chn, out_chn,
                 patch_size, hid_pch, d_model, norm="ln", activ="gelu", drop=0.0, jump_conn="proj", **kwargs):
        """
        Implements a single Patch MLP layer.
        Args:
          in_len: total window length.
          hid_len: hidden units for patch number mixing.
          in_chn: number of input channels.
          hid_chn: hidden units for channel mixing (can be in_chn * expansion, e.g. int(in_chn*1.2)).
          out_chn: output channels for channel mixing.
          patch_size: current patch size.
          hid_pch: hidden units for patch size mixing.
          d_model: final dimension.
          norm, activ, drop, jump_conn: parameters passed to MLPBlock_TF.
        """
        super().__init__(**kwargs)
        self.ch_mixing1 = MLPBlock(dim=1, in_features=in_chn, hid_features=hid_chn, out_features=out_chn, activ=activ, drop=drop, jump_conn=jump_conn)
        self.patch_num_mix = MLPBlock(dim=2, in_features=in_len // patch_size, hid_features=hid_len, out_features=in_len // patch_size, activ=activ, drop=drop, jump_conn=jump_conn)
        self.patch_size_mix = MLPBlock(dim=2, in_features=patch_size, hid_features=hid_pch, out_features=patch_size, activ=activ, drop=drop, jump_conn=jump_conn)
        self.d_mixing1 = MLPBlock(dim=3, in_features=d_model, hid_features=d_model, out_features=d_model, activ=activ, drop=drop, jump_conn=jump_conn)
        # self.norm1 = tf.keras.layers.LayerNormalization()
        # self.norm2 = tf.keras.layers.LayerNormalization()
    def call(self, x_patch_num, x_patch_size, training=False):
        # x_patch_num shape: [B, C, N, D] ; x_patch_size shape: [B, C, P, D]
        # x_patch_num = self.norm1(x_patch_num)
        x_patch_num = self.ch_mixing1(x_patch_num, training=training)
        # x_patch_num = self.norm2(x_patch_num)
        x_patch_num = self.patch_num_mix(x_patch_num, training=training)
        # x_patch_num = self.norm2(x_patch_num)
        x_patch_num = self.d_mixing1(x_patch_num, training=training)
        # x_patch_size = self.norm1(x_patch_size)
        x_patch_size = self.ch_mixing1(x_patch_size, training=training)
        # x_patch_size = self.norm2(x_patch_size)
        x_patch_size = self.patch_size_mix(x_patch_size, training=training)
        # x_patch_size = self.norm2(x_patch_size)
        x_patch_size = self.d_mixing1(x_patch_size, training=training)
        return x_patch_num, x_patch_size

##############################
# Encoder Ensemble (TF)      #
##############################
class EncoderEnsemble(tf.keras.layers.Layer):
    def __init__(self, encoder_layers, **kwargs):
        """
        Wrap a list of PatchMLP_layer_TF layers and combine their outputs via learned weights.
        """
        super().__init__(**kwargs)
        self.encoder_layers = encoder_layers
        self.num_layers = len(encoder_layers)
        # Learnable weights for combining outputs from each layer.
        self.alpha = self.add_weight("alpha", shape=(self.num_layers,), initializer="ones")
        self.beta  = self.add_weight("beta", shape=(self.num_layers,), initializer="ones")
    def call(self, x_patch_num, x_patch_size, training=False):
        inter_reps = []
        intra_reps = []
        
        for layer in self.encoder_layers:
            x_patch_num, x_patch_size = layer(x_patch_num, x_patch_size, training=training)
            inter_reps.append(x_patch_num)
            intra_reps.append(x_patch_size)
        # Weight and sum outputs.
        alpha = tf.nn.softmax(self.alpha)
        beta  = tf.nn.softmax(self.beta)
        x_inter = tf.add_n([alpha[i] * inter_reps[i] for i in range(self.num_layers)])
        x_intra = tf.add_n([beta[i] * intra_reps[i] for i in range(self.num_layers)])
        return x_inter, x_intra, inter_reps, intra_reps


class ProjectHead(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(dim)
        self.fc2 = tf.keras.layers.Dense(dim)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        return x