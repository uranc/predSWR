import tensorflow as tf
from tensorflow_addons.activations import gelu # Assuming gelu is used based on other code

class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, dim, in_features, hid_features, out_features, activ="gelu", drop=0.0, jump_conn="proj", name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        assert out_features > 0, f"MLPBlock: out_features must be > 0, got {out_features}"
 
        self.dim = dim
        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = out_features

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.fc1 = tf.keras.layers.Dense(hid_features)
        if isinstance(activ, str):
            # Use tf.keras.activations directly for strings in graph mode
            self.act = tf.keras.activations.get(activ)
        elif callable(activ):
             self.act = activ
        else:
             self.act = tf.keras.activations.get('gelu')

        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.fc2 = tf.keras.layers.Dense(out_features)
        self.drop = tf.keras.layers.Dropout(drop)

        # Always use projection for jump connection since runtime shapes might differ
        # from what's specified in in_features/out_features
        self.jump = tf.keras.layers.Dense(out_features)

    def call(self, x, training=False):
        # Use tf.shape and tf.rank for graph compatibility
        input_shape = tf.shape(x)
        rank = tf.rank(x)
        target_axis = rank - self.dim

        # Calculate permutations dynamically
        perm_list = tf.range(rank)
        perm = tf.concat([perm_list[:target_axis], perm_list[target_axis+1:], [target_axis]], axis=0)
        inv_perm = tf.math.invert_permutation(perm)

        x_perm = tf.transpose(x, perm=perm)

        # --- Jump Connection ---
        jump_out = self.jump(x_perm)

        # --- Main Path ---
        out = self.norm1(x_perm)
        out = self.fc1(out)
        out = self.act(out)
        out = self.drop(out, training=training)
        out = self.norm2(out)
        out = self.fc2(out)

        # --- Residual Addition ---
        # Debug shapes if needed
        # tf.print("Jump shape:", tf.shape(jump_out), "Output shape:", tf.shape(out))
        out = jump_out + out

        # --- Inverse Transpose ---
        out = tf.transpose(out, perm=inv_perm)

        return out

##############################
# PatchMLP_layer_TF          #
##############################
class PatchMixerLayer(tf.keras.layers.Layer):
    def __init__(self, in_len, hid_len, in_chn, hid_chn, out_chn,
                 patch_size, hid_pch, d_model, norm="ln", activ="gelu", drop=0.0, jump_conn="proj", name=None, **kwargs):
        """
        Implements a single Patch MLP layer according to the PatchAD architecture:
        1. Shared channel mixer for both inter and intra patches
        2. Separate patch mixers for inter and intra patches 
        3. Feature mixer at the end
        """
        base_name = name if name else f"patch_mixer_p{patch_size}"
        super().__init__(name=base_name, **kwargs)

        N = in_len              # Number of patches for inter-patch (x_patch_num)
        P = patch_size          # Number of patches for intra-patch (x_patch_size)

        assert N > 0 and P > 0, f"PatchMixerLayer: N={N}, P={P} must be > 0. (in_len={in_len}, patch_size={patch_size})"

        # 1. Shared channel mixer (operates on the feature dimension D for both types of patches)
        self.channel_mixer = MLPBlock(dim=1, in_features=d_model, hid_features=hid_chn, out_features=d_model,
                                      activ=activ, drop=drop, jump_conn=jump_conn, name=f"{base_name}_channel_mixer")
        
        # 2. Separate patch mixers
        # Inter-patch mixer (operates on N dimension)
        self.inter_patch_mixer = MLPBlock(dim=2, in_features=N, hid_features=hid_len, out_features=N,
                                         activ=activ, drop=drop, jump_conn=jump_conn, name=f"{base_name}_inter_patch_mixer")
        
        # Intra-patch mixer (operates on P dimension)
        self.intra_patch_mixer = MLPBlock(dim=2, in_features=P, hid_features=hid_pch, out_features=P,
                                          activ=activ, drop=drop, jump_conn=jump_conn, name=f"{base_name}_intra_patch_mixer")
        
        # 3. Feature mixer at the end (operates on feature dimension D for both types of patches)
        self.feature_mixer = MLPBlock(dim=1, in_features=d_model, hid_features=hid_chn, out_features=out_chn,
                                      activ=activ, drop=drop, jump_conn=jump_conn, name=f"{base_name}_feature_mixer")

    def call(self, x_patch_num, x_patch_size, training=False):
        # Input shapes: x_patch_num [B, C, N, D], x_patch_size [B, C, P, D]
        
        # 1. Apply shared channel mixer to both types of patches
        chan_num = self.channel_mixer(x_patch_num, training=training)
        chan_size = self.channel_mixer(x_patch_size, training=training)
        
        # 2. Apply separate patch mixers
        patch_num = self.inter_patch_mixer(chan_num, training=training)    # Inter-patch mixing
        patch_size = self.intra_patch_mixer(chan_size, training=training)  # Intra-patch mixing
        
        # 3. Apply feature mixer at the end
        out_num = self.feature_mixer(patch_num, training=training)
        out_size = self.feature_mixer(patch_size, training=training)
        
        return out_num, out_size

# ... rest of EncoderEnsemble and ProjectHead ...
class EncoderEnsemble(tf.keras.layers.Layer):
    def __init__(self, encoder_layers, name=None, **kwargs):
        base_name = name if name else "encoder_ensemble"
        super().__init__(name=base_name, **kwargs)
        self.encoder_layers = encoder_layers
        self.num_layers = len(encoder_layers)
        self.alpha = self.add_weight("alpha", shape=(self.num_layers,), initializer="ones")
        self.beta  = self.add_weight("beta", shape=(self.num_layers,), initializer="ones")

    def call(self, x_patch_num, x_patch_size, training=False):
        inter_reps = []
        intra_reps = []

        current_num = x_patch_num
        current_size = x_patch_size

        for layer in self.encoder_layers:
            # Pass the outputs of the previous layer to the next
            current_num, current_size = layer(current_num, current_size, training=training)
            inter_reps.append(current_num)
            intra_reps.append(current_size)

        # Weight and sum outputs. Use tf.nn.softmax for stability
        alpha_w = tf.nn.softmax(self.alpha)
        beta_w  = tf.nn.softmax(self.beta)

        # Ensure broadcasting works correctly if reps are lists of tensors
        weighted_inter = [alpha_w[i] * inter_reps[i] for i in range(self.num_layers)]
        weighted_intra = [beta_w[i] * intra_reps[i] for i in range(self.num_layers)]

        x_inter = tf.add_n(weighted_inter)
        x_intra = tf.add_n(weighted_intra)

        return x_inter, x_intra, inter_reps, intra_reps # Return original reps if needed elsewhere


class ProjectHead(tf.keras.layers.Layer):
    def __init__(self, dim, name=None, **kwargs):
        base_name = name if name else "project_head"
        super().__init__(name=base_name, **kwargs)
        # Consider adding activation/normalization if needed
        self.fc1 = tf.keras.layers.Dense(dim, name=f"{base_name}_fc1")
        self.fc2 = tf.keras.layers.Dense(dim, name=f"{base_name}_fc2")

    def call(self, x, training=False):
        # Add activation, e.g., ReLU, between layers?
        x = self.fc1(x)
        # x = tf.nn.relu(x) # Example activation
        x = self.fc2(x)
        return x