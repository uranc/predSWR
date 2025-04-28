import tensorflow as tf
import numpy as np
from einops import rearrange
from model.patchAD.modules import MLPBlock, PatchMixerLayer, ProjectHead, EncoderEnsemble

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, max_len=5000, name="positional_embedding", **kwargs):
        super().__init__(name=name, **kwargs)
        position = np.arange(max_len)[:, None]
        div_term = np.exp(np.arange(0, input_dim, 2, dtype=np.float32) * -(np.log(10000.0) / input_dim))
        pe = np.zeros((max_len, input_dim), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        if input_dim % 2 != 0:
            div_term_odd = np.exp(np.arange(0, input_dim - 1, 2, dtype=np.float32) * -(np.log(10000.0) / (input_dim - 1)))
            cos_len = pe[:, 1::2].shape[1]
            pe[:, 1::2] = np.cos(position * div_term_odd[:cos_len])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[None], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

class PatchAD(tf.keras.Model):
    def __init__(self,
                 input_channels,
                 seq_length,
                 patch_sizes,
                 d_model=256,
                 num_layers=3,
                 dropout=0.0,
                 activation="gelu",
                 norm="ln",
                 name="patch_ad_model", **kwargs):
        super().__init__(name=name, **kwargs)
        self.patch_sizes    = sorted(patch_sizes)
        self.seq_length     = seq_length
        self.input_channels = input_channels
        self.d_model        = d_model

        self.win_emb = PositionalEmbedding(input_channels, max_len=seq_length, name=f"{name}_pos_emb")
        self.patch_num_emb  = [tf.keras.layers.Dense(d_model, name=f"{name}_patch_num_emb_p{p}") for p in self.patch_sizes]
        self.patch_size_emb = [tf.keras.layers.Dense(d_model, name=f"{name}_patch_size_emb_p{p}") for p in self.patch_sizes]

        self.patch_encoders = []
        for i, p in enumerate(self.patch_sizes):
            mixer_layers_list = []
            N = seq_length // p
            hid_mixer_dim = d_model
            for layer_idx in range(num_layers):
                mixer_layers_list.append(PatchMixerLayer(
                    in_len=N,
                    hid_len=hid_mixer_dim,
                    in_chn=self.input_channels,
                    hid_chn=hid_mixer_dim,
                    out_chn=d_model,
                    patch_size=p,
                    hid_pch=hid_mixer_dim,
                    d_model=d_model,
                    norm=norm,
                    activ=activation,
                    drop=dropout,
                    jump_conn="proj",
                    name=f"{name}_p{p}_mixer_layer_{layer_idx}"
                ))
            self.patch_encoders.append(EncoderEnsemble(mixer_layers_list, name=f"{name}_p{p}_encoder_ensemble"))

        self.recons_num  = []
        self.recons_size = []
        for i, p in enumerate(self.patch_sizes):
            N = seq_length // p
            self.recons_num.append(tf.keras.Sequential([
                tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"{name}_p{p}_recons_num_norm"),
                tf.keras.layers.Dense(p, name=f"{name}_p{p}_recons_num_dense"),
            ], name=f"{name}_p{p}_recons_num_head"))
            self.recons_size.append(tf.keras.Sequential([
                tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"{name}_p{p}_recons_size_norm"),
                tf.keras.layers.Dense(N, name=f"{name}_p{p}_recons_size_dense"),
            ], name=f"{name}_p{p}_recons_size_head"))

        self.rec_alpha = self.add_weight(
            'rec_alpha',
            shape=(len(self.patch_sizes),),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True
        )

        proj_dim = d_model
        self.proj_head_inter = ProjectHead(proj_dim, name=f"{name}_proj_head_inter")
        self.proj_head_intra = ProjectHead(proj_dim, name=f"{name}_proj_head_intra")
        self.proj_head_inter.proj_dim = proj_dim
        self.proj_head_intra.proj_dim = proj_dim

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        T = self.seq_length
        C = self.input_channels
        D = self.d_model

        x_embedded = self.win_emb(x)
        r1_total = tf.zeros_like(x)
        r2_total = tf.zeros_like(x)
        last_scale_inter = None
        last_scale_intra = None
        last_scale_p = None

        for idx, p in enumerate(self.patch_sizes):
            N = T // p
            x_patches = tf.reshape(x_embedded, (B, N, p, C))
            inter_view_input = rearrange(x_patches, 'b n p c -> b c n p')
            inter_view_embedded = self.patch_num_emb[idx](inter_view_input)
            intra_view_input = rearrange(x_patches, 'b n p c -> b c p n')
            intra_view_embedded = self.patch_size_emb[idx](intra_view_input)
            encoder = self.patch_encoders[idx]
            final_inter, final_intra, _, _ = encoder(inter_view_embedded, intra_view_embedded, training=training)
            last_scale_inter = final_inter
            last_scale_intra = final_intra
            last_scale_p = p

            r1_patches = self.recons_num[idx](final_inter)
            r2_patches = self.recons_size[idx](final_intra)
            r1_temp = rearrange(r1_patches, 'b c n p -> b c (n p)')
            r1 = rearrange(r1_temp, 'b c t -> b t c')
            r2_temp = rearrange(r2_patches, 'b c p n -> b c (p n)')
            r2 = rearrange(r2_temp, 'b c t -> b t c')
            alpha = tf.sigmoid(self.rec_alpha[idx])
            r1_total += alpha * r1
            r2_total += (1.0 - alpha) * r2

        rec_final = r1_total + r2_total
        reconstruction_inter = r1_total
        reconstruction_intra = r2_total

        # --- PatchAD/DCdetector upsampling ---
        p = last_scale_p
        N = T // p
        # Inter: repeat each patch feature for each point in the patch
        inter_avg_c = tf.reduce_mean(last_scale_inter, axis=1)  # [B, N, D]
        upsampled_inter = tf.repeat(inter_avg_c, repeats=p, axis=1)  # [B, N*P, D] == [B, T, D]
        # Intra: tile the patch's sequence for each patch
        intra_avg_c = tf.reduce_mean(last_scale_intra, axis=1)  # [B, P, D]
        upsampled_intra = tf.tile(intra_avg_c, [1, N, 1])  # [B, P*N, D] == [B, T, D]

        proj_inter = self.proj_head_inter(upsampled_inter, training=training)
        proj_intra = self.proj_head_intra(upsampled_intra, training=training)

        return {
            'reconstruction': rec_final,
            'reconstruction_inter': reconstruction_inter,
            'reconstruction_intra': reconstruction_intra,
            'final_features_inter': upsampled_inter,
            'final_features_intra': upsampled_intra,
            'proj_inter': proj_inter,
            'proj_intra': proj_intra
        }