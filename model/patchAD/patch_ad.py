import tensorflow as tf
import numpy as np
from einops.layers.tensorflow import Rearrange
from model.patchAD.modules import MLPBlock, PatchMixerLayer, ProjectHead, EncoderEnsemble


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, max_len=5000):
        super().__init__()
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, input_dim, 2) * -(np.log(10000.0) / input_dim))
        pe = np.zeros((max_len, input_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return self.pe[:, :seq_len, :]


class PatchAD(tf.keras.Model):
    def __init__(self, input_channels, seq_length, patch_sizes, d_model=256, num_layers=3,
                 dropout=0.0, activation="gelu", norm="ln"):
        super().__init__()
        self.patch_sizes = sorted(patch_sizes)
        self.win_size = seq_length

        self.win_emb = PositionalEmbedding(input_channels)

        self.patch_num_emb = [tf.keras.layers.Dense(d_model) for _ in self.patch_sizes]
        self.patch_size_emb = [tf.keras.layers.Dense(d_model) for _ in self.patch_sizes]

        self.patch_encoders = [
            EncoderEnsemble([
                PatchMixerLayer(
                    in_len=seq_length, hid_len=40, in_chn=input_channels,
                    hid_chn=int(input_channels * 1.2), out_chn=input_channels,
                    patch_size=p, hid_pch=int(p * 1.2), d_model=d_model,
                    norm=norm, activ=activation, drop=dropout, jump_conn='proj'
                ) for _ in range(num_layers)
            ]) for p in self.patch_sizes
        ]

        # self.patch_num_mixer = tf.keras.Sequential([
        #     tf.keras.layers.Dense(d_model // 2, activation=activation),
        #     tf.keras.layers.Dense(d_model),
        #     tf.keras.layers.Softmax(-1)
        # ])
        # self.patch_size_mixer = tf.keras.Sequential([
        #     tf.keras.layers.Dense(d_model // 2, activation=activation),
        #     tf.keras.layers.Dense(d_model),
        #     tf.keras.layers.Softmax(-1)
        # ])

        self.recons_num = [tf.keras.Sequential([
            Rearrange('b c n d -> b c (n d)'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(seq_length),
            Rearrange('b c l -> b l c')
        ]) for p in self.patch_sizes]

        self.recons_size = [tf.keras.Sequential([
            Rearrange('b c p d -> b c (p d)'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(seq_length),
            Rearrange('b c l -> b l c')
        ]) for p in self.patch_sizes]

        self.rec_alpha = self.add_weight("rec_alpha", shape=(len(self.patch_sizes),),
                                         initializer=tf.keras.initializers.Constant(0.5), trainable=True)

        self.proj_head_inter = ProjectHead(d_model)
        self.proj_head_intra = ProjectHead(d_model)

    def call(self, x, training=False):
        B, T, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        
        x = x + self.win_emb(x)

        rec_total = 0
        outputs_inter, outputs_intra = None, None
        patch_num_dist_list, patch_size_dist_list = [], []

        for i, patch_size in enumerate(self.patch_sizes):
            num_patches = T // patch_size

            x_patched = tf.reshape(x, [B, num_patches, patch_size, C])
            x_patch_num = tf.transpose(x_patched, [0, 3, 1, 2])
            x_patch_size = tf.transpose(x_patched, [0, 3, 2, 1])

            x_patch_num = self.patch_num_emb[i](x_patch_num)
            x_patch_size = self.patch_size_emb[i](x_patch_size)

            encoder = self.patch_encoders[i]
            inter_out, intra_out, _, _ = encoder(x_patch_num, x_patch_size, training=training)

            # patch_num_dist_list.append(self.patch_num_mixer(inter_out))
            # patch_size_dist_list.append(self.patch_size_mixer(intra_out))

            patch_num_dist_list.append(inter_out)
            patch_size_dist_list.append(intra_out)

            rec1 = self.recons_num[i](inter_out)
            rec2 = self.recons_size[i](intra_out)
            rec_i = self.rec_alpha[i] * rec1 + (1 - self.rec_alpha[i]) * rec2

            rec_total += rec_i

            outputs_inter, outputs_intra = inter_out, intra_out

        rec_final = rec_total / len(self.patch_sizes)

        proj_inter = self.proj_head_inter(tf.reduce_mean(outputs_inter, axis=1))
        proj_intra = self.proj_head_intra(tf.reduce_mean(outputs_intra, axis=1))

        return {
            'patch_num_dist_list': patch_num_dist_list,
            'patch_size_dist_list': patch_size_dist_list,
            'reconstruction': rec_final,
            'proj_inter': proj_inter,
            'proj_intra': proj_intra
        }
