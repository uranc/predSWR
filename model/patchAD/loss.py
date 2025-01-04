import tensorflow as tf

def kl_divergence(p, q):
    return tf.reduce_sum(p * tf.math.log(p / q), axis=-1)

class PatchADLoss(tf.keras.losses.Loss):
    def __init__(self, constraint_coef=0.2):
        super().__init__()
        self.c = constraint_coef
        
    def call(self, y_true, outputs):
        # Unpack outputs
        x_inter = outputs['inter']
        x_intra = outputs['intra']
        proj_inter = outputs['proj_inter']
        proj_intra = outputs['proj_intra']
        rec_inter = outputs['rec_inter']
        rec_intra = outputs['rec_intra']
        
        # Reconstruction loss
        rec_loss = tf.keras.losses.MSE(rec_inter + rec_intra, y_true)
        
        # Contrastive loss
        l_n = kl_divergence(x_inter, tf.stop_gradient(x_intra)) + kl_divergence(tf.stop_gradient(x_intra), x_inter)
        l_p = kl_divergence(x_intra, tf.stop_gradient(x_inter)) + kl_divergence(tf.stop_gradient(tf.stop_gradient(x_inter), x_intra)
        cont_loss = (l_n - l_p) / tf.cast(tf.shape(x_inter)[1], tf.float32)
        
        # Projection loss
        l_n_proj = kl_divergence(proj_inter, tf.stop_gradient(x_intra)) + kl_divergence(x_inter, tf.stop_gradient(proj_intra))
        proj_loss = l_n_proj / tf.cast(tf.shape(x_inter)[1], tf.float32)
        
        # Total loss
        total_loss = (1 - self.c) * cont_loss + self.c * proj_loss + rec_loss
        return total_loss

@tf.function
def compute_anomaly_score(inter_rep, intra_rep):
    score = kl_divergence(inter_rep, intra_rep) + kl_divergence(intra_rep, inter_rep)
    return score
