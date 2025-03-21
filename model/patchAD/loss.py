import tensorflow as tf

def kl_divergence(p, q):
    return tf.reduce_sum(p * tf.math.log(p / q + 1e-8), axis=-1)

class PatchADLoss(tf.keras.losses.Loss):
    def __init__(self, constraint_coef=0.2):
        super().__init__()
        self.c = constraint_coef
        
    def call(self, y_true, outputs):
        # Unpack outputs from PatchAD
        x_inter = outputs['inter']
        x_intra = outputs['intra']
        proj_inter = outputs['proj_inter']
        proj_intra = outputs['proj_intra']
        rec_inter = outputs['rec_inter']
        rec_intra = outputs['rec_intra']
        
        pdb.set_trace()
        # Reconstruction: combine both branches
        rec_pred = rec_inter + rec_intra
        rec_loss = tf.keras.losses.MeanSquaredError()(y_true, rec_pred)
        
        # Contrastive loss between the two views.
        l_n = kl_divergence(x_inter, tf.stop_gradient(x_intra)) \
              + kl_divergence(tf.stop_gradient(x_intra), x_inter)
        l_p = kl_divergence(x_intra, tf.stop_gradient(x_inter)) \
              + kl_divergence(tf.stop_gradient(x_inter), x_intra)
        # Divide by the number of patches (assumed along axis=1)
        cont_loss = (l_n - l_p) / tf.cast(tf.shape(x_inter)[1], tf.float32)
        
        # Projection loss: enforce similarity between projected features and the other view.
        l_n_proj = kl_divergence(proj_inter, tf.stop_gradient(x_intra)) \
                   + kl_divergence(x_inter, tf.stop_gradient(proj_intra))
        proj_loss = l_n_proj / tf.cast(tf.shape(x_inter)[1], tf.float32)
        
        # Total loss: a weighted combination of contrastive, projection, and reconstruction losses.
        total_loss = (1 - self.c) * cont_loss + self.c * proj_loss + rec_loss
        return total_loss

@tf.function
def compute_anomaly_score(outputs):
    """
    Computes an anomaly score based on the discrepancy between the inter- and intra- representations.
    Also returns the reconstruction prediction.
    
    Args:
      outputs: A dictionary with keys:
          'inter'    : inter-patch representation (tensor)
          'intra'    : intra-patch representation (tensor)
          'rec_inter': reconstruction from the inter branch (tensor)
          'rec_intra': reconstruction from the intra branch (tensor)
    
    Returns:
      anomaly_score: A tensor containing the anomaly score computed as the sum of KL divergences.
      rec_pred: The reconstructed signal (rec_inter + rec_intra).
    """
    x_inter = outputs['inter']
    x_intra = outputs['intra']
    rec_pred = outputs['rec_inter'] + outputs['rec_intra']
    anomaly_score = kl_divergence(x_inter, x_intra) + kl_divergence(x_intra, x_inter)
    return anomaly_score, rec_pred
