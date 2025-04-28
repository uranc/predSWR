import tensorflow as tf

def kl_divergence(p, q, epsilon=1e-8):
    """
    Symmetric KL divergence between two probability distributions along the last axis.
    """
    safe_p = p + epsilon
    safe_q = q + epsilon
    kl1 = tf.reduce_sum(safe_p * tf.math.log(safe_p / safe_q), axis=-1)
    kl2 = tf.reduce_sum(safe_q * tf.math.log(safe_q / safe_p), axis=-1)
    return (kl1 + kl2) / 2.0

def patch_loss(prediction_targets, prediction_out, proj_inter, proj_intra, x_inter, x_intra, constraint_coeff=0.5):
    """
    PatchAD objective as described in the paper.
    - prediction_targets: [B, T, C]
    - prediction_out: [B, T, C]
    - proj_inter, proj_intra: [B, T, D_proj]
    - x_inter, x_intra: [B, T, D]
    - constraint_coeff: c in the final loss equation
    """
    # 1. Softmax for KL (probabilities)
    x_inter_prob = tf.nn.sigmoid(x_inter)
    x_intra_prob = tf.nn.sigmoid(x_intra)
    proj_inter_prob = tf.nn.sigmoid(proj_inter)
    proj_intra_prob = tf.nn.sigmoid(proj_intra)

    # 2. Inter-intra discrepancy (content loss, L_cont)
    # L_N{P,N}
    kl_N = tf.reduce_mean(
        kl_divergence(x_inter_prob, tf.stop_gradient(x_intra_prob)) +
        kl_divergence(tf.stop_gradient(x_intra_prob), x_inter_prob)
    )
    # L_P{P,N}
    kl_P = tf.reduce_mean(
        kl_divergence(x_intra_prob, tf.stop_gradient(x_inter_prob)) +
        kl_divergence(tf.stop_gradient(x_inter_prob), x_intra_prob)
    )
    # L_cont = L_N - L_P / len(N) (len(N) = T)
    epsilon = 1e-8 # Define epsilon for numerical stability
    seq_len_float = tf.cast(tf.shape(x_inter)[1], tf.float32) + epsilon # Add epsilon

    L_cont = (kl_N - kl_P) / seq_len_float

    # 3. Projection constraint (L_proj) according to Eq. 9
    # Calculate L_N' = KL(N', P) + KL(P, N') where N'=proj_inter, P=x_intra
    kl_N_prime = tf.reduce_mean(
        kl_divergence(proj_inter_prob, tf.stop_gradient(x_intra_prob)) +
        kl_divergence(tf.stop_gradient(x_intra_prob), proj_inter_prob)
    )
    # Calculate L_P' = KL(P', N) + KL(N, P') where P'=proj_intra, N=x_inter
    kl_P_prime = tf.reduce_mean(
        kl_divergence(proj_intra_prob, tf.stop_gradient(x_inter_prob)) +
        kl_divergence(tf.stop_gradient(x_inter_prob), proj_intra_prob)
    )

    # L_proj = (L_N' - L_P) / len(N) + (L_N - L_P') / len(N)
    # Use kl_N_prime for L_N', kl_P for L_P, kl_N for L_N, kl_P_prime for L_P'
    term1 = (kl_N_prime - kl_P) / seq_len_float
    term2 = (kl_N - kl_P_prime) / seq_len_float
    L_proj = term1 + term2
    
    # 4. Reconstruction loss (MSE)
    L_rec = tf.reduce_mean(tf.square(prediction_targets - prediction_out))

    # 5. Final loss
    proj_weight = 0.2
    total_loss = (1 - proj_weight) * L_cont + proj_weight * L_proj + constraint_coeff*L_rec

    # tf.print("Loss components - Rec:", L_rec, "L_cont:", L_cont, "L_proj:", L_proj)

    return total_loss