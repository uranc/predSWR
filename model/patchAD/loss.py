import tensorflow as tf
import pdb

def tf_my_kl_loss(p, q, epsilon=1e-7):
    """
    TensorFlow equivalent of my_kl_loss. KL(p || q).
    Assumes p and q are probability distributions (last dim sums to 1 or are normalized).
    Input shape: [B, N, D]
    Output shape: [B, N]
    """
    # Add epsilon for numerical stability, similar to PyTorch 0.0000001
    safe_p = tf.clip_by_value(p, epsilon, 1.0)
    safe_q = tf.clip_by_value(q, epsilon, 1.0)
    # Element-wise KL divergence calculation
    res = safe_p * (tf.math.log(safe_p) - tf.math.log(safe_q))
    # Sum over the feature dimension (D)
    return tf.reduce_sum(res, axis=-1)

def tf_inter_intra_dist(p, q, w_de=True, training=True, temp=1.0, epsilon=1e-7):
    """
    TensorFlow equivalent of inter_intra_dist.
    Calculates symmetric KL divergence terms with stop_gradient.
    Input shapes: [B, N, D]
    Output: p_loss (scalar), q_loss (scalar)
    """
    # Apply temperature scaling if temp != 1.0
    # Note: Temperature scaling on probabilities requires re-normalization (softmax).
    # The PyTorch code applies it *after* softmax, which is unusual.
    # Replicating PyTorch directly:
    p_scaled = p * temp
    q_scaled = q * temp

    if training:
        if w_de: # Symmetric KL with stop_gradient
            kl_p_stop_q = tf_my_kl_loss(p_scaled, tf.stop_gradient(q_scaled), epsilon)
            kl_stop_q_p = tf_my_kl_loss(tf.stop_gradient(q_scaled), p_scaled, epsilon)
            p_loss = tf.reduce_mean(kl_p_stop_q) + tf.reduce_mean(kl_stop_q_p)

            kl_p_stop_q_detach = tf_my_kl_loss(tf.stop_gradient(p_scaled), q_scaled, epsilon)
            kl_q_p_detach = tf_my_kl_loss(q_scaled, tf.stop_gradient(p_scaled), epsilon)
            q_loss = tf.reduce_mean(kl_p_stop_q_detach) + tf.reduce_mean(kl_q_p_detach)
        else: # Asymmetric KL (negative)
             # This branch seems less likely based on the paper, but replicating PyTorch
            p_loss = -tf.reduce_mean(tf_my_kl_loss(p_scaled, tf.stop_gradient(q_scaled), epsilon))
            q_loss = -tf.reduce_mean(tf_my_kl_loss(q_scaled, tf.stop_gradient(p_scaled), epsilon))
    else: # Evaluation mode (no stop_gradient?) - PyTorch code is ambiguous here
          # Assuming we still want gradients for evaluation metrics if needed, but maybe not stop_gradient
          # Let's replicate the PyTorch structure, which seems to remove stop_gradient for eval
        if w_de:
            kl_p_q = tf_my_kl_loss(p_scaled, q_scaled, epsilon)
            kl_q_p = tf_my_kl_loss(q_scaled, p_scaled, epsilon)
            # Note: PyTorch uses p/q.detach() which is equivalent to stop_gradient in TF during training.
            # For eval, it calculates KL(p, q) + KL(q, p) without detach.
            p_loss = tf.reduce_mean(kl_p_q) + tf.reduce_mean(kl_q_p) # Symmetric KL
            q_loss = p_loss # Should be the same if calculated symmetrically
        else:
            p_loss = -tf.reduce_mean(tf_my_kl_loss(p_scaled, q_scaled, epsilon))
            q_loss = -tf.reduce_mean(tf_my_kl_loss(q_scaled, p_scaled, epsilon)) # Check if this sign is correct

    return p_loss, q_loss

def tf_normalize_tensor(tensor, epsilon=1e-7):
    """
    TensorFlow equivalent of normalize_tensor. Divides by sum over last axis.
    Input shape: [B, N, D]
    Output shape: [B, N, D]
    """
    sum_tensor = tf.reduce_sum(tensor, axis=-1, keepdims=True)
    # Add epsilon to prevent division by zero
    normalized_tensor = tensor / (sum_tensor + epsilon)
    return normalized_tensor

def tf_anomaly_score(dist_list_1, dist_list_2, win_size, training=True, temp=1.0, w_de=True, epsilon=1e-7):
    """
    TensorFlow equivalent of anomaly_score.
    Processes lists of distributions/mixers.
    """
    total_loss_1 = tf.constant(0.0, dtype=tf.float32)
    # total_loss_2 = tf.constant(0.0, dtype=tf.float32)
    num_items = len(dist_list_1)

    if num_items == 0 or num_items != len(dist_list_2):
        tf.print("Warning: tf_anomaly_score received empty or mismatched lists.")
        return tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)

    for i in range(num_items):
        dist_1 = dist_list_1[i] # Shape e.g., [B, N, D] or [B, P, D]
        dist_2 = dist_list_2[i] # Shape e.g., [B, P, D] or [B, N, D]

        # --- Upsampling / Repeat logic ---
        # This needs to match the PyTorch 'repeat' logic based on expected shapes.
        # Assuming dist_1/dist_2 have shape [B, current_len, D] and need to be tiled to win_size.
        shape1 = tf.shape(dist_1)
        shape2 = tf.shape(dist_2)
        current_len1 = shape1[1]
        current_len2 = shape2[1]

        # --- Normalization (like PyTorch normalize_tensor) ---
        # Applied *after* repeat, matching PyTorch logic.
        # This assumes the inputs dist_1/dist_2 were already probability-like.
        dist_1_norm = tf_normalize_tensor(dist_1, epsilon)
        dist_2_norm = tf_normalize_tensor(dist_2, epsilon)

        # --- Calculate Loss ---
        loss_1, loss_2 = tf_inter_intra_dist(dist_1_norm, dist_2_norm, w_de, training, temp, epsilon)

        # pdb.set_trace()  # Debugging point to check values
        total_loss_1 += loss_1-loss_2
        # total_loss_2 += loss_2

    # Average loss over the number of items in the list? PyTorch divides patch_num_loss by len().
    # Let's return the sums for now, averaging can happen in the main loss function.
    # Or average here if PyTorch logic implies it for the combined score.
    # PyTorch example divides patch_num_loss/patch_size_loss by len() *before* subtraction.
    if num_items > 0:
        avg_loss_1 = total_loss_1 / tf.cast(num_items, tf.float32)
        # avg_loss_2 = total_loss_2 / tf.cast(num_items, tf.float32)
    else:
        avg_loss_1 = tf.constant(0.0, dtype=tf.float32)
        # avg_loss_2 = tf.constant(0.0, dtype=tf.float32)


    return avg_loss_1#, avg_loss_2 # Return averaged losses

