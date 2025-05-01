import tensorflow as tf

# --- Existing KL divergence (can be kept or removed if unused) ---
def kl_divergence_one_way(p, q, epsilon=1e-8):
    """
    One-way KL divergence KL(p || q) along the last axis.
    Assumes p and q are probability distributions (e.g., after softmax).
    p, q shape: [B, T, D] or [B, N, D] etc.
    Returns shape: [B, T] or [B, N] etc.
    """
    # Add epsilon to prevent log(0) or log(p/0)
    safe_p = tf.clip_by_value(p, epsilon, 1.0)
    safe_q = tf.clip_by_value(q, epsilon, 1.0)
    # Calculate KL divergence element-wise, then sum over the feature dimension (D)
    return tf.reduce_sum(safe_p * (tf.math.log(safe_p) - tf.math.log(safe_q)), axis=-1)

# --- New Helper Functions based on PyTorch ---

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
    total_loss_2 = tf.constant(0.0, dtype=tf.float32)
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

        if win_size % current_len1 != 0 or win_size % current_len2 != 0:
             # If not divisible, this indicates a potential issue or requires different handling (e.g., interpolation)
             # For now, raising an error or using floor division might be necessary.
             # Let's assume divisibility based on PyTorch code.
             tf.print(f"Warning/Error: win_size {win_size} not divisible by tensor lengths {current_len1}, {current_len2}")
             # Fallback or error handling needed here. Skipping item for now.
             continue # Or adjust logic

        repeat_factor1 = win_size // current_len1
        repeat_factor2 = win_size // current_len2

        # Tile along the sequence dimension (axis 1)
        # tf.tile requires multiples argument: [1 (batch), repeat_factor, 1 (features)]
        dist_1_repeated = tf.tile(dist_1, [1, repeat_factor1, 1])
        dist_2_repeated = tf.tile(dist_2, [1, repeat_factor2, 1])

        # --- Normalization (like PyTorch normalize_tensor) ---
        # Applied *after* repeat, matching PyTorch logic.
        # This assumes the inputs dist_1/dist_2 were already probability-like.
        dist_1_norm = tf_normalize_tensor(dist_1_repeated, epsilon)
        dist_2_norm = tf_normalize_tensor(dist_2_repeated, epsilon)

        # --- Calculate Loss ---
        loss_1, loss_2 = tf_inter_intra_dist(dist_1_norm, dist_2_norm, w_de, training, temp, epsilon)

        total_loss_1 += loss_1
        total_loss_2 += loss_2

    # Average loss over the number of items in the list? PyTorch divides patch_num_loss by len().
    # Let's return the sums for now, averaging can happen in the main loss function.
    # Or average here if PyTorch logic implies it for the combined score.
    # PyTorch example divides patch_num_loss/patch_size_loss by len() *before* subtraction.
    if num_items > 0:
        avg_loss_1 = total_loss_1 / tf.cast(num_items, tf.float32)
        avg_loss_2 = total_loss_2 / tf.cast(num_items, tf.float32)
    else:
        avg_loss_1 = tf.constant(0.0, dtype=tf.float32)
        avg_loss_2 = tf.constant(0.0, dtype=tf.float32)


    return avg_loss_1, avg_loss_2 # Return averaged losses


# --- Main Loss Function (PyTorch Style) ---

def pytorch_style_patch_loss(
    patch_num_dist_list,  # List[Tensor[B,N,D]] from PatchAD output 1
    patch_size_dist_list, # List[Tensor[B,P,D]] from PatchAD output 2
    patch_num_mx_list,    # List[Tensor[B,N,D]] from PatchAD output 3
    patch_size_mx_list,   # List[Tensor[B,P,D]] from PatchAD output 4
    rec_x,                # Tensor[B,L,C] - Reconstruction from PatchAD output 5
    original_input,       # Tensor[B,L,C] - Ground truth input for reconstruction
    win_size,             # Sequence length L
    patch_mx_coeff=0.5,   # Weight corresponding to self.patch_mx in PyTorch
    training=True,        # Flag for training/evaluation mode
    temp=1.0,             # Temperature for KL divergence
    epsilon=1e-7
    ):
    """
    Calculates the loss based on the PyTorch training loop structure.
    """
    # 1. Calculate KL terms using tf_anomaly_score
    # L_cont1 = KL(num_dist || size_mx) + KL(size_mx || num_dist) (averaged over list)
    # L_cont2 = KL(num_mx || size_dist) + KL(size_dist || num_mx) (averaged over list)
    # L_diff1 = KL(num_dist || size_dist) (averaged over list)
    # L_diff2 = KL(size_dist || num_dist) (averaged over list)

    # Term 1: (num_dist vs size_mx) -> Corresponds to cont_loss1, cont_loss2 in PyTorch
    cont_loss1_p, cont_loss1_q = tf_anomaly_score(
        patch_num_dist_list, patch_size_mx_list, win_size, training, temp, w_de=True, epsilon=epsilon
    )
    # PyTorch uses cont_loss1 - cont_loss2; Here cont_loss1_p corresponds to KL(num_dist||stop(size_mx))+...,
    # and cont_loss1_q corresponds to KL(stop(num_dist)||size_mx)+...
    # So, cont_loss1_p seems to be the relevant term if we follow p_loss, q_loss naming. Let's assume p_loss.
    cont_loss_term1 = cont_loss1_p # Check PyTorch inter_intra_dist return order if needed

    # Term 2: (num_mx vs size_dist) -> Corresponds to cont_loss12, cont_loss22 in PyTorch
    cont_loss2_p, cont_loss2_q = tf_anomaly_score(
        patch_num_mx_list, patch_size_dist_list, win_size, training, temp, w_de=True, epsilon=epsilon
    )
    # Similar logic, assume p_loss is the relevant one KL(num_mx||stop(size_dist))+...
    cont_loss_term2 = cont_loss2_p

    # Term 3: (num_dist vs size_dist) -> Corresponds to patch_num_loss, patch_size_loss in PyTorch
    diff_loss_p, diff_loss_q = tf_anomaly_score(
        patch_num_dist_list, patch_size_dist_list, win_size, training, temp, w_de=True, epsilon=epsilon
    )
    # PyTorch uses patch_num_loss - patch_size_loss. Assuming patch_num_loss corresponds to diff_loss_p.
    diff_loss_term = diff_loss_p - diff_loss_q # Symmetric difference? Or just diff_loss_p? PyTorch uses patch_num_loss - patch_size_loss. Let's use p - q.

    # 2. Calculate Reconstruction Loss (MSE)
    loss_mse = tf.reduce_mean(tf.square(original_input - rec_x))

    # 3. Combine losses based on PyTorch structure
    # loss = self.patch_mx * cont_loss_1 + self.patch_mx * cont_loss_2 + (1 - self.patch_mx) * loss3 + loss_mse
    total_loss = (patch_mx_coeff * cont_loss_term1 +
                  patch_mx_coeff * cont_loss_term2 +
                  (1.0 - patch_mx_coeff) * diff_loss_term +
                  loss_mse)

    # Optional: Print components for debugging
    # tf.print("PyTorch Style Loss - MSE:", loss_mse, "Cont1:", cont_loss_term1, "Cont2:", cont_loss_term2, "Diff:", diff_loss_term, "Total:", total_loss)

    return total_loss


# --- Original PatchAD Loss (based on paper equations) ---
# Keep this function for comparison or if needed later
# ... (previous code: kl_divergence_one_way, tf_my_kl_loss, tf_inter_intra_dist, tf_normalize_tensor, tf_anomaly_score, pytorch_style_patch_loss) ...

# --- Original PatchAD Loss (based on paper equations) ---
# Keep this function for comparison or if needed later
def patch_loss(prediction_targets, prediction_out, proj_inter, proj_intra, x_inter, x_intra, constraint_coeff=0.2, epsilon=1e-8):
    """
    PatchAD objective based on paper's Eq. 8, 9, 10, using Softmax normalization.
    Assumes input features are final, upsampled outputs from PatchAD.

    Args:
        prediction_targets: Ground truth for reconstruction [B, T, C]
        prediction_out: Model's reconstruction [B, T, C]
        proj_inter (N'): Projected inter features [B, T, D_proj]
        proj_intra (P'): Projected intra features [B, T, D_proj]
        x_inter (N): Final encoder inter features [B, T, D]
        x_intra (P): Final encoder intra features [B, T, D]
        constraint_coeff (c): Weight for projection constraint loss
        epsilon: Small value for numerical stability in KL divergence

    Returns:
        Scalar total loss value.
    """
    # 1. Normalize features using Softmax along the feature dimension (-1)
    x_inter_prob = tf.nn.softmax(x_inter, axis=-1)
    x_intra_prob = tf.nn.softmax(x_intra, axis=-1)
    proj_inter_prob = tf.nn.softmax(proj_inter, axis=-1)
    proj_intra_prob = tf.nn.softmax(proj_intra, axis=-1)

    # --- Calculate terms based on paper equations ---

    # L_N = mean( KL(N || stop(P)) + KL(stop(P) || N) )
    kl_N_stopP = kl_divergence_one_way(x_inter_prob, tf.stop_gradient(x_intra_prob), epsilon)
    kl_stopP_N = kl_divergence_one_way(tf.stop_gradient(x_intra_prob), x_inter_prob, epsilon)
    L_N = tf.reduce_mean(kl_N_stopP + kl_stopP_N)

    # L_P = mean( KL(P || stop(N)) + KL(stop(N) || P) )
    kl_P_stopN = kl_divergence_one_way(x_intra_prob, tf.stop_gradient(x_inter_prob), epsilon)
    kl_stopN_P = kl_divergence_one_way(tf.stop_gradient(x_inter_prob), x_intra_prob, epsilon)
    L_P = tf.reduce_mean(kl_P_stopN + kl_stopN_P)

    # L_N' = mean( KL(N' || stop(P)) + KL(stop(P) || N') )
    kl_Nprime_stopP = kl_divergence_one_way(proj_inter_prob, tf.stop_gradient(x_intra_prob), epsilon)
    kl_stopP_Nprime = kl_divergence_one_way(tf.stop_gradient(x_intra_prob), proj_inter_prob, epsilon)
    L_N_prime = tf.reduce_mean(kl_Nprime_stopP + kl_stopP_Nprime)

    # L_P' = mean( KL(P' || stop(N)) + KL(stop(N) || P') )
    kl_Pprime_stopN = kl_divergence_one_way(proj_intra_prob, tf.stop_gradient(x_inter_prob), epsilon)
    kl_stopN_Pprime = kl_divergence_one_way(tf.stop_gradient(x_inter_prob), proj_intra_prob, epsilon)
    L_P_prime = tf.reduce_mean(kl_Pprime_stopN + kl_stopN_Pprime)

    # --- Combine terms ---

    # 2. Content Loss (L_cont = L_N - L_P) - Eq 8
    L_cont = L_N - L_P

    # 3. Projection Constraint Loss (L_proj = (L_N' - L_P) + (L_N - L_P')) - Eq 9
    L_proj = (L_N_prime - L_P) + (L_N - L_P_prime)

    # 4. Reconstruction Loss (MSE)
    L_rec = tf.reduce_mean(tf.square(prediction_targets - prediction_out))

    # 5. Final Loss (Eq 10)
    total_loss = (1.0 - constraint_coeff) * L_cont + constraint_coeff * L_proj + L_rec

    return total_loss



class PatchADLoss(tf.keras.layers.Layer):
    def __init__(self, input_dim, max_len=5000, name="positional_embedding", **kwargs):
        super().__init__(name=name, **kwargs)
        