import tensorflow as tf
import tensorflow as tf
import numpy as np

def tversky_loss(alpha=0.5, beta=0.5):
    def tversky_fn(y_true, y_pred, alpha=0.5, beta=0.5, gamma=0.75):
        """Computes the Tversky loss value between `y_true` and `y_pred`.

        This loss function is weighted by the alpha and beta coefficients
        that penalize false positives and false negatives.

        With `alpha=0.5` and `beta=0.5`, the loss value becomes equivalent to
        Dice Loss.

        Args:
            y_true: tensor of true targets.
            y_pred: tensor of predicted targets.
            alpha: coefficient controlling incidence of false positives.
            beta: coefficient controlling incidence of false negatives.

        Returns:
            Tversky loss value.

        Reference:

        - [Salehi et al., 2017](https://arxiv.org/abs/1706.05721)
        """
        
        inputs = tf.keras.backend.reshape(y_true, [-1])
        targets = tf.keras.backend.reshape(y_pred, [-1])

        intersection = tf.reduce_sum(inputs * targets)
        fp = tf.reduce_sum((1 - targets) * inputs)
        fn = tf.reduce_sum(targets * (1 - inputs))
        tversky = tf.divide(
            intersection,
            intersection + fp * alpha + fn * beta + tf.keras.backend.epsilon(),
        )
        return 1 - tversky
    return tversky_fn


def constrained_dtw_distance(pred, target, max_warping_window=5):
    n, m = len(pred), len(target)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(max(1, i - max_warping_window), min(m + 1, i + max_warping_window)):
            cost = (pred[i-1] - target[j-1]) ** 2
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                          dtw_matrix[i, j-1],    # deletion
                                          dtw_matrix[i-1, j-1])  # match
    return dtw_matrix[n, m]

def compute_early_penalty_and_reward(pred, target):
    penalty = 0.0
    reward = 0.0
    for i in range(len(pred)):
        if target[i] == 1:  # Only consider times when the event occurs
            event_index = np.argmax(pred > 0.5)  # Index where the prediction exceeds threshold
            if event_index > i:
                penalty += (event_index - i) ** 2
            elif event_index < i:
                reward += (i - event_index) ** 2
    return penalty, reward

@tf.function
def early_prediction_dtw_loss(y_true, y_pred, penalty_factor=1.0, reward_factor=0.5):
    batch_size = tf.shape(y_pred)[0]
    loss = 0.0
    for i in range(batch_size):
        pred_seq = y_pred[i]
        target_seq = y_true[i]
        dtw_cost = constrained_dtw_distance(pred_seq, target_seq)
        early_penalty, early_reward = compute_early_penalty_and_reward(pred_seq.numpy(), target_seq.numpy())
        loss += dtw_cost + penalty_factor * early_penalty - reward_factor * early_reward
    return loss / tf.cast(batch_size, tf.float32)