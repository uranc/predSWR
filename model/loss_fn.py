import tensorflow as tf

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