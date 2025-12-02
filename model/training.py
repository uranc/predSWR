"""Tensorflow utility functions for training"""
import logging
import os
import pdb
import tensorflow as tf
from tensorflow.keras import callbacks as cb
from tensorflow.keras import backend as K
import numpy as np
import optuna

def lr_scheduler(epoch, lr):
    min_lr = 1e-4
    max_lr = 1e-3
    cycle_length = 15  # Number of epochs per cycle
    cycle = np.floor(1 + epoch / (2 * cycle_length))
    x = np.abs(epoch / cycle_length - 2 * cycle + 1)
    new_lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1 - x))
    return new_lr

def get_weight_decay(epoch, max_epochs=200):
    """Calculate weight decay based on epoch
    Start small and gradually increase to avoid early overfitting"""
    initial_decay = 1e-6
    final_decay = 1e-4
    warmup_epochs = 100  # No weight decay during first 100 epochs

    # Return zero weight decay during warmup period
    if epoch < warmup_epochs:
        return 0.0

    # After warmup, scale from initial to final decay
    remaining_epochs = max_epochs - warmup_epochs
    progress = min((epoch - warmup_epochs) / (remaining_epochs * 0.8), 1.0)
    return initial_decay + (final_decay - initial_decay) * progress

class TerminateOnNaN(cb.Callback):
    """Callback that terminates training when a NaN loss is encountered."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            # Check if loss is NaN or infinity
            if np.isnan(loss) or np.isinf(loss):
                print(f"\nEpoch {epoch+1}: Invalid loss detected (NaN or Inf). Terminating training.")
                self.model.stop_training = True
                
# Custom callback for updating prototypes
# based on the grid sampling
class GridEMA(tf.keras.callbacks.Callback):
    def __init__(self, model, sampler, alpha=0.98,
                 warmup=3, min_count=20):
        super().__init__()
        self.m   = model
        self.sampler = sampler     # returns (z_sub, bin_id) 5 k points
        self.alpha   = alpha
        self.warmup  = warmup
        self.min_cnt = min_count

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.warmup: return
        z_sub, bid = self.sampler(self.model.encoder)        # [5k,D], [5k]
        K = self.m.prototypes.shape[0]
        centroid = tf.math.unsorted_segment_mean(z_sub, bid, K)
        counts   = tf.math.unsorted_segment_sum(
                        tf.ones_like(bid, tf.float32), bid, K)
        mask = (counts[:,None] >= self.min_cnt)
        new  = (self.alpha * self.m.prototypes +
                (1.-self.alpha) * tf.where(mask, centroid,
                                            self.m.prototypes))
        self.m.prototypes.assign(new)
                        
# Custom callback for weight decay
class WeightDecayCallback(cb.Callback):
    def __init__(self, max_epochs=100):
        super(WeightDecayCallback, self).__init__()
        self.max_epochs = max_epochs

    def on_epoch_begin(self, epoch, logs=None):
        decay = get_weight_decay(epoch, self.max_epochs)
        self.model.optimizer.weight_decay = tf.cast(decay, dtype=tf.float32)

def train_pred(model,
               train_inputs,
               valid_inputs,
               n_epoch,
               save_dir,
               steps_per_epoch=None,
               checkpoint_metric='val_sample_pr_auc',
               dataset_params=None):
    """Train the model on `num_steps` batches
    Args:
        model: (Keras Model) contains the graph
        num_steps: (int) train for this number of batches
        current_epoch: (Params) hyperparameters
    """
    # Check if this is a triplet model based on input/output structure
    # Original training code for non-triplet models
    callbacks = []

    callbacks.append(cb.EarlyStopping(monitor='val_loss',
                                    min_delta=0.0001,
                                    patience=30,  # Adjusted patience
                                    verbose=1))

    # callbacks.append(WeightDecayCallback())

    callbacks.append(cb.TensorBoard(log_dir=save_dir,
                                    write_graph=True,
                                    write_images=True,
                                    update_freq='epoch'))

    callbacks.append(cb.ModelCheckpoint(save_dir + '/event.weights.h5',
                                        monitor=checkpoint_metric,
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='max',
                                        save_freq="epoch"))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if steps_per_epoch is not None:
        # Set steps_per_epoch in model.fit() if provided
        hist = model.fit(train_inputs,
                        epochs=n_epoch,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=valid_inputs,
                        validation_steps=steps_per_epoch,
                        callbacks=callbacks,
                        verbose=1)
    else:
        hist = model.fit(train_inputs,
                        epochs=n_epoch,
                        validation_data=valid_inputs,
                        callbacks=callbacks,
                        verbose=1)
    return hist
