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
    # Reach max decay at 80% of typical training length (80 epochs if max_epochs=100)
    progress = min(epoch / (max_epochs * 0.8), 1.0)
    return initial_decay + (final_decay - initial_decay) * progress

# Custom callback for weight decay
class WeightDecayCallback(cb.Callback):
    def __init__(self, max_epochs=100):
        super(WeightDecayCallback, self).__init__()
        self.max_epochs = max_epochs
        
    def on_epoch_begin(self, epoch, logs=None):
        decay = get_weight_decay(epoch, self.max_epochs)
        self.model.optimizer.weight_decay = tf.cast(decay, dtype=tf.float32)


class F1PruningCallback(tf.keras.callbacks.Callback):
    """
    A pruning callback for multi-objective studies that prunes based solely on the F1 metric.
    
    Since trial.report is not supported for multi-objective studies in Optuna,
    this callback uses its own internal logic to track the monitored metric (e.g. validation F1)
    and raises a TrialPruned exception if no improvement is observed for a specified number of epochs.
    """
    def __init__(self, trial, monitor='val_max_f1_metric_horizon', patience=30, greater_is_better=True):
        """
        Args:
            trial: The Optuna trial object.
            monitor: Name of the metric to monitor from training logs.
            patience: Number of epochs to wait for improvement before pruning.
            greater_is_better: True if a higher metric is better.
        """
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.patience = patience
        self.greater_is_better = greater_is_better
        self.best_value = -np.inf if greater_is_better else np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_value = logs.get(self.monitor)
        if current_value is None:
            return  # Metric not available in logs.

        # Check if there is improvement.
        if self.greater_is_better:
            if current_value > self.best_value:
                self.best_value = current_value
                self.wait = 0
            else:
                self.wait += 1
        else:
            if current_value < self.best_value:
                self.best_value = current_value
                self.wait = 0
            else:
                self.wait += 1

        if self.wait >= self.patience:
            self.model.stop_training = True
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {epoch}: no improvement in {self.monitor} "
                f"(best: {self.best_value}, current: {current_value}) for {self.patience} epochs."
            )

def train_pred(model,
               train_inputs,
               valid_inputs,
               n_epoch,
               save_dir,
               checkpoint_metric='val_max_f1_metric_horizon'):
    """Train the model on `num_steps` batches
    Args:
        model: (Keras Model) contains the graph
        num_steps: (int) train for this number of batches
        current_epoch: (Params) hyperparameters
    """
    callbacks = []
    
    # # Update ReduceLROnPlateau for better convergence
    # callbacks.append(cb.ReduceLROnPlateau(monitor='val_loss',
    #                                     factor=0.2,  # More aggressive reduction
    #                                     patience=15, # Reduced patience
    #                                     verbose=1,
    #                                     mode='auto',
    #                                     min_delta=0.0001,
    #                                     cooldown=3,
    #                                     min_lr=1e-6))

    callbacks.append(cb.EarlyStopping(monitor='val_loss',
                                    min_delta=0.0001,
                                    patience=25,  # Adjusted patience
                                    verbose=1))
    
    # callbacks.append(WeightDecayCallback())
    
    callbacks.append(cb.TensorBoard(log_dir=save_dir,
                                      write_graph=True,
                                      write_images=True,
                                      update_freq='epoch'))
    
    
    # callbacks.append(OptunaPruningCallback(
    #     trial,
    #     monitor='val_max_f1_metric_horizon'  # Primary metric we want to optimize
    # ))
    # callbacks.append(cb.ModelCheckpoint(save_dir + '/weights.last.keras',
    callbacks.append(cb.ModelCheckpoint(save_dir + '/last.weights.h5',
                                        # monitor='val_custom_binary_accuracy', # val_ssim
                                        # monitor='val_loss',
                                        monitor=checkpoint_metric,
                                        # monitor='val_binary_accuracy', # val_ssim
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='max',
                                        save_freq="epoch"))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return model.fit(train_inputs,
                     epochs=n_epoch,
                    #  steps_per_epoch=4,
                     validation_data=valid_inputs,
                     callbacks=callbacks,
                     verbose=1)
