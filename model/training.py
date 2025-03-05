"""Tensorflow utility functions for training"""
import logging
import os
import pdb
import tensorflow as tf
from tensorflow.keras import callbacks as cb
from tensorflow.keras import backend as K
import numpy as np

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

class MultiObjectivePruningCallback(tf.keras.callbacks.Callback):
    """Callback for Optuna to prune unpromising trials for multi-objective optimization.
    
    This callback is designed to work with multi-objective studies where trial.report()
    is not supported. Instead, it tracks the monitored metrics and prunes the trial
    if the performance doesn't improve for a specified number of epochs.
    """
    
    def __init__(self, trial, monitor='val_loss', patience=10, greater_is_better=False):
        """Initialize the callback.
        
        Args:
            trial: A trial corresponding to the current optimization trial.
            monitor: The metric to monitor for pruning decisions.
            patience: Number of epochs with no improvement before pruning.
            greater_is_better: Whether higher values of the metric are better.
        """
        super(MultiObjectivePruningCallback, self).__init__()
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
            return
            
        # Check if we need to prune the trial
        improved = False
        
        if self.greater_is_better:
            if current_value > self.best_value:
                self.best_value = current_value
                improved = True
        else:
            if current_value < self.best_value:
                self.best_value = current_value
                improved = True
                
        if improved:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                self.trial.storage.set_trial_state(self.trial._trial_id, optuna.trial.TrialState.PRUNED)


class OptunaPruningCallback(tf.keras.callbacks.Callback):
    def __init__(self, trial, monitor=['val_max_f1_metric_horizon', 'val_latency_metric']):
        super(OptunaPruningCallback, self).__init__()
        self.trial = trial
        # Support monitoring multiple metrics for multi-objective optimization
        self.monitor = monitor if isinstance(monitor, list) else [monitor]
        self.best_values = {m: float('-inf') for m in self.monitor}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Calculate combined objective for pruning
        current_values = []
        for metric in self.monitor:
            value = logs.get(metric)
            if value is None:
                return
            
            # For latency metric, we want to minimize it
            if 'latency' in metric:
                value = -value  # Convert to maximization problem
            
            current_values.append(value)
            self.trial.set_user_attr(f"best_{metric}", max(self.best_values[metric], value))
            self.best_values[metric] = max(self.best_values[metric], value)

        # For multi-objective optimization, report all values
        if len(current_values) > 1:
            self.trial.report(current_values, epoch)
        else:
            self.trial.report(current_values[0], epoch)
            
        # Prune if trial should be pruned
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial pruned at epoch {epoch}")
        
        
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
