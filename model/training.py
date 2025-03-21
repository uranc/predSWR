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
               checkpoint_metric='val_max_f1_metric_horizon',
               dataset_params=None):
    """Train the model on `num_steps` batches
    Args:
        model: (Keras Model) contains the graph
        num_steps: (int) train for this number of batches
        current_epoch: (Params) hyperparameters
    """
    # Check if this is a triplet model based on input/output structure
    is_triplet_model = 'TripletOnly' in dataset_params['TYPE_ARCH']

    # For TripletOnly models, use the specialized training function
    if is_triplet_model:

        # Early stopping parameters
        best_metric = float('-inf')
        patience = 30
        min_delta = 0.0001
        patience_counter = 0
        
        # Create a list to collect history from each epoch
        history_list = []
        
        # Setup callbacks including the verifier
        callbacks = [
            cb.TensorBoard(log_dir=save_dir, write_graph=True, write_images=True, update_freq='epoch'),
            cb.ModelCheckpoint(save_dir + '/last.weights.h5',
                             monitor=checkpoint_metric, verbose=1,
                             save_best_only=True, save_weights_only=True,
                             mode='max')
        ]
        
        # Loop through epochs manually
        for epoch in range(n_epoch):
            print(f"\nEpoch {epoch+1}/{n_epoch}")
            if dataset_params is not None and 'triplet_regenerator' in dataset_params:
                regenerating_dataset = dataset_params['triplet_regenerator']
                print(f"Regenerating triplet samples for epoch {epoch+1}")
                
                if epoch > 0:
                    regenerating_dataset.reinitialize()
                train_data = regenerating_dataset.dataset if hasattr(regenerating_dataset, 'dataset') else regenerating_dataset
 
                steps = dataset_params.get('steps_per_epoch', 500)
                # pdb.set_trace()
                epoch_history = model.fit(train_data,
                    steps_per_epoch=steps,
                    initial_epoch=epoch,
                    epochs=epoch+1,
                    validation_data=valid_inputs,
                    callbacks=callbacks,
                    verbose=1
                )

            # Collect history
            history_list.append(epoch_history.history)
                        
            # Early stopping check after each epoch
            current_metric = epoch_history.history.get(checkpoint_metric, [float('-inf')])[0]
            if current_metric > (best_metric + min_delta):
                best_metric = current_metric
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                break
            

        # Combine histories from all epochs
        combined_history = {}
        for key in history_list[0].keys():
            combined_history[key] = []
            for h in history_list:
                combined_history[key].extend(h[key])
        hist = combined_history
    else:    
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

        callbacks.append(cb.ModelCheckpoint(save_dir + '/last.weights.h5',
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
                            callbacks=callbacks,
                            verbose=1)
        else:
            hist = model.fit(train_inputs,
                            epochs=n_epoch,
                            validation_data=valid_inputs,
                            callbacks=callbacks,
                            verbose=1)
    return hist
