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
    callbacks.append(cb.ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.5,
                                          patience=20,
                                          verbose=1,
                                          mode='auto',
                                          min_delta=0.00001,
                                          cooldown=5,
                                          min_lr=1e-5))
    callbacks.append(cb.EarlyStopping(monitor='val_loss',
                                        min_delta=0.00001,
                                        patience=50,
                                        verbose=1))
    # callbacks.append(cb.TensorBoard(log_dir=save_dir,
    #                                   write_graph=True,
    #                                   write_images=True,
    #                                   update_freq='epoch'))
    # callbacks.append(cb.LearningRateScheduler(lr_scheduler))
    callbacks.append(cb.TensorBoard(log_dir=save_dir,
                                      write_graph=True,
                                      write_images=True,
                                      update_freq='epoch'))
    # callbacks.append(cb.ModelCheckpoint(save_dir + '/weights.last.keras',
    callbacks.append(cb.ModelCheckpoint(save_dir + '/weights.last.h5',
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
                     validation_data=valid_inputs,
                     callbacks=callbacks,
                     verbose=1)
