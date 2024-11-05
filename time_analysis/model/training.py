"""Tensorflow utility functions for training"""
import logging
import os
import pdb
import tensorflow as tf
from tensorflow.keras import callbacks as cb
from tensorflow.keras import backend as K

def train_pred(model,
               train_inputs,
               valid_inputs,
               n_epoch,
               save_dir):
    """Train the model on `num_steps` batches
    Args:
        model: (Keras Model) contains the graph
        num_steps: (int) train for this number of batches
        current_epoch: (Params) hyperparameters
    """
    callbacks = []
    # callbacks.append(cb.ReduceLROnPlateau(monitor='val_loss',
    #                                       factor=0.5,
    #                                       patience=20,
    #                                       verbose=1,
    #                                       mode='auto',
    #                                       min_delta=0.0001,
    #                                       cooldown=10,
    #                                       min_lr=1e-5))
    # callbacks.append(cb.TensorBoard(log_dir=save_dir,
    #                                   write_graph=True,
    #                                   write_images=True,
    #                                   update_freq='epoch'))
    callbacks.append(cb.TensorBoard(log_dir=save_dir,
                                      write_graph=True,
                                      write_images=True,
                                      update_freq='epoch'))
    callbacks.append(cb.ModelCheckpoint(save_dir + '/weights.last.h5',
                                        monitor='val_binary_accuracy', # val_ssim
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=False,
                                        mode='auto',
                                        save_freq="epoch"))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return model.fit(train_inputs,
                     epochs=n_epoch,
                     validation_data=valid_inputs,
                     callbacks=callbacks,
                     verbose=1)                     #  steps_per_epoch=100, # batch=256,
