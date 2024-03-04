from tensorflow.keras import backend as K
from tensorflow.keras.layers import Cropping2D
import tensorflow as tf
import numpy as np


def loss_fn_pred():
    """Loss function"""
    def loss(y_true, y_pred):
        print(y_true.shape)
        print(y_pred.shape)
        return K.sum(K.abs(y_pred-y_true))
