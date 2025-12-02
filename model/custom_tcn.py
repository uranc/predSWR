import inspect
from typing import List

import tensorflow as tf
from tensorflow.keras import backend as K, Model, Input, optimizers
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, SpatialDropout1D, Lambda
from tensorflow.keras.layers import Layer, Conv1D, Dense, BatchNormalization, LayerNormalization, Add


def is_power_of_two(num: int):
    return num != 0 and ((num & (num - 1)) == 0)


def adjust_dilations(dilations: list):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations


class ResidualBlock(Layer):

    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 padding: str,
                 activation: str = 'relu',
                 dropout_rate: float = 0,
                 kernel_initializer: str = 'he_normal',
                 kernel_regularizer=None,
                 data_format='channels_first', # Default to NVIDIA Optimized format
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_weight_norm: bool = False,
                 **kwargs):

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.data_format = data_format
        
        # Determine channel axis based on data_format
        self.channel_axis = 1 if self.data_format == 'channels_first' else -1
        
        self.layers = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)

    def _build_layer(self, layer):
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):

        with K.name_scope(self.name):
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):
                name = 'conv1D_{}'.format(k)
                with K.name_scope(name):
                    conv = Conv1D(
                        filters=self.nb_filters,
                        kernel_size=self.kernel_size,
                        dilation_rate=self.dilation_rate,
                        padding=self.padding,
                        name=name,
                        data_format=self.data_format, # Respect N-C-W
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer
                    )
                    if self.use_weight_norm:
                        try:
                            from tensorflow_addons.layers import WeightNormalization
                            with K.name_scope('norm_{}'.format(k)):
                                conv = WeightNormalization(conv)
                        except ImportError:
                            print("TF Addons not found, skipping WeightNorm")
                            
                    self._build_layer(conv)

                with K.name_scope('norm_{}'.format(k)):
                    if self.use_batch_norm:
                        # BN needs to know the axis too
                        self._build_layer(BatchNormalization(axis=self.channel_axis))
                    elif self.use_layer_norm:
                        self._build_layer(LayerNormalization(axis=self.channel_axis))

                with K.name_scope('act_and_dropout_{}'.format(k)):
                    self._build_layer(Activation(self.activation, name='Act_Conv1D_{}'.format(k)))
                    self._build_layer(SpatialDropout1D(rate=self.dropout_rate, name='SDropout_{}'.format(k)))

            # FIX: Correctly identify channel dimension
            if self.nb_filters != input_shape[self.channel_axis]:
                name = 'matching_conv1D'
                with K.name_scope(name):
                    self.shape_match_conv = Conv1D(
                        filters=self.nb_filters,
                        kernel_size=1,
                        padding='same',
                        name=name,
                        data_format=self.data_format,
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer
                    )
            else:
                name = 'matching_identity'
                self.shape_match_conv = Lambda(lambda x: x, name=name)

            with K.name_scope(name):
                self.shape_match_conv.build(input_shape)
                self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self._build_layer(Activation(self.activation, name='Act_Conv_Blocks'))
            self.final_activation = Activation(self.activation, name='Act_Res_Block')
            self.final_activation.build(self.res_output_shape)

            for layer in self.layers:
                self.__setattr__(layer.name, layer)
            self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
            self.__setattr__(self.final_activation.name, self.final_activation)

            super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        x1 = inputs
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x1 = layer(x1, training=training) if training_flag else layer(x1)
        x2 = self.shape_match_conv(inputs)
        x1_x2 = self.final_activation(layers.add([x2, x1], name='Add_Res'))
        return [x1_x2, x1]

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]


class TCN(Layer):
    def __init__(self,
                 nb_filters=64,
                 kernel_size=3,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='relu',
                 data_format='channels_first', # Default N-C-W
                 kernel_initializer='he_normal',
                 kernel_regularizer=None,
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False,
                 go_backwards=False,
                 return_state=False,
                 **kwargs):

        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation_name = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.go_backwards = go_backwards
        self.return_state = return_state
        self.data_format = data_format
        
        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None
        self.slicer_layer = None
        self.output_slice_index = None
        self.padding_same_and_time_dim_unknown = False

        if self.use_batch_norm + self.use_layer_norm + self.use_weight_norm > 1:
            raise ValueError('Only one normalization can be specified at once.')

        super(TCN, self).__init__(**kwargs)

    @property
    def receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * self.nb_stacks * sum(self.dilations)

    def build(self, input_shape):
        self.build_output_shape = input_shape
        self.residual_blocks = []
        
        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                res_block_filters = self.nb_filters[i] if isinstance(self.nb_filters, list) else self.nb_filters
                self.residual_blocks.append(ResidualBlock(dilation_rate=d,
                                                          nb_filters=res_block_filters,
                                                          kernel_size=self.kernel_size,
                                                          padding=self.padding,
                                                          activation=self.activation_name,
                                                          dropout_rate=self.dropout_rate,
                                                          data_format=self.data_format, # Pass format
                                                          use_batch_norm=self.use_batch_norm,
                                                          use_layer_norm=self.use_layer_norm,
                                                          use_weight_norm=self.use_weight_norm,
                                                          kernel_initializer=self.kernel_initializer,
                                                          kernel_regularizer=self.kernel_regularizer,
                                                          name='residual_block_{}'.format(len(self.residual_blocks))))
                self.residual_blocks[-1].build(self.build_output_shape)
                self.build_output_shape = self.residual_blocks[-1].res_output_shape

        for layer in self.residual_blocks:
            self.__setattr__(layer.name, layer)

        # FIX: Find correct time dimension based on data_format
        # channels_last = (Batch, Time, Feat) -> Time is index 1
        # channels_first = (Batch, Feat, Time) -> Time is index 2
        time_axis = 2 if self.data_format == 'channels_first' else 1
        
        self.output_slice_index = None
        if self.padding == 'same':
            time_dim = self.build_output_shape.as_list()[time_axis]
            if time_dim is not None:
                self.output_slice_index = int(time_dim / 2)
            else:
                self.padding_same_and_time_dim_unknown = True
        else:
            self.output_slice_index = -1

        # FIX: Slicer lambda must handle the correct axis
        if self.data_format == 'channels_first':
            # Slice last element of Time (axis 2) -> returns (Batch, Channels)
            self.slicer_layer = Lambda(lambda tt: tt[:, :, self.output_slice_index], name='Slice_Output')
        else:
            # Slice last element of Time (axis 1) -> returns (Batch, Channels)
            self.slicer_layer = Lambda(lambda tt: tt[:, self.output_slice_index, :], name='Slice_Output')
            
        self.slicer_layer.build(self.build_output_shape.as_list())

    def compute_output_shape(self, input_shape):
        if not self.built:
            self.build(input_shape)
        if not self.return_sequences:
            batch_size = self.build_output_shape[0]
            # Filters (Features) dimension location depends on format
            feat_idx = 1 if self.data_format == 'channels_first' else -1
            nb_filters = self.build_output_shape[feat_idx]
            return [batch_size, nb_filters]
        else:
            return [v.value if hasattr(v, 'value') else v for v in self.build_output_shape]

    def call(self, inputs, training=None, **kwargs):
        x = inputs

        if self.go_backwards:
            # FIX: Reverse the correct Time axis
            time_axis = 2 if self.data_format == 'channels_first' else 1
            x = tf.reverse(x, axis=[time_axis])

        self.layers_outputs = [x]
        self.skip_connections = []
        for res_block in self.residual_blocks:
            try:
                x, skip_out = res_block(x, training=training)
            except TypeError:
                x, skip_out = res_block(K.cast(x, 'float32'), training=training)
            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)

        if self.use_skip_connections:
            if len(self.skip_connections) > 1:
                x = layers.add(self.skip_connections, name='Add_Skip_Connections')
            else:
                x = self.skip_connections[0]
            self.layers_outputs.append(x)

        if not self.return_sequences:
            # Handle unknown time dim case if needed
            if self.padding_same_and_time_dim_unknown:
                time_axis = 2 if self.data_format == 'channels_first' else 1
                self.output_slice_index = K.shape(self.layers_outputs[-1])[time_axis] // 2
            x = self.slicer_layer(x)
            self.layers_outputs.append(x)
        return x

    def get_config(self):
        config = super(TCN, self).get_config()
        config.update({
            'data_format': self.data_format, # Save the format!
            'nb_filters': self.nb_filters,
            'kernel_size': self.kernel_size,
            'nb_stacks': self.nb_stacks,
            'dilations': self.dilations,
            'padding': self.padding,
            'use_skip_connections': self.use_skip_connections,
            'dropout_rate': self.dropout_rate,
            'return_sequences': self.return_sequences,
            'activation': self.activation_name,
            'use_batch_norm': self.use_batch_norm,
            'use_layer_norm': self.use_layer_norm,
            'use_weight_norm': self.use_weight_norm,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'go_backwards': self.go_backwards,
            'return_state': self.return_state
        })
        return config


def compiled_tcn(num_feat,
                 num_classes,
                 nb_filters,
                 kernel_size,
                 dilations,
                 nb_stacks,
                 max_len,
                 output_len=1,
                 padding='causal',
                 use_skip_connections=False,
                 return_sequences=True,
                 regression=False,
                 dropout_rate=0.05,
                 name='tcn',
                 kernel_initializer='he_normal',
                 kernel_regularizer=None,
                 activation='relu',
                 opt='adam',
                 lr=0.002,
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False,
                 data_format='channels_first'): # Default to Fast

    dilations = adjust_dilations(dilations)

    # FIX: Input Shape Definition based on format
    # Channels Last: (Time, Feat)
    # Channels First: (Feat, Time)
    if data_format == 'channels_first':
        input_layer = Input(shape=(num_feat, max_len)) 
    else:
        input_layer = Input(shape=(max_len, num_feat))

    x = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
            use_skip_connections, dropout_rate, return_sequences,
            activation, data_format, # Pass format
            kernel_initializer, use_batch_norm, use_layer_norm,
            use_weight_norm, name=name)(input_layer)

    def get_opt():
        if opt == 'adam':
            return optimizers.Adam(learning_rate=lr, clipnorm=1.)
        elif opt == 'rmsprop':
            return optimizers.RMSprop(learning_rate=lr, clipnorm=1.)
        else:
            raise Exception('Only Adam and RMSProp are available here')

    if not regression:
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)
        output_layer = x
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    else:
        x = Dense(output_len)(x)
        x = Activation('linear')(x)
        output_layer = x
        loss = 'mean_squared_error'
        metrics = ['mse']

    model = Model(input_layer, output_layer)
    model.compile(get_opt(), loss=loss, metrics=metrics)
    
    return model