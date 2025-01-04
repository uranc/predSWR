import inspect
from typing import List  # noqa

import tensorflow as tf
# pylint: disable=E0611,E0401
from tensorflow.keras import backend as K, Model, Input, optimizers
# pylint: disable=E0611,E0401
from tensorflow.keras import layers
# pylint: disable=E0611,E0401
from tensorflow.keras.layers import Activation, SpatialDropout1D, Lambda
# pylint: disable=E0611,E0401
from tensorflow.keras.layers import Layer, Conv1D, Dense, BatchNormalization, LayerNormalization
from tensorflow.keras.layers import GroupNormalization, UpSampling1D, AveragePooling1D, MultiHeadAttention

import numpy as np

# Add after the imports
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D

class DeformableConv1D(Layer):
    """Deformable 1D Convolution Layer"""
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(DeformableConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        
        # Offset conv learns the deformation for each filter position
        self.offset_conv = Conv1D(
            filters=kernel_size,
            kernel_size=kernel_size,
            padding='same',
            use_bias=False
        )
        
        # Main convolution
        self.conv = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            use_bias=True
        )

    def build(self, input_shape):
        super(DeformableConv1D, self).build(input_shape)

    def call(self, x):
        # Generate offsets
        offsets = self.offset_conv(x)
        
        # Create sampling grid
        batch_size = tf.shape(x)[0]
        input_length = tf.shape(x)[1]
        
        # Generate basic sampling coordinates
        coords = tf.range(input_length, dtype=tf.float32)
        coords = tf.expand_dims(coords, 0)
        coords = tf.tile(coords, [batch_size, 1])
        
        # Add learned offsets to the base sampling coordinates
        coords_deformed = coords + offsets * self.dilation_rate
        
        # Ensure coordinates stay within bounds
        coords_deformed = tf.clip_by_value(coords_deformed, 0, tf.cast(input_length-1, tf.float32))
        
        # Convert to integers for gathering
        coords_deformed = tf.cast(coords_deformed, tf.int32)
        
        # Gather features using deformed coordinates
        batch_idx = tf.range(batch_size)
        batch_idx = tf.reshape(batch_idx, (-1, 1))
        batch_idx = tf.tile(batch_idx, [1, input_length])
        
        indices = tf.stack([batch_idx, coords_deformed], axis=-1)
        x_deformed = tf.gather_nd(x, indices)
        
        # Apply convolution on deformed features
        return self.conv(x_deformed)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

class DeformableTCNBlock(Layer):
    """Deformable TCN block that replaces standard dilated convolutions"""
    def __init__(self,
                 nb_filters,
                 kernel_size,
                 dilation_rate,
                 dropout_rate=0.0,
                 use_batch_norm=False,
                 activation='relu',
                 **kwargs):
        super(DeformableTCNBlock, self).__init__(**kwargs)
        
        self.deformable_conv1 = DeformableConv1D(
            filters=nb_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate
        )
        
        self.deformable_conv2 = DeformableConv1D(
            filters=nb_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate
        )
        
        self.dropout1 = SpatialDropout1D(dropout_rate)
        self.dropout2 = SpatialDropout1D(dropout_rate)
        self.activation = Activation(activation)
        self.use_batch_norm = use_batch_norm
        
        if use_batch_norm:
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()

    def call(self, inputs, training=None):
        x = self.deformable_conv1(inputs)
        if self.use_batch_norm:
            x = self.bn1(x, training=training)
        x = self.activation(x)
        x = self.dropout1(x, training=training)
        
        x = self.deformable_conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x, training=training)
        x = self.activation(x)
        x = self.dropout2(x, training=training)
        
        # Add skip connection if input and output shapes match
        if inputs.shape[-1] == x.shape[-1]:
            return layers.add([inputs, x])
        return x


def is_power_of_two(num: int):
    return num != 0 and ((num & (num - 1)) == 0)

# Add these new dilation pattern functions below the existing is_power_of_two function
def get_exponential_dilations(num_layers, base=2):
    """Returns exponential dilation sequence: [1, base, base^2, ...]"""
    return [base ** i for i in range(num_layers)]

def get_fibonacci_dilations(num_layers):
    """Returns Fibonacci-based dilation sequence: [1, 2, 3, 5, 8, 13, ...]"""
    dilations = [1, 2]
    while len(dilations) < num_layers:
        dilations.append(dilations[-1] + dilations[-2])
    return dilations

def get_linear_dilations(num_layers, step=2):
    """Returns linear dilation sequence: [1, 1+step, 1+2*step, ...]"""
    return [1 + i * step for i in range(num_layers)]

def get_hybrid_dilations(num_layers):
    """Returns hybrid dilation mixing different patterns"""
    # First half exponential, second half linear
    mid = num_layers // 2
    exp_dilations = get_exponential_dilations(mid)
    linear_dilations = get_linear_dilations(num_layers - mid, step=exp_dilations[-1])
    return exp_dilations + linear_dilations

def get_pyramid_dilations(num_layers):
    """Returns pyramid-style dilations that increase then decrease
    Example: [1,2,4,8,4,2] for num_layers=6"""
    mid = num_layers // 2
    ascending = get_exponential_dilations(mid)
    if num_layers % 2 == 0:
        descending = ascending[-2::-1]  # Exclude peak for even length
    else:
        descending = ascending[::-1]  # Include peak for odd length
    return ascending + descending

def get_smoothed_dilations(num_layers, smooth_factor=0.5):
    """Returns dilations with smoothing at higher rates to reduce artifacts
    Example: [1,2,4,6,8,8] for num_layers=6"""
    standard = get_exponential_dilations(num_layers)
    for i in range(num_layers // 2, num_layers):
        standard[i] = int(standard[i] * smooth_factor + standard[i-1] * (1-smooth_factor))
    return standard

def get_aspp_dilations(num_layers):
    """Inspired by DeepLab's ASPP - parallel dilated convolutions
    Example: [1,2,4,8,8,8] for deeper layers"""
    dilations = get_exponential_dilations(num_layers)
    # Keep maximum dilation rate fixed for last few layers
    max_dilation = dilations[-3]  # Limit maximum dilation
    dilations[-2:] = [max_dilation] * 2
    return dilations


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
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_weight_norm: bool = False,
                 batch_dropout_rate: float = 0.0,
                 **kwargs):
        """Defines the residual block for the WaveNet TCN
        Args:
            x: The previous layer in the model
            training: boolean indicating whether the layer should behave in training mode or in inference mode
            dilation_rate: The dilation power of 2 we are using for this residual block
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            padding: The padding used in the convolutional layers, 'same' or 'causal'.
            activation: The final activation used in o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            use_weight_norm: Whether to use weight normalization in the residual layers or not.
            batch_dropout_rate: Float between 0 and 1. Fraction of the input units to drop at batch level.
            kwargs: Any initializers for Layer class.
        """

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
        self.layers = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)

    def _build_layer(self, layer):
        """Helper function for building layer
        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.
        """
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):

        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):  # dilated conv block.
                name = 'conv1D_{}'.format(k)
                with K.name_scope(name):  # name scope used to make sure weights get unique names
                    conv = MaskedConv1D(
                        filters=self.nb_filters,
                        kernel_size=self.kernel_size,
                        dilation_rate=self.dilation_rate,
                        padding=self.padding,
                        name=name,
                        kernel_initializer=self.kernel_initializer
                    )
                    if self.use_weight_norm:
                        from tensorflow_addons.layers import WeightNormalization
                        # wrap it. WeightNormalization API is different than BatchNormalization or LayerNormalization.
                        with K.name_scope('norm_{}'.format(k)):
                            conv = WeightNormalization(conv)
                    self._build_layer(conv)

                with K.name_scope('norm_{}'.format(k)):
                    if self.use_batch_norm:
                        self._build_layer(BatchNormalization())
                    elif self.use_layer_norm:
                        self._build_layer(LayerNormalization())
                    elif self.use_weight_norm:
                        pass  # done above.

                with K.name_scope('act_and_dropout_{}'.format(k)):
                    self._build_layer(Activation(self.activation, name='Act_Conv1D_{}'.format(k)))
                    self._build_layer(SpatialDropout1D(rate=self.dropout_rate, name='SDropout_{}'.format(k)))

            if self.nb_filters != input_shape[-1]:
                # 1x1 conv to match the shapes (channel dimension).
                name = 'matching_conv1D'
                with K.name_scope(name):
                    # make and build this layer separately because it directly uses input_shape.
                    # 1x1 conv.
                    self.shape_match_conv = MaskedConv1D(
                        filters=self.nb_filters,
                        kernel_size=1,
                        padding='same',
                        name=name,
                        kernel_initializer=self.kernel_initializer
                    )
            else:
                name = 'matching_identity'
                self.shape_match_conv = Lambda(lambda x: x, name=name)

            with K.name_scope(name):
                self.shape_match_conv.build(input_shape)
                self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self._build_layer(Activation(self.activation, name='Act_Conv_Blocks'))
            self.final_activation = Activation(self.activation, name='Act_Res_Block')
            self.final_activation.build(self.res_output_shape)  # probably isn't necessary

            # this is done to force Keras to add the layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)
            self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
            self.__setattr__(self.final_activation.name, self.final_activation)

            super(ResidualBlock, self).build(input_shape)  # done to make sure self.built is set True

    def call(self, inputs, training=None, mask=None, **kwargs):
        """
        Returns: A tuple where the first element is the residual model tensor, and the second
                 is the skip connection tensor.
        """
        # https://arxiv.org/pdf/1803.01271.pdf  page 4, Figure 1 (b).
        # x1: Dilated Conv -> Norm -> Dropout (x2).
        # x2: Residual (1x1 matching conv - optional).
        # Output: x1 + x2.
        # x1 -> connected to skip connections.
        # x1 + x2 -> connected to the next block.
        #       input
        #     x1      x2
        #   conv1D    1x1 Conv1D (optional)
        #    ...
        #   conv1D
        #    ...
        #       x1 + x2
        x1 = inputs

        for layer in self.layers:
            if isinstance(layer, MaskedConv1D):
                x1 = layer(x1, mask=mask)
            else:
                training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
                x1 = layer(x1, training=training) if training_flag else layer(x1)
        x2 = self.shape_match_conv(inputs)
        x1_x2 = self.final_activation(layers.add([x2, x1], name='Add_Res'))
        return [x1_x2, x1]

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]


class TCN(Layer):
    """Creates a TCN layer.

        Input shape:
            A 3D tensor with shape (batch_size, timesteps, input_dim).

        Args:
            nb_filters: The number of filters to use in the convolutional layers. Can be a list.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual blocK.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            activation: The activation used in the residual blocks o = Activation(x + F(x)).
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            use_weight_norm: Whether to use weight normalization in the residual layers or not.
            go_backwards: Boolean (default False). If True, process the input sequence backwards and
            return the reversed sequence.
            return_state: Boolean. Whether to return the last state in addition to the output. Default: False.
            kwargs: Any other arguments for configuring parent class Layer. For example "name=str", Name of the model.
                    Use unique names when using multiple TCN.
        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=3,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal',
                 dilation_pattern='pyramid',  # Changed default
                 reduce_gridding=True,  # New parameter
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='relu',
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False,
                 go_backwards=False,
                 return_state=False,
                 use_deformable=False,  # New parameter
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
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.go_backwards = go_backwards
        self.return_state = return_state
        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None
        self.slicer_layer = None  # in case return_sequence=False
        self.output_slice_index = None  # in case return_sequence=False
        self.padding_same_and_time_dim_unknown = False  # edge case if padding='same' and time_dim = None
        self.use_deformable = use_deformable

        if self.use_batch_norm + self.use_layer_norm + self.use_weight_norm > 1:
            raise ValueError('Only one normalization can be specified at once.')

        if isinstance(self.nb_filters, list):
            assert len(self.nb_filters) == len(self.dilations)
            if len(set(self.nb_filters)) > 1 and self.use_skip_connections:
                raise ValueError('Skip connections are not compatible '
                                 'with a list of filters, unless they are all equal.')

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        # Set dilations based on pattern if not explicitly provided
        if isinstance(dilations, str):
            num_layers = 6  # Default, can be made configurable
            if dilation_pattern == 'exponential':
                self.dilations = get_exponential_dilations(num_layers)
            elif dilation_pattern == 'fibonacci':
                self.dilations = get_fibonacci_dilations(num_layers)
            elif dilation_pattern == 'linear':
                self.dilations = get_linear_dilations(num_layers)
            elif dilation_pattern == 'hybrid':
                self.dilations = get_hybrid_dilations(num_layers)
            elif dilation_pattern == 'pyramid':
                self.dilations = get_pyramid_dilations(num_layers)
            elif dilation_pattern == 'smoothed':
                self.dilations = get_smoothed_dilations(num_layers)
            elif dilation_pattern == 'aspp':
                self.dilations = get_aspp_dilations(num_layers)
        else:
            self.dilations = dilations

        if reduce_gridding:
            # Add additional convolution layers with smaller dilation at the end
            self.use_gridding_reduction = True
            self.final_conv1 = Conv1D(nb_filters, 3, dilation_rate=2, padding=padding)
            self.final_conv2 = Conv1D(nb_filters, 3, dilation_rate=1, padding=padding)
        else:
            self.use_gridding_reduction = False

        # initialize parent class
        super(TCN, self).__init__(**kwargs)

    @property
    def receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * self.nb_stacks * sum(self.dilations)

    def tolist(self, shape):
        try:
            return shape.as_list()
        except AttributeError:
            return shape

    def build(self, input_shape):

        # member to hold current output shape of the layer for building purposes
        self.build_output_shape = input_shape

        # list to hold all the member ResidualBlocks
        self.residual_blocks = []
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1  # cheap way to do a false case for below

        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                if self.use_deformable:
                    self.residual_blocks.append(
                        DeformableTCNBlock(
                            nb_filters=self.nb_filters,
                            kernel_size=self.kernel_size,
                            dilation_rate=d,
                            dropout_rate=self.dropout_rate,
                            use_batch_norm=self.use_batch_norm,
                            activation=self.activation_name
                        )
                    )
                else:
                    res_block_filters = self.nb_filters[i] if isinstance(self.nb_filters, list) else self.nb_filters
                    self.residual_blocks.append(ResidualBlock(dilation_rate=d,
                                                              nb_filters=res_block_filters,
                                                              kernel_size=self.kernel_size,
                                                              padding=self.padding,
                                                              activation=self.activation_name,
                                                              dropout_rate=self.dropout_rate,
                                                              use_batch_norm=self.use_batch_norm,
                                                              use_layer_norm=self.use_layer_norm,
                                                              use_weight_norm=self.use_weight_norm,
                                                              kernel_initializer=self.kernel_initializer,
                                                              name='residual_block_{}'.format(len(self.residual_blocks))))
                # build newest residual block
                self.residual_blocks[-1].build(self.build_output_shape)
                self.build_output_shape = self.residual_blocks[-1].res_output_shape

        # this is done to force keras to add the layers in the list to self._layers
        for layer in self.residual_blocks:
            self.__setattr__(layer.name, layer)

        self.output_slice_index = None
        if self.padding == 'same':
            time = self.tolist(self.build_output_shape)[1]
            if time is not None:  # if time dimension is defined. e.g. shape = (bs, 500, input_dim).
                self.output_slice_index = int(self.tolist(self.build_output_shape)[1] / 2)
            else:
                # It will known at call time. c.f. self.call.
                self.padding_same_and_time_dim_unknown = True

        else:
            self.output_slice_index = -1  # causal case.
        self.slicer_layer = Lambda(lambda tt: tt[:, self.output_slice_index, :], name='Slice_Output')
        self.slicer_layer.build(self.tolist(self.build_output_shape))

    def compute_output_shape(self, input_shape):
        """
        Overridden in case keras uses it somewhere... no idea. Just trying to avoid future errors.
        """
        if not self.built:
            self.build(input_shape)
        if not self.return_sequences:
            batch_size = self.build_output_shape[0]
            batch_size = batch_size.value if hasattr(batch_size, 'value') else batch_size
            nb_filters = self.build_output_shape[-1]
            return [batch_size, nb_filters]
        else:
            # Compatibility tensorflow 1.x
            return [v.value if hasattr(v, 'value') else v for v in self.build_output_shape]

    def call(self, inputs, training=None, mask=None, **kwargs):
        x = inputs

        if self.go_backwards:
            # reverse x in the time axis
            x = tf.reverse(x, axis=[1])

        self.layers_outputs = [x]
        self.skip_connections = []
        for res_block in self.residual_blocks:
            try:
                x, skip_out = res_block(x, training=training, mask=mask)
            except TypeError:  # compatibility with tensorflow 1.x
                x, skip_out = res_block(K.cast(x, 'float32'), training=training, mask=mask)
            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)

        if self.use_skip_connections:
            if len(self.skip_connections) > 1:
                # Keras: A merge layer should be called on a list of at least 2 inputs. Got 1 input.
                x = layers.add(self.skip_connections, name='Add_Skip_Connections')
            else:
                x = self.skip_connections[0]
            self.layers_outputs.append(x)

        if self.use_gridding_reduction:
            # Apply additional convolutions to reduce gridding artifacts
            x = self.final_conv1(x)
            x = self.final_conv2(x)

        if not self.return_sequences:
            # case: time dimension is unknown. e.g. (bs, None, input_dim).
            if self.padding_same_and_time_dim_unknown:
                self.output_slice_index = K.shape(self.layers_outputs[-1])[1] // 2
            x = self.slicer_layer(x)
            self.layers_outputs.append(x)
        return x

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(TCN, self).get_config()
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['dilations'] = self.dilations
        config['padding'] = self.padding
        config['use_skip_connections'] = self.use_skip_connections
        config['dropout_rate'] = self.dropout_rate
        config['return_sequences'] = self.return_sequences
        config['activation'] = self.activation_name
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['use_weight_norm'] = self.use_weight_norm
        config['kernel_initializer'] = self.kernel_initializer
        config['go_backwards'] = self.go_backwards
        config['return_state'] = self.return_state
        config['use_deformable'] = self.use_deformable
        return config


def compiled_tcn(num_feat,  # type: int
                 num_classes,  # type: int
                 nb_filters,  # type: int
                 kernel_size,  # type: int
                 dilations,  # type: List[int]
                 nb_stacks,  # type: int
                 max_len,  # type: int
                 output_len=1,  # type: int
                 padding='causal',  # type: str
                 use_skip_connections=False,  # type: bool
                 return_sequences=True,
                 regression=False,  # type: bool
                 dropout_rate=0.05,  # type: float
                 name='tcn',  # type: str,
                 kernel_initializer='he_normal',  # type: str,
                 activation='relu',  # type:str,
                 opt='adam',
                 lr=0.002,
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False,
                 use_deformable=False):  # New parameter
    # type: (...) -> Model
    """Creates a compiled TCN model for a given task (i.e. regression or classification).
    Classification uses a sparse categorical loss. Please input class ids and not one-hot encodings.

    Args:
        num_feat: The number of features of your input, i.e. the last dimension of: (batch_size, timesteps, input_dim).
        num_classes: The size of the final dense layer, how many classes we are predicting.
        nb_filters: The number of filters to use in the convolutional layers.
        kernel_size: The size of the kernel to use in each convolutional layer.
        dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
        nb_stacks : The number of stacks of residual blocks to use.
        max_len: The maximum sequence length, use None if the sequence length is dynamic.
        padding: The padding to use in the convolutional layers.
        use_skip_connections: Boolean. If we want to add skip connections from input to each residual blocK.
        return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
        regression: Whether the output should be continuous or discrete.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        activation: The activation used in the residual blocks o = Activation(x + F(x)).
        name: Name of the model. Useful when having multiple TCN.
        kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
        opt: Optimizer name.
        lr: Learning rate.
        use_batch_norm: Whether to use batch normalization in the residual layers or not.
        use_layer_norm: Whether to use layer normalization in the residual layers or not.
        use_weight_norm: Whether to use weight normalization in the residual layers or not.
        use_deformable: Whether to use deformable convolutions in the residual layers or not.
    Returns:
        A compiled keras TCN.
    """

    dilations = adjust_dilations(dilations)

    input_layer = Input(shape=(max_len, num_feat))

    x = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
            use_skip_connections, dropout_rate, return_sequences,
            activation, kernel_initializer, use_batch_norm, use_layer_norm,
            use_weight_norm, use_deformable=use_deformable, name=name)(input_layer)

    print('x.shape=', x.shape)

    def get_opt():
        if opt == 'adam':
            return optimizers.Adam(lr=lr, clipnorm=1.)
        elif opt == 'rmsprop':
            return optimizers.RMSprop(lr=lr, clipnorm=1.)
        else:
            raise Exception('Only Adam and RMSProp are available here')

    if not regression:
        # classification
        x = Dense(num_classes)(x)
        x = Activation('softmax')(x)
        output_layer = x
        model = Model(input_layer, output_layer)

        # https://github.com/keras-team/keras/pull/11373
        # It's now in Keras@master but still not available with pip.
        # TODO remove later.
        def accuracy(y_true, y_pred):
            # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
            if K.ndim(y_true) == K.ndim(y_pred):
                y_true = K.squeeze(y_true, -1)
            # convert dense predictions to labels
            y_pred_labels = K.argmax(y_pred, axis=-1)
            y_pred_labels = K.cast(y_pred_labels, K.floatx())
            return K.cast(K.equal(y_true, y_pred_labels), K.floatx())

        model.compile(get_opt(), loss='sparse_categorical_crossentropy', metrics=[accuracy])
    else:
        # regression
        x = Dense(output_len)(x)
        x = Activation('linear')(x)
        output_layer = x
        model = Model(input_layer, output_layer)
        model.compile(get_opt(), loss='mean_squared_error')
    print('model.x = {}'.format(input_layer.shape))
    print('model.y = {}'.format(output_layer.shape))
    return model


def tcn_full_summary(model: Model, expand_residual_blocks=True):
    import tensorflow as tf
    # 2.6.0-rc1, 2.5.0...
    versions = [int(v) for v in tf.__version__.split('-')[0].split('.')]
    if versions[0] <= 2 and versions[1] < 5:
        layers = model._layers.copy()  # store existing layers
        model._layers.clear()  # clear layers

        for i in range(len(layers)):
            if isinstance(layers[i], TCN):
                for layer in layers[i]._layers:
                    if not isinstance(layer, ResidualBlock):
                        if not hasattr(layer, '__iter__'):
                            model._layers.append(layer)
                    else:
                        if expand_residual_blocks:
                            for lyr in layer._layers:
                                if not hasattr(lyr, '__iter__'):
                                    model._layers.append(lyr)
                        else:
                            model._layers.append(layer)
            else:
                model._layers.append(layers[i])

        model.summary()  # print summary

        # restore original layers
        model._layers.clear()
        [model._layers.append(lyr) for lyr in layers]
    else:
        print('WARNING: tcn_full_summary: Compatible with tensorflow 2.5.0 or below.')
        print('Use tensorboard instead. Example in keras-tcn/tasks/tcn_tensorboard.py.')


# Add these new layers for PatchAD-inspired features
class PatchEmbedding(Layer):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = Conv1D(embed_dim, kernel_size=patch_size, strides=patch_size)
        
    def call(self, x):
        # Convert signal into patches and embed
        x = self.proj(x)
        return x

class MemoryBank(Layer):
    def __init__(self, memory_size, feature_dim):
        super().__init__()
        self.memory_size = memory_size
        self.memory = self.add_weight(
            shape=(memory_size, feature_dim),
            initializer="random_normal",
            trainable=True,
            name='memory_bank'
        )
        
    def call(self, query_features):
        # Compute similarity with memory items
        similarity = tf.matmul(query_features, self.memory, transpose_b=True)
        attention = tf.nn.softmax(similarity, axis=-1)
        # Retrieve memory features
        memory_features = tf.matmul(attention, self.memory)
        return memory_features, attention

class LocalGlobalInteraction(Layer):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.local_attn = MultiHeadAttention(num_heads, dim//num_heads)
        self.global_attn = MultiHeadAttention(num_heads, dim//num_heads)
        self.fusion = Conv1D(dim, 1)
        
    def call(self, x):
        # Local attention within temporal window
        local_features = self.local_attn(x, x)
        
        # Global attention across entire sequence
        global_features = self.global_attn(
            tf.reduce_mean(x, axis=1, keepdims=True), x)
        
        # Fuse local and global features
        return self.fusion(local_features + global_features)

# Modify TCN to incorporate PatchAD ideas
def build_patch_tcn(input_shape, params):
    inputs = Input(shape=input_shape)
    
    # Patch embedding
    patch_size = params.get('patch_size', 16)
    embed_dim = params.get('embed_dim', 128)
    x = PatchEmbedding(patch_size, embed_dim)(inputs)
    
    # TCN backbone
    tcn = TCN(
        nb_filters=embed_dim,
        kernel_size=params['kernel_size'],
        dilations=[2**i for i in range(params['num_layers'])],
        use_skip_connections=True
    )(x)
    
    # Local-global feature interaction
    x = LocalGlobalInteraction(embed_dim)(tcn)
    
    # Memory bank for pattern comparison
    memory_bank = MemoryBank(
        memory_size=params.get('memory_size', 1024),
        feature_dim=embed_dim
    )
    memory_features, attention_weights = memory_bank(x)
    
    # Fuse memory features with current features
    x = Concatenate()([x, memory_features])
    
    # Prediction heads
    reconstruction = Dense(input_shape[-1], activation='linear')(x)
    anomaly_score = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=[reconstruction, anomaly_score])
    
    # Add contrastive loss
    def contrastive_loss(memory_features, current_features, temperature=0.07):
        # Normalize features
        memory_features = tf.nn.l2_normalize(memory_features, axis=-1)
        current_features = tf.nn.l2_normalize(current_features, axis=-1)
        
        # Compute similarity matrix
        similarity = tf.matmul(current_features, memory_features, transpose_b=True)
        
        # Contrastive loss
        positive_loss = -tf.reduce_mean(
            tf.nn.log_softmax(similarity / temperature, axis=-1))
        
        return positive_loss
    
    # Custom training step
    @tf.function
    def train_step(data):
        with tf.GradientTape() as tape:
            reconstruction, anomaly_score = model(data)
            
            # Reconstruction loss
            recon_loss = tf.reduce_mean(
                tf.square(data - reconstruction))
            
            # Contrastive loss between memory and current features
            cont_loss = contrastive_loss(memory_features, x)
            
            # Total loss
            total_loss = recon_loss + params.get('lambda_contrast', 0.1) * cont_loss
            
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return total_loss

    return model


# Add these new layer classes
class GatedResidualBlock(Layer):
    def __init__(self, filters):
        super(GatedResidualBlock, self).__init__()
        self.gate = Conv1D(filters, 1, activation='sigmoid')
        self.projection = Conv1D(filters, 1)
        
    def call(self, x, residual):
        gate_values = self.gate(x)
        projected = self.projection(residual)
        return x + (gate_values * projected)

class RelativePositionalEncoding(Layer):
    def __init__(self, max_position, d_model):
        super(RelativePositionalEncoding, self).__init__()
        self.rel_pos_emb = self.add_weight(
            shape=(max_position * 2 + 1, d_model),
            initializer='random_normal',
            trainable=True,
            name='rel_pos_emb'
        )
        
    def call(self, length):
        pos_indices = tf.range(length)
        relative_positions = pos_indices[None, :] - pos_indices[:, None]
        relative_positions += length
        return tf.gather(self.rel_pos_emb, relative_positions)

class AxialAttention(Layer):
    def __init__(self, dim, num_heads=8):
        super(AxialAttention, self).__init__()
        self.temporal_attention = MultiHeadAttention(num_heads, dim)
        self.channel_attention = MultiHeadAttention(num_heads, dim)
        
    def call(self, x):
        # Temporal attention
        temporal = self.temporal_attention(x, x)
        # Channel attention
        x_t = tf.transpose(x, [0, 2, 1])
        channel = self.channel_attention(x_t, x_t)
        channel = tf.transpose(channel, [0, 2, 1])
        return temporal + channel

# Modify the build_DBI_TCN_Dorizon function
def build_DBI_TCN_Dorizon(input_timepoints, input_chans=8, params=None):
    # ...existing code until TCN definition...

    if model_type=='Base':
        from tcn import TCN
        
        # Add relative positional encoding
        rel_pos_encoding = RelativePositionalEncoding(input_timepoints, n_filters)
        
        # First TCN branch
        tcn_op = TCN(
            # ...existing TCN parameters...
        )
        
        # Second TCN branch for hierarchical features
        tcn_hierarchical = TCN(
            nb_filters=n_filters//2,
            kernel_size=n_kernels*2,
            dilations=[4 ** i for i in range(n_dilations)],
            # ...rest of parameters same as tcn_op
        )

        # Main flow
        nets = tcn_op(inputs_nets)
        hierarchical_features = tcn_hierarchical(inputs_nets)

        # Add axial attention
        axial_attention = AxialAttention(n_filters)
        nets = axial_attention(nets)
        
        # Gated residual connection between branches
        gated_residual = GatedResidualBlock(n_filters)
        nets = gated_residual(nets, hierarchical_features)

        # Add group normalization (as an alternative to batch/layer norm)
        nets = tf.keras.layers.GroupNormalization(groups=8)(nets)

        # Feature pyramid skip connections
        pyramid_features = []
        for i in range(3):  # 3 levels of pyramid
            pyramid_conv = Conv1D(n_filters//(2**i), kernel_size=3, padding='same')
            pyramid_features.append(pyramid_conv(nets))
            nets = AveragePooling1D(2)(nets)
        
        # Multi-scale feature fusion
        upsampled_features = []
        for i, feat in enumerate(pyramid_features):
            if i > 0:
                feat = UpSampling1D(2**i)(feat)
            upsampled_features.append(feat)
        
        nets = Concatenate(axis=-1)(upsampled_features)

        # ...rest of your existing code...

    return model


# Add these new layer classes
class CrossChannelAttention(Layer):
    def __init__(self, dim):
        super().__init__()
        self.attention = MultiHeadAttention(4, dim//4)
        self.norm = LayerNormalization()
        
    def call(self, x):
        # Transpose to handle channel interactions
        x_t = tf.transpose(x, [0, 2, 1])
        attn = self.attention(x_t, x_t)
        attn = tf.transpose(attn, [0, 2, 1])
        return self.norm(x + attn)

class AdaptiveInstanceNorm(Layer):
    def __init__(self):
        super().__init__()
        
    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        std = tf.math.reduce_std(x, axis=[1, 2], keepdims=True) + 1e-5
        return (x - mean) / std

# Add to build_patch_tcn after the TCN backbone
x = CrossChannelAttention(embed_dim)(tcn)
x = AdaptiveInstanceNorm()(x)


# Example usage
params = {
    'patch_size': 16,
    'embed_dim': 128, 
    'memory_size': 1024,
    'kernel_size': 3,
    'num_layers': 4,
    'lambda_contrast': 0.1
}

model = build_patch_tcn(input_shape=(None, 8), params=params)

