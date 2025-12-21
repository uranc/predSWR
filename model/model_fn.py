import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, ELU, Input, LSTM, Dense, Dropout, Layer, MultiHeadAttention, LayerNormalization, Normalization, SeparableConv1D
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, Concatenate, Lambda, RepeatVector, Multiply
from tensorflow.keras.layers import (Input, Lambda, Concatenate, Multiply, Activation,
    Conv1D, Dense, Add, LayerNormalization, DepthwiseConv1D, DepthwiseConv2D, ZeroPadding2D)
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.initializers import GlorotUniform, Orthogonal, Constant
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import applications
from model.patchAD.patch_ad import PatchAD
from tensorflow.keras.layers import Layer, Input, Flatten, Dense, Dropout, RepeatVector, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.activations import gelu 
import pdb, math
import numpy as np


def build_DBI_TCN_MixerOnly(input_timepoints, input_chans=8, params=None):

    # logit oer sigmoid
    flag_sigmoid = 'SigmoidFoc' in params['TYPE_LOSS']
    print('Using Sigmoid:', flag_sigmoid)
    # params
    use_batch_norm = params['TYPE_REG'].find('BN')>-1
    use_weight_norm = params['TYPE_REG'].find('WN')>-1
    use_layer_norm = params['TYPE_REG'].find('LN')>-1
    use_group_norm = params['TYPE_REG'].find('GN')>-1

    # this_activation = 'relu'
    if params['TYPE_REG'].find('RELU')>-1:
        this_activation = 'relu'
    elif params['TYPE_REG'].find('GELU')>-1:
        this_activation = gelu
    elif params['TYPE_REG'].find('ELU')>-1:
        this_activation = ELU(alpha=1)
    else:
        this_activation = 'linear'

    # optimizer
    if params['TYPE_REG'].find('AdamW')>-1:
        # Increase initial learning rate for better exploration
        initial_lr = params['LEARNING_RATE'] * 2.0
        this_optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_lr, clipnorm=1.0)
    elif params['TYPE_REG'].find('Adam')>-1:
        this_optimizer = tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE'])
    elif params['TYPE_REG'].find('SGD')>-1:
        this_optimizer = tf.keras.optimizers.SGD(learning_rate=params['LEARNING_RATE'], momentum=0.9)

    if params['TYPE_REG'].find('Glo')>-1:
        print('Using Glorot')
        this_kernel_initializer = 'glorot_uniform'
    elif params['TYPE_REG'].find('He')>-1:
        print('Using He')
        this_kernel_initializer = 'he_normal'
    model_type = params['TYPE_MODEL']
    n_filters = params['NO_FILTERS']
    n_kernels = params['NO_KERNELS']
    n_dilations = params['NO_DILATIONS']
    if params['TYPE_ARCH'].find('Drop')>-1:
        r_drop = float(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Drop')+4:params['TYPE_ARCH'].find('Drop')+6])/100
        print('Using Dropout')
        print(r_drop)
    else:
        r_drop = 0.0

    hori_shift = 0
    if params['TYPE_ARCH'].find('Only')>-1:
        hori_shift = int(int(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Only')+4:params['TYPE_ARCH'].find('Only')+6])/1000*params['SRATE'])
        print('Using Horizon Timesteps:', hori_shift)
        is_classification_only = True

    print('Using Loss Weight:', params['LOSS_WEIGHT'])
    loss_weight = params['LOSS_WEIGHT']

    inputs = Input(shape=(None, input_chans), name='inputs')

    if params['TYPE_ARCH'].find('ZNorm')>-1:
        print('Using ZNorm')
        # Manual z-score normalization along time axis
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        std = tf.math.reduce_std(inputs, axis=1, keepdims=True) + 1e-6
        inputs = (inputs - mean) / std

    if params['TYPE_ARCH'].find('CSD')>-1:
        csd_inputs = CSDLayer()(inputs)
        inputs_nets = Concatenate(axis=-1)([inputs, csd_inputs])
    else:
        inputs_nets = inputs

    # get TCN
    if model_type=='Base':
        print('Using Base TCN')
        from tcn import TCN

        tcn_op = TCN(nb_filters=n_filters,
                        kernel_size=n_kernels,
                        nb_stacks=1,  # Increase stacks for deeper temporal processing
                        dilations=[2 ** i for i in range(n_dilations)],
                        padding='causal',
                        use_skip_connections=True,
                        dropout_rate=r_drop,
                        # kernel_regularizer=tf.keras.regularizers.l2(1e-4),  # Add L2 regularization
                        return_sequences=True,
                        activation=this_activation,
                        kernel_initializer=this_kernel_initializer,
                        use_batch_norm=use_batch_norm,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
        print(tcn_op.receptive_field)
        nets = tcn_op(inputs_nets)
        nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)
    elif model_type=='SingleCh':
        print('Using Single Channel TCN')
        from tcn import TCN
        tcn_op = TCN(nb_filters=n_filters,
                        kernel_size=n_kernels,
                        nb_stacks=1,
                        dilations=[2 ** i for i in range(n_dilations)],
                        padding='causal',
                        use_skip_connections=True,
                        dropout_rate=r_drop,
                        return_sequences=True,
                        activation=this_activation,
                        kernel_initializer=this_kernel_initializer,
                        use_batch_norm=use_batch_norm,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
        single_outputs = []
        for c in range(input_chans):
            ch_slice = Lambda(lambda x: x[:, :, c:c+1])(inputs)
            single_outputs.append(tcn_op(ch_slice))
        nets = Concatenate(axis=-1)(single_outputs)
        nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)

    if params['TYPE_ARCH'].find('L2N')>-1:
        tcn_output = Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(nets)
    else:
        tcn_output = nets


    if params['TYPE_ARCH'].find('Att')>-1:
        print('Using Attention')
        # anchor_output = self_att_gate(anchor_output, anchor_input)
    # sigmoid out
    if flag_sigmoid:
        tmp_class = Conv1D(1, kernel_size=1, use_bias=True, activation='linear', name='tmp_class')(tcn_output)
    else:
        tmp_class = Conv1D(1, kernel_size=1, use_bias=True, kernel_initializer='glorot_uniform', activation='sigmoid', name='tmp_class')(tcn_output)

    # compute probability
    concat_outputs = tmp_class

    # Define model with both outputs
    if params['mode']=='embedding':
        concat_outputs = Concatenate(axis=-1)([concat_outputs, tcn_output])
    elif params['mode']=='predict':
        concat_outputs = Lambda(lambda tt: tt[:, -1:, :], name='Last_Output')(concat_outputs)

    model = Model(inputs=inputs, outputs=concat_outputs)
    model._is_classification_only = True

    f1_metric = MaxF1MetricHorizon(model=model)
    # r1_metric = RobustF1Metric(model=model)
    # latency_metric = LatencyMetric(model=model)
    event_f1_metric = EventAwareF1(model=model)
    fp_event_metric = EventFalsePositiveRateMetric(model=model)

    # Create loss function without calling it
    loss_fn = custom_fbfce(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model, this_op=tcn_op)

    model.compile(optimizer=this_optimizer,
                  loss=loss_fn,
                  metrics=[f1_metric, event_f1_metric, fp_event_metric])

    if params['WEIGHT_FILE']:
        print('load model')
        model.load_weights(params['WEIGHT_FILE'])

    return model

def build_DBI_TCN_HorizonMixer(input_timepoints, input_chans=8, params=None):

    # logit oer sigmoid
    flag_sigmoid = 'SigmoidFoc' in params['TYPE_LOSS']
    print('Using Sigmoid:', flag_sigmoid)
    # params
    use_batch_norm = params['TYPE_REG'].find('BN')>-1
    use_weight_norm = params['TYPE_REG'].find('WN')>-1
    use_layer_norm = params['TYPE_REG'].find('LN')>-1
    use_group_norm = params['TYPE_REG'].find('GN')>-1

    # this_activation = 'relu'
    if params['TYPE_REG'].find('RELU')>-1:
        this_activation = 'relu'
    elif params['TYPE_REG'].find('GELU')>-1:
        this_activation = gelu
    elif params['TYPE_REG'].find('ELU')>-1:
        this_activation = ELU(alpha=1)
    else:
        this_activation = 'linear'

    # optimizer
    if params['TYPE_REG'].find('AdamW')>-1:
        # Increase initial learning rate for better exploration
        initial_lr = params['LEARNING_RATE'] * 2.0
        this_optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_lr, clipnorm=1.0)
    elif params['TYPE_REG'].find('Adam')>-1:
        this_optimizer = tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE'])
    elif params['TYPE_REG'].find('SGD')>-1:
        this_optimizer = tf.keras.optimizers.SGD(learning_rate=params['LEARNING_RATE'], momentum=0.9)

    if params['TYPE_REG'].find('Glo')>-1:
        print('Using Glorot')
        this_kernel_initializer = 'glorot_uniform'
    elif params['TYPE_REG'].find('He')>-1:
        print('Using He')
        this_kernel_initializer = 'he_normal'

    model_type = params['TYPE_MODEL']
    n_filters = params['NO_FILTERS']
    n_kernels = params['NO_KERNELS']
    n_dilations = params['NO_DILATIONS']
    if params['TYPE_ARCH'].find('Drop')>-1:
        r_drop = float(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Drop')+4:params['TYPE_ARCH'].find('Drop')+6])/100
        print('Using Dropout')
        print(r_drop)
    else:
        r_drop = 0.0

    hori_shift = 0
    if params['TYPE_ARCH'].find('Hori')>-1:
        hori_shift = int(int(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Hori')+4:params['TYPE_ARCH'].find('Hori')+6])/1000*params['SRATE'])
        print('Using Horizon Timesteps:', hori_shift)
        is_classification_only = False

    print('Using Loss Weight:', params['LOSS_WEIGHT'])
    loss_weight = params['LOSS_WEIGHT']

    inputs = Input(shape=(None, input_chans), name='inputs')

    if params['TYPE_ARCH'].find('ZNorm')>-1:
        print('Using ZNorm')
        # Manual z-score normalization along time axis
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        std = tf.math.reduce_std(inputs, axis=1, keepdims=True) + 1e-6
        inputs = (inputs - mean) / std

    if params['TYPE_ARCH'].find('CSD')>-1:
        csd_inputs = CSDLayer()(inputs)
        inputs_nets = Concatenate(axis=-1)([inputs, csd_inputs])
    else:
        inputs_nets = inputs

    # get TCN
    if model_type=='Base':
        print('Using Base TCN')
        from tcn import TCN
        tcn_op = TCN(nb_filters=n_filters,
                        kernel_size=n_kernels,
                        nb_stacks=1,
                        dilations=[2 ** i for i in range(n_dilations)],
                        padding='causal',
                        use_skip_connections=True,
                        dropout_rate=r_drop,
                        return_sequences=True,
                        activation=this_activation,
                        kernel_initializer=this_kernel_initializer,
                        use_batch_norm=use_batch_norm,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
        print(tcn_op.receptive_field)
        nets = tcn_op(inputs_nets)
        nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)
    elif model_type=='SingleCh':
        print('Using Single Channel TCN')
        from tcn import TCN
        tcn_op = TCN(nb_filters=n_filters,
                        kernel_size=n_kernels,
                        nb_stacks=1,
                        dilations=[2 ** i for i in range(n_dilations)],
                        padding='causal',
                        use_skip_connections=True,
                        dropout_rate=r_drop,
                        return_sequences=True,
                        activation=this_activation,
                        kernel_initializer=this_kernel_initializer,
                        use_batch_norm=use_batch_norm,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
        single_outputs = []
        for c in range(input_chans):
            ch_slice = Lambda(lambda x: x[:, :, c:c+1])(inputs)
            single_outputs.append(tcn_op(ch_slice))
        nets = Concatenate(axis=-1)(single_outputs)
        nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)

    if params['TYPE_ARCH'].find('L2N')>-1:
        tcn_output = Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(nets)
    else:
        tcn_output = nets

    output_dim = 8
    # tmp_pred = Dense(32, activation=this_activation, name='tmp_pred')(tcn_output)  # Output future values
    prediction_output = Conv1D(output_dim, kernel_size=1, kernel_initializer=this_kernel_initializer, use_bias=True, activation='linear', name='prediction_output')(tcn_output)  # Output future values
    # pdb.set_trace()
    # prediction_output = Dense(output_dim, kernel_initializer=this_kernel_initializer, activation='linear', name='prediction_output')(tcn_output)  # Output future values

    # sigmoid out
    # tmp_class = Dense(32, activation=this_activation, name='tmp_class')(tcn_output)
    if flag_sigmoid:
        tmp_class = Conv1D(1, kernel_size=1, use_bias=True, activation='linear', name='tmp_class')(tcn_output)
    else:
        tmp_class = Conv1D(1, kernel_size=1, use_bias=True, kernel_initializer='glorot_uniform', activation='sigmoid', name='tmp_class')(tcn_output)
        # tmp_class = Dense(1, kernel_initializer='glorot_uniform', use_bias=True, activation='sigmoid', name='tmp_class')(tcn_output)

    # add confidence layer
    if params['TYPE_ARCH'].find('Confidence')>-1:
        print('Using Confidence Inputs')
        conf_inputs = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Confidence')(inputs)
        confidence = tf.reduce_mean(tf.square(conf_inputs-prediction_output), axis=-1, keepdims=True)
        tmp_class = Concatenate(axis=-1)([tmp_class, confidence])

    # compute probability
    # classification_output = Dense(1, activation='sigmoid', name='classification_output')(tmp_class)
    classification_output = tmp_class
    concat_outputs = Concatenate(axis=-1)([prediction_output, classification_output])
    # concat_outputs = Lambda(lambda tt: tt[:, -50:, :], name='Slice_Output')(concat_outputs)
    # Define model with both outputs

    if params['mode']=='embedding':
        concat_outputs = Concatenate(axis=-1)([concat_outputs, tcn_output])
    elif params['mode']=='predict':
        concat_outputs = Lambda(lambda tt: tt[:, -1:, :], name='Last_Output')(concat_outputs)

    model = Model(inputs=inputs, outputs=concat_outputs)

    # if flag_sigmoid:
    #     # f1_metric = MaxF1MetricHorizonMixer()
    #     # this_binary_accuracy = custom_binary_accuracy_mixer
    #     f1_metric = MaxF1MetricHorizon(is_classification_only=is_classification_only)
    #     r1_metric = RobustF1Metric(is_classification_only=is_classification_only)
    #     this_binary_accuracy = custom_binary_accuracy
    # else:
    #     f1_metric = MaxF1MetricHorizon(is_classification_only=is_classification_only, thresholds=tf.linspace(0.0, 1.0, 11))
    #     r1_metric = RobustF1Metric(is_classification_only=is_classification_only, thresholds=tf.linspace(0.0, 1.0, 11))
    #     latency_metric = LatencyMetric(is_classification_only=is_classification_only, max_early_detection=25)

    f1_metric = MaxF1MetricHorizon(model=model)
    # r1_metric = RobustF1Metric(model=model)
    # latency_metric = LatencyMetric(model=model)
    event_f1_metric = EventAwareF1(model=model)
    fp_event_metric = EventFalsePositiveRateMetric(model=model)

    # Create loss function without calling it
    loss_fn = custom_fbfce(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model, this_op=tcn_op)

    model.compile(optimizer=this_optimizer,
                  loss=loss_fn,
                  metrics=[f1_metric, event_f1_metric, fp_event_metric])

    if params['WEIGHT_FILE']:
        print('load model')
        model.load_weights(params['WEIGHT_FILE'])

    return model

def build_DBI_TCN_DorizonMixer(input_timepoints, input_chans=8, params=None):
    # logit or sigmoid
    flag_sigmoid = 'SigmoidFoc' in params['TYPE_LOSS']
    print('Using Sigmoid:', flag_sigmoid)

    # params
    use_batch_norm = params['TYPE_REG'].find('BN')>-1
    use_weight_norm = params['TYPE_REG'].find('WN')>-1
    use_layer_norm = params['TYPE_REG'].find('LN')>-1
    use_group_norm = params['TYPE_REG'].find('GN')>-1

    # activation function
    if params['TYPE_REG'].find('RELU')>-1:
        this_activation = 'relu'
    elif params['TYPE_REG'].find('GELU')>-1:
        this_activation = gelu
    elif params['TYPE_REG'].find('ELU')>-1:
        this_activation = ELU(alpha=1)
    else:
        this_activation = 'linear'

    # optimizer
    if params['TYPE_REG'].find('AdamW')>-1:
        # Increase initial learning rate for better exploration
        initial_lr = params['LEARNING_RATE'] * 2.0
        this_optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_lr, clipnorm=1.0)
    elif params['TYPE_REG'].find('Adam')>-1:
        this_optimizer = tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE'])
    elif params['TYPE_REG'].find('SGD')>-1:
        this_optimizer = tf.keras.optimizers.SGD(learning_rate=params['LEARNING_RATE'], momentum=0.9)

    if params['TYPE_REG'].find('Glo')>-1:
        print('Using Glorot')
        this_kernel_initializer = 'glorot_uniform'
    elif params['TYPE_REG'].find('He')>-1:
        print('Using He')
        this_kernel_initializer = 'he_normal'

    # rest of parameters
    model_type = params['TYPE_MODEL']
    n_filters = params['NO_FILTERS']
    n_kernels = params['NO_KERNELS']
    n_dilations = params['NO_DILATIONS']

    if params['TYPE_ARCH'].find('Drop')>-1:
        r_drop = float(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Drop')+4:params['TYPE_ARCH'].find('Drop')+6])/100
        print('Using Dropout')
        print(r_drop)
    else:
        r_drop = 0.0

    # Get horizon shift
    if params['TYPE_ARCH'].find('Dori')>-1:
        hori_shift = int(int(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Dori')+4:params['TYPE_ARCH'].find('Dori')+6])/1000*params['SRATE'])
        print('Using Horizon Timesteps:', hori_shift)
        is_classification_only = False

    print('Using Loss Weight:', params['LOSS_WEIGHT'])
    loss_weight = params['LOSS_WEIGHT']

    # load labels
    inputs = Input(shape=(None, input_chans), name='inputs')

    if params['TYPE_ARCH'].find('ZNorm')>-1:
        print('Using ZNorm')
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        std = tf.math.reduce_std(inputs, axis=1, keepdims=True) + 1e-6
        inputs = (inputs - mean) / std

    if params['TYPE_ARCH'].find('CSD')>-1:
        csd_inputs = CSDLayer()(inputs)
        inputs_nets = Concatenate(axis=-1)([inputs, csd_inputs])
    else:
        inputs_nets = inputs

    # get TCN
    if model_type=='Base':
        from tcn import TCN
        tcn_op = TCN(nb_filters=n_filters,
                        kernel_size=n_kernels,
                        nb_stacks=1,
                        dilations=[2 ** i for i in range(n_dilations)],
                        padding='causal',
                        use_skip_connections=True,
                        dropout_rate=r_drop,
                        return_sequences=True,
                        activation=this_activation,
                        kernel_initializer=this_kernel_initializer,
                        use_batch_norm=use_batch_norm,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
        print(tcn_op.receptive_field)
        nets = tcn_op(inputs_nets)

        tcn_clas = TCN(nb_filters=n_filters,
                        kernel_size=n_kernels,
                        nb_stacks=1,
                        dilations=[2 ** i for i in range(n_dilations)],
                        padding='causal',
                        use_skip_connections=True,
                        dropout_rate=r_drop,
                        return_sequences=True,
                        activation=this_activation,
                        kernel_initializer=this_kernel_initializer,
                        use_batch_norm=use_batch_norm,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)

        if params['TYPE_ARCH'].find('L2N')>-1:
            nets = Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(nets)
        output_dim = 8
        prediction_output = Conv1D(output_dim, kernel_size=1, use_bias=True, kernel_initializer=this_kernel_initializer, activation='linear', name='prediction_output')(nets)  # Output future values

        prediction_out_class = tcn_clas(prediction_output)
        if params['TYPE_ARCH'].find('L2N')>-1:
            prediction_out_class = Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(prediction_out_class)

        if params['TYPE_ARCH'].find('Dual')>-1:
            print('Using DualLoss')
            out_class = tcn_clas(inputs)

            if params['TYPE_ARCH'].find('L2N')>-1:
                out_class = Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(out_class)
            prediction_out_class = Concatenate(axis=-1)([prediction_out_class, out_class])

        pred_out = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Pred_Output')(prediction_output)
        tcn_out = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Class_Output')(prediction_out_class)

        if params['TYPE_ARCH'].find('L2N')>-1:
            tcn_output = Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(nets)

        classification_output = Conv1D(1, kernel_size=1, use_bias=True, kernel_initializer='glorot_uniform', activation='sigmoid', name='tmp_class')(tcn_out)

        concat_outputs = Concatenate(axis=-1)([pred_out, classification_output])

    if params['mode']=='embedding':
        concat_outputs = Concatenate(axis=-1)([concat_outputs, tcn_output])
    elif params['mode']!='train':
        concat_outputs = Lambda(lambda tt: tt[:, -1:, :], name='Last_Output')(concat_outputs)

    # Define model with both outputs
    f1_metric = MaxF1MetricHorizon(is_classification_only=is_classification_only, thresholds=tf.linspace(0.0, 1.0, 11))
    r1_metric = RobustF1Metric(is_classification_only=is_classification_only, thresholds=tf.linspace(0.0, 1.0, 11))
    latency_metric = LatencyMetric(is_classification_only=is_classification_only, max_early_detection=25)

    model = Model(inputs=inputs, outputs=concat_outputs)

    # Create loss function without calling it
    loss_fn = custom_fbfce(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model, this_op=tcn_op)

    model.compile(optimizer=this_optimizer,
                    loss=loss_fn,
                    metrics=[f1_metric, r1_metric, latency_metric, FalsePositiveMonitorMetric(model=model)]
                  )

    if params['WEIGHT_FILE']:
        print('load model')
        model.load_weights(params['WEIGHT_FILE'])

    return model

def build_DBI_TCN_CorizonMixer(input_timepoints, input_chans=8, params=None):

    # logit or sigmoid
    flag_sigmoid = 'SigmoidFoc' in params['TYPE_LOSS']
    print('Using Sigmoid:', flag_sigmoid)

    # params
    use_batch_norm = params['TYPE_REG'].find('BN')>-1
    use_weight_norm = params['TYPE_REG'].find('WN')>-1
    use_layer_norm = params['TYPE_REG'].find('LN')>-1
    use_group_norm = params['TYPE_REG'].find('GN')>-1

    # activation function
    if params['TYPE_REG'].find('RELU')>-1:
        this_activation = 'relu'
    elif params['TYPE_REG'].find('GELU')>-1:
        this_activation = gelu
    elif params['TYPE_REG'].find('ELU')>-1:
        this_activation = ELU(alpha=1)
    else:
        this_activation = 'linear'

    # optimizer
    if params['TYPE_REG'].find('AdamW')>-1:
        # Increase initial learning rate for better exploration
        initial_lr = params['LEARNING_RATE'] * 2.0
        this_optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_lr, clipnorm=1.0)
    elif params['TYPE_REG'].find('Adam')>-1:
        this_optimizer = tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE'])
    elif params['TYPE_REG'].find('SGD')>-1:
        this_optimizer = tf.keras.optimizers.SGD(learning_rate=params['LEARNING_RATE'], momentum=0.9)

    if params['TYPE_REG'].find('Glo')>-1:
        print('Using Glorot')
        this_kernel_initializer = 'glorot_uniform'
    elif params['TYPE_REG'].find('He')>-1:
        print('Using He')
        this_kernel_initializer = 'he_normal'

    # rest of parameters
    model_type = params['TYPE_MODEL']
    n_filters = params['NO_FILTERS']
    n_kernels = params['NO_KERNELS']
    n_dilations = params['NO_DILATIONS']

    if params['TYPE_ARCH'].find('Drop')>-1:
        r_drop = float(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Drop')+4:params['TYPE_ARCH'].find('Drop')+6])/100
        print('Using Dropout')
        print(r_drop)
    else:
        r_drop = 0.0

    # Get horizon shift
    if params['TYPE_ARCH'].find('Cori')>-1:
        hori_shift = int(int(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Cori')+4:params['TYPE_ARCH'].find('Cori')+6])/1000*params['SRATE'])
        print('Using Horizon Timesteps:', hori_shift)
        is_classification_only = False

    print('Using Loss Weight:', params['LOSS_WEIGHT'])
    loss_weight = params['LOSS_WEIGHT']

    # load labels
    inputs = Input(shape=(None, input_chans), name='inputs')

    if params['TYPE_ARCH'].find('ZNorm')>-1:
        print('Using ZNorm')
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        std = tf.math.reduce_std(inputs, axis=1, keepdims=True) + 1e-6
        inputs = (inputs - mean) / std

    csd_inputs = CSDLayer()(inputs)
    inputs_nets = Concatenate(axis=-1)([inputs, csd_inputs])

    # get TCN
    if model_type=='Base':
        from tcn import TCN
        tcn_op = TCN(nb_filters=n_filters,
                        kernel_size=n_kernels,
                        nb_stacks=1,
                        dilations=[2 ** i for i in range(n_dilations)],
                        padding='causal',
                        use_skip_connections=True,
                        dropout_rate=r_drop,
                        return_sequences=True,
                        activation=this_activation,
                        kernel_initializer=this_kernel_initializer,
                        use_batch_norm=use_batch_norm,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
        print(tcn_op.receptive_field)
        nets = tcn_op(inputs_nets)

        tcn_clas = TCN(nb_filters=n_filters,
                        kernel_size=n_kernels,
                        nb_stacks=1,
                        dilations=[2 ** i for i in range(n_dilations)],
                        padding='causal',
                        use_skip_connections=True,
                        dropout_rate=r_drop,
                        return_sequences=True,
                        activation=this_activation,
                        kernel_initializer=this_kernel_initializer,
                        use_batch_norm=use_batch_norm,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)

        output_dim = 8
        prediction_output = Conv1D(output_dim, kernel_initializer=this_kernel_initializer, kernel_size=1, use_bias=True, activation='linear', name='prediction_output')(nets)  # Output future values

        prediction_out_class = tcn_clas(prediction_output)
        in_out_class = tcn_clas(inputs)
        csd_out_class = tcn_clas(csd_inputs)
        class_concat = Concatenate(axis=-1)([in_out_class, csd_out_class, prediction_out_class])

        pred_out = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Pred_Output')(prediction_output)
        tcn_out = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Class_Output')(class_concat)

        if params['TYPE_ARCH'].find('L2N')>-1:
            pred_out = Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(pred_out)
            tcn_out = Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(tcn_out)

        classification_output = Conv1D(1, kernel_size=1, kernel_initializer='glorot_uniform', use_bias=True, activation='sigmoid', name='tmp_class')(tcn_out)

        concat_outputs = Concatenate(axis=-1)([pred_out, classification_output])

    if params['mode']=='embedding':
        concat_outputs = Concatenate(axis=-1)([concat_outputs, tcn_output])
    elif params['mode']=='predict':
        concat_outputs = Lambda(lambda tt: tt[:, -1:, :], name='Last_Output')(concat_outputs)

    model = Model(inputs=inputs, outputs=concat_outputs)
    f1_metric = MaxF1MetricHorizon(is_classification_only=is_classification_only, thresholds=tf.linspace(0.0, 1.0, 11))
    r1_metric = RobustF1Metric(is_classification_only=is_classification_only, thresholds=tf.linspace(0.0, 1.0, 11))
    latency_metric = LatencyMetric(is_classification_only=is_classification_only, max_early_detection=25)

    # Create loss function without calling it
    loss_fn = custom_fbfce(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model, this_op=tcn_op)

    model.compile(optimizer=this_optimizer,
                    loss=loss_fn,
                    metrics=[f1_metric, r1_metric, latency_metric, FalsePositiveMonitorMetric(model=model)]
                  )

    if params['WEIGHT_FILE']:
        print('load model')
        model.load_weights(params['WEIGHT_FILE'])

    return model


def build_CAD_Downsampler(input_shape=(1536*2, 8), target_length=128*2, embed_dim=32):
    """
    Causal Context-Aware Downsampler with ELU activations and residual connections.
    Downsamples from 1024 → 128 timepoints.
    """
    inputs = Input(shape=input_shape, name="highfreq_input")
    x = inputs

    num_blocks = int(tf.math.log(float(input_shape[0]) / target_length) / tf.math.log(2.0))
    for i in range(num_blocks):
        # Store the input before processing for skip connection
        input_layer = x

        # Apply convolution with downsampling (stride=2)
        x = layers.Conv1D(embed_dim,
                          kernel_size=5 if i == 0 else 3,
                          strides=3 if i == 0 else 2,
                          padding="causal",
                          activation=None,
                        name=f"conv_block_{i}")(x)
        x = layers.LayerNormalization()(x)
        x = layers.ELU()(x)

        # Use average pooling to downsample the input to match conv output dimensions
        pooled_input = layers.AveragePooling1D(pool_size=2, strides=2, padding="same")(input_layer)

        # Project channels if needed (regardless of sequence length)
        if pooled_input.shape[-1] != embed_dim:
            pooled_input = layers.Conv1D(embed_dim, kernel_size=1, padding="same")(pooled_input)

        # Always trim pooled_input to match x's length using a Lambda layer
        # This avoids comparing symbolic tensors directly
        pooled_input = layers.Lambda(
            lambda tensors: tensors[0][:, :tf.shape(tensors[1])[1], :],
            name=f"trim_skip_{i}"
        )([pooled_input, x])

        # Add the transformed input as a skip connection
        x = layers.Add()([x, pooled_input])

    # Final projection to match original channel dim (8)
    x = layers.Conv1D(input_shape[1], kernel_size=1, activation="elu")(x)

    return Model(inputs, x, name="CAD_Downsampler")

def build_DBI_TCN_CADMixerOnly(input_timepoints, input_chans=8, params=None, pretrained_tcn=None):
    print('Input Timepoints', input_timepoints)
    # optimizer
    if params['TYPE_REG'].find('AdamW')>-1:
        # Increase initial learning rate for better exploration
        initial_lr = params['LEARNING_RATE'] * 2.0
        this_optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_lr, clipnorm=1.0)
    elif params['TYPE_REG'].find('Adam')>-1:
        this_optimizer = tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE'])
    elif params['TYPE_REG'].find('SGD')>-1:
        this_optimizer = tf.keras.optimizers.SGD(learning_rate=params['LEARNING_RATE'], momentum=0.9)

    downsample_dim = params['NO_FILTERS']
    # horizontal shift
    hori_shift = 0
    if params['TYPE_ARCH'].find('Only')>-1:
        hori_shift = int(int(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Only')+4:params['TYPE_ARCH'].find('Only')+6])/1000*2500)
        print('Using Horizon Timesteps:', hori_shift)

    print('Using Loss Weight:', params['LOSS_WEIGHT'])
    loss_weight = params['LOSS_WEIGHT']
    # model to be trained
    # with K.name_scope('CAD'):  # name scope used to make sure weights get unique names
    # CAD = build_CAD_Downsampler(input_shape=(None, 8), target_length=input_timepoints, embed_dim=downsample_dim)  # CAD module
    CAD = build_CAD_Downsampler(input_shape=(1104, 8), target_length=92, embed_dim=downsample_dim)  # CAD module

    # Inputs
    X_highfreq = Input(shape=(None, 8), name="X_highfreq")     # 30 kHz input
    # y_label = Input(shape=(T,), name="y_label")                # label for classification (same T as TCN output)

    # CAD module
    X_downsampled = CAD(X_highfreq)                            # Output shape: (96, 8)

    # Frozen TCN
    TCN_output = pretrained_tcn(X_downsampled)
    # pdb.set_trace()
    TCN_output = Lambda(lambda x: x[:, -1:, :], name='Last_Output')(TCN_output)  # Output shape: (128, 8)
    # TCN_output = Lambda(lambda x: x[:, -input_timepoints:, :], name='Last_Output')(TCN_output)  # Output shape: (128, 8)


    model = Model(inputs=X_highfreq, outputs=TCN_output)
    model._is_classification_only = True
    # model._is_cad = True

    f1_metric = MaxF1MetricHorizon(model=model)
    event_f1_metric = EventAwareF1(model=model)
    fp_event_metric = EventFalsePositiveRateMetric(model=model)

    # Create loss function without calling it
    loss_fn = custom_fbfce(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model)

    model.compile(optimizer=this_optimizer,
                  loss=loss_fn,
                  metrics=[f1_metric, event_f1_metric, fp_event_metric])

    if params['WEIGHT_FILE']:
        print('load model')
        model.load_weights(params['WEIGHT_FILE'])

    return model

class WarmStableCool(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Piecewise schedule on *steps*:
      - warmup:   linear 0 -> base_lr over warmup_ratio of total_steps
      - stable:   constant base_lr until cool phase starts
      - cooldown: cosine from base_lr -> base_lr*final_scale over cool_ratio of total_steps
    """
    def __init__(self, base_lr, total_steps, warmup_ratio=0.02, cool_ratio=0.80, final_scale=0.10):
        self.base_lr = tf.convert_to_tensor(base_lr, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.warmup_steps = tf.cast(tf.round(self.total_steps * warmup_ratio), tf.float32)
        self.cool_steps   = tf.cast(tf.round(self.total_steps * cool_ratio),  tf.float32)
        self.final_scale  = tf.convert_to_tensor(final_scale, tf.float32)

        # phase boundaries
        self.warm_end = self.warmup_steps                     # [0, warm_end)
        self.cool_start = self.total_steps - self.cool_steps  # [cool_start, total_steps]

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # warmup: linear 0 -> base_lr
        warm_lr = self.base_lr * (step / tf.maximum(1.0, self.warmup_steps))

        # stable: constant base_lr
        stable_lr = self.base_lr

        # cooldown: cosine to base_lr*final_scale
        t = (step - self.cool_start) / tf.maximum(1.0, self.cool_steps)
        t = tf.clip_by_value(t, 0.0, 1.0)
        cool_lr = (self.base_lr * self.final_scale) + (self.base_lr - self.base_lr * self.final_scale) * 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * t))

        # piecewise select
        lr = tf.where(step < self.warm_end, warm_lr,
             tf.where(step < self.cool_start, stable_lr, cool_lr))
        return lr

    def get_config(self):
        return {
            "base_lr": float(self.base_lr.numpy()),
            "total_steps": int(self.total_steps.numpy()),
            "warmup_ratio": float((self.warmup_steps / self.total_steps).numpy()),
            "cool_ratio": float((self.cool_steps / self.total_steps).numpy()),
            "final_scale": float(self.final_scale.numpy())
        }

class MultiScaleCausalGate(Layer):
    """
    Multi-scale, channel-agnostic, causal gating without grouped Conv1D.
    Depthwise path uses DepthwiseConv2D on [B,T,1,C] with explicit left (causal) padding.
    Pointwise/mix/gate/GLU are 1x1 Conv1D (no groups → ONNX-friendly).
    Accepts `groups` for API compatibility, but ignores it internally.
    """
    def __init__(self,
                 scales=((7,1),(7,2),(11,4)),
                 groups=1,                 # <- accepted for compatibility
                 gate_bias=-2.5,
                 l1_gate=0.0,
                 use_residual_mix=True,
                 mix_activation='gelu',
                 name="ms_causal_gate",
                 **kw):
        # IMPORTANT: do NOT forward unknown kwargs like `groups` to super().__init__
        super().__init__(name=name, **kw)
        self.scales = list(scales)
        self.groups = int(groups)          # stored but not used (keeps call sites working)
        self.gate_bias = float(gate_bias)
        self.l1_gate = float(l1_gate)
        self.use_residual_mix = bool(use_residual_mix)
        self.mix_activation = mix_activation

        # will be built in build()
        self._dw_blocks = []   # list of (pad2d, depthwise2d)
        self._pw_blocks = []   # per-scale pointwise Conv1D
        self._mix_pw    = None # cross-scale 1x1 Conv1D
        self._gate_logits = None
        self._res_scale  = None
        self._res_glu    = None

    def _causal_depthwise_1d(self, x, pad2d, dw2d):
        """
        x: [B,T,C] -> pad causally on time, view as [B,T,1,C],
        depthwise conv2d with kernel=(k,1), dilation=(d,1), then squeeze back.
        """
        x2 = tf.expand_dims(x, axis=2)              # [B,T,1,C]
        x2 = pad2d(x2)                               # ZeroPadding2D((pad_left,0),(0,0))
        y2 = dw2d(x2)                                # [B,T,1,C]
        return tf.squeeze(y2, axis=2)                # [B,T,C]

    def build(self, input_shape):
        assert len(input_shape) == 3, "Expected [B, T, C]"
        C = int(input_shape[-1])

        # Per-scale: causal pad + depthwise2D + pointwise1D
        for (k, d) in self.scales:
            k = int(k); d = int(d)
            pad_left = d * (k - 1)
            pad2d = ZeroPadding2D(
                padding=((pad_left, 0), (0, 0)),
                name=f"{self.name}_pad_k{k}_d{d}"
            )
            dw2d = DepthwiseConv2D(
                kernel_size=(k, 1),
                dilation_rate=(d, 1),
                padding="valid",              # causal via explicit left pad
                depth_multiplier=1,
                use_bias=False,
                kernel_initializer="he_normal",
                name=f"{self.name}_dw2d_k{k}_d{d}"
            )
            self._dw_blocks.append((pad2d, dw2d))

            pw1d = Conv1D(
                filters=C,
                kernel_size=1,
                activation=self.mix_activation,
                kernel_initializer="glorot_uniform",
                name=f"{self.name}_pw1d_k{k}_d{d}"
            )
            self._pw_blocks.append(pw1d)

        # Cross-scale mix back to C (1x1 Conv1D)
        self._mix_pw = Conv1D(
            filters=C, kernel_size=1,
            activation=self.mix_activation,
            kernel_initializer="glorot_uniform",
            name=f"{self.name}_mix_pw"
        )

        # Gate logits per channel (1x1 Conv1D)
        reg = l1(self.l1_gate) if self.l1_gate > 0 else None
        self._gate_logits = Conv1D(
            filters=C, kernel_size=1, activation=None,
            kernel_initializer="glorot_uniform",
            bias_initializer=Constant(self.gate_bias),
            kernel_regularizer=reg,
            name=f"{self.name}_gate_logits"
        )

        # Optional residual GLU (+ learnable scaler starting at 0)
        if self.use_residual_mix:
            self._res_scale = self.add_weight(
                name=f"{self.name}_res_scale",
                shape=(),
                initializer=Constant(0.0),
                trainable=True
            )
            self._res_glu = Conv1D(
                filters=2*C, kernel_size=1, activation=None,
                kernel_initializer="glorot_uniform",
                name=f"{self.name}_res_glu"
            )

        super().build(input_shape)

    def call(self, x):
        # Per-scale features
        feats = []
        for (pad2d, dw2d), pw1d in zip(self._dw_blocks, self._pw_blocks):
            z = self._causal_depthwise_1d(x, pad2d, dw2d)  # [B,T,C]
            z = pw1d(z)                                    # [B,T,C]
            feats.append(z)

        # Lightweight contrasts/ratios to aid SNR-like behavior
        concat_list = list(feats)
        if len(feats) >= 3:
            s, m, l = feats[0], feats[1], feats[2]
            concat_list += [s - m, m - l, s - l, s / (tf.abs(l) + 1e-3)]

        ms = tf.concat(concat_list, axis=-1)               # [B,T, C*(#terms)]

        h    = self._mix_pw(ms)                            # [B,T,C]
        gate = tf.nn.sigmoid(self._gate_logits(h))         # [B,T,C]
        y    = x * gate

        if self.use_residual_mix:
            ab  = self._res_glu(ms)                        # [B,T,2C]
            a, b = tf.split(ab, 2, axis=-1)
            glu = a * tf.nn.sigmoid(b)                     # [B,T,C]
            y   = y + tf.tanh(self._res_scale) * glu

        return y

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            scales=self.scales,
            groups=self.groups,               # preserved in config for symmetry
            gate_bias=self.gate_bias,
            l1_gate=self.l1_gate,
            use_residual_mix=self.use_residual_mix,
            mix_activation=self.mix_activation,
            name=self.name
        ))
        return cfg
    
def build_DBI_TCN_TripletOnly(input_timepoints, input_chans=8, params=None):

    # LR schedure 
    # def make_lr_schedule(base_lr, total_steps, warmup_ratio=0.05, min_lr_ratio=0.05):
    #     warmup_steps = int(total_steps * warmup_ratio)
    #     min_lr = base_lr * min_lr_ratio

    #     def lr_fn(step):
    #         step = tf.cast(step, tf.float32)
    #         # linear warmup
    #         lr = base_lr * step / tf.maximum(1.0, tf.cast(warmup_steps, tf.float32))
    #         # cosine decay after warmup
    #         progress = (step - warmup_steps) / tf.maximum(1.0, float(total_steps - warmup_steps))
    #         progress = tf.clip_by_value(progress, 0.0, 1.0)
    #         cosine = 0.5 * (1 + tf.cos(tf.constant(3.14159265359) * progress))
    #         lr = tf.where(step < warmup_steps, lr, min_lr + (base_lr - min_lr) * cosine)
    #         return lr

    #     return tf.keras.optimizers.schedules.LearningRateSchedule(lr_fn)    

    # params
    use_batch_norm = params['TYPE_REG'].find('BN')>-1
    use_weight_norm = params['TYPE_REG'].find('WN')>-1
    use_layer_norm = params['TYPE_REG'].find('LN')>-1
    use_group_norm = params['TYPE_REG'].find('GN')>-1

    # this_activation = 'relu'
    if params['TYPE_REG'].find('RELU')>-1:
        this_activation = 'relu'
    elif params['TYPE_REG'].find('GELU')>-1:
        this_activation = gelu
    elif params['TYPE_REG'].find('ELU')>-1:
        this_activation = ELU(alpha=1)
    else:
        this_activation = 'linear'

    if params['TYPE_REG'].find('LOne')>-1:
        reg_weight = params.get('reg_weight', 1e-4)
        this_regularizer = tf.keras.regularizers.L1(reg_weight)
    elif params['TYPE_REG'].find('LTwo')>-1:
        reg_weight = params.get('reg_weight', 1e-4)
        this_regularizer = tf.keras.regularizers.L2(reg_weight)
    else:
        this_regularizer = None

    # optimizer
    if params['TYPE_REG'].find('AdamW')>-1:

        # --- derive total steps from dataset config ---
        steps_per_epoch = int(params.get('ESTIMATED_STEPS_PER_EPOCH', params.get('steps_per_epoch', 1000)))
        total_steps = float(params.get('TOTAL_STEPS', 100000))

        lr_schedule = WarmStableCool(
            base_lr=params['LEARNING_RATE'],
            total_steps=total_steps,
            warmup_ratio=params.get('LR_WARMUP_RATIO', 0.04),
            cool_ratio=params.get('LR_COOL_RATIO', 0.60),
            final_scale=params.get('LR_FINAL_SCALE', 0.06),
        )
        this_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=(lr_schedule if params.get('USE_LR_SCHEDULE', True) else params['LEARNING_RATE']),
            weight_decay=params.get('WEIGHT_DECAY', 1e-4),
            clipnorm=float(params.get('CLIP_NORM', 1.5)),
            epsilon=1e-8
        )
               
    elif params['TYPE_REG'].find('Adam')>-1:
        this_optimizer = tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE'])
    elif params['TYPE_REG'].find('SGD')>-1:
        this_optimizer = tf.keras.optimizers.SGD(learning_rate=params['LEARNING_RATE'], momentum=0.9)

    if params['TYPE_REG'].find('Glo')>-1:
        print('Using Glorot')
        this_kernel_initializer = 'glorot_uniform'
    elif params['TYPE_REG'].find('He')>-1:
        print('Using He')
        this_kernel_initializer = 'he_normal'

    model_type = params['TYPE_MODEL']
    n_filters = params['NO_FILTERS']
    n_kernels = params['NO_KERNELS']
    n_dilations = params['NO_DILATIONS']

    if params['TYPE_ARCH'].find('Drop')>-1:
        r_drop = float(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Drop')+4:params['TYPE_ARCH'].find('Drop')+6])/100
        print('Using Dropout')
        print(r_drop)
    else:
        r_drop = 0.0

    hori_shift = 0
    if params['TYPE_ARCH'].find('Only')>-1:
        hori_shift = int(int(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Only')+4:params['TYPE_ARCH'].find('Only')+6])/1000*params['SRATE'])
        print('Using Horizon Timesteps:', hori_shift)
    is_classification_only = True

    print('Using Loss Weight:', params['LOSS_WEIGHT'])
    loss_weight = params['LOSS_WEIGHT']
         
    # Function to create the TCN backbone for processing LFP signals
    def create_tcn_model(input_timepoints):

        signal_input = Input(shape=(None, input_chans))

        # Apply CSD if needed
        if params['TYPE_ARCH'].find('CSD')>-1:
            csd_inputs = CSDLayer()(signal_input)
            inputs_nets = Concatenate(axis=-1)([signal_input, csd_inputs])
        else:
            inputs_nets = signal_input

        if params['TYPE_ARCH'].find('Att')>-1:
            # Multi-scale causal, channel-agnostic input layer (learns SNR-like gates + patterns)
            groups = 2 if input_chans in (16, 24, 32) else 1   # e.g., (low,high) pairs per sensor
            inputs_nets = MultiScaleCausalGate(
                scales=((7,1),(7,2),(11,4)),                   # RFs: 7, 13, 41 (all ≤ 43)
                groups=groups,
                gate_bias=(-3.0),  # tighter in pretrain
                # gate_bias=(-3.0 if params['mode']=="train" else -2.0),  # tighter in pretrain
                l1_gate=params.get("l1_gate", 1e-5), 
                use_residual_mix=True,
                name="ms_causal_gate"
            )(inputs_nets)            

        # Apply TCN
        if model_type=='Base':
            print('Using Base TCN')
            from tcn import TCN
            tcn_op = TCN(nb_filters=n_filters,
                        kernel_size=n_kernels,
                        nb_stacks=1,
                        dilations=[2 ** i for i in range(n_dilations)],
                        padding='causal',
                        use_skip_connections=True,
                        dropout_rate=r_drop,
                        return_sequences=True,
                        activation=this_activation,
                        kernel_initializer=this_kernel_initializer,
                        kernel_regularizer=this_regularizer,
                        use_batch_norm=use_batch_norm,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
            print(tcn_op.receptive_field)
            nets = tcn_op(inputs_nets)  # [B, T, C_tcn]
            feats = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='slice_last_T')(nets)
        elif model_type=='SingleCh':
            print('Using Single Channel TCN')
            from tcn import TCN
            tcn_op = TCN(nb_filters=n_filters,
                        kernel_size=n_kernels,
                        nb_stacks=1,
                        dilations=[2 ** i for i in range(n_dilations)],
                        padding='causal',
                        use_skip_connections=True,
                        dropout_rate=r_drop,
                        return_sequences=True,
                        activation=this_activation,
                        kernel_initializer=this_kernel_initializer,
                        use_batch_norm=use_batch_norm,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
            single_outputs = []
            for c in range(input_chans):
                ch_slice = Lambda(lambda x: x[:, :, c:c+1])(signal_input)
                single_outputs.append(tcn_op(ch_slice))
            nets = Concatenate(axis=-1)(single_outputs)
            feats = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)

        # if params['TYPE_ARCH'].find('Att')>-1:
        #     y     = LayerNormalization(name='ln_mhsa_post')(feats)
        #     y     = CausalMHA1D(num_heads=4, key_dim=4, use_rope=True, alibi_slope=0.15,name='mhsa_post')(y)
        #     alpha = tf.Variable(0.0, trainable=True, name='res_scaler')  # start at 0
        #     scaled_y = Multiply(name='scaled_y')(tf.tanh(alpha), y)
        #     feats = Add(name='res_mhsa_post')([feats, scaled_y])
        #     y     = LayerNormalization(name='ln_local')(feats)
        #     y = tf.keras.layers.SeparableConv1D(
        #             filters=feats.shape[-1], kernel_size=7, padding='causal',
        #             depth_multiplier=1, pointwise_initializer='he_normal',
        #             name='sep_dwconv7_causal')(y)
        #     y     = Activation('gelu', name='gelu_local')(y)
        #     tgate = Conv1D(1, 1, activation='sigmoid',
        #                 kernel_initializer='glorot_uniform',
        #                 bias_initializer=tf.keras.initializers.Constant(-2.0),
        #                 name='time_gate')(y)
        #     feats = Multiply(name='apply_time_gate')([feats, tgate])


        # ---- Classification head ----
        # h = tf.stop_gradient(feats) if 'StopGrad' in params['TYPE_ARCH'] else feats
        # h = LayerNormalization(name='emb_class')(h)
        # # h = LayerNormalization()(h)
        # h = Conv1D(64, 
        #             1, 
        #             kernel_initializer='glorot_uniform', 
        #             kernel_regularizer=tf.keras.regularizers.L1(1e-4),
        #             name="cls_pw1")(h)
        # h = Activation('gelu')(h)
        # # cls_in = Dropout(0.1)(h)
        # cls_in = h
        # cls_logits = Conv1D(1, 1,
        #                 kernel_initializer='glorot_uniform',
        #                 kernel_regularizer=tf.keras.regularizers.L1(1e-4),
        #                 bias_initializer='zeros',
        #                 activation=None, name='cls_logits')(cls_in)
        # cls_prob = Activation('sigmoid', name='cls_prob')(cls_logits)
        # T = int(input_timepoints) # 64
        # print('TCN input timepoints:', T)
        # R = tcn_op.receptive_field # 43
        # print('TCN receptive field:', R)
        # K_MAX = max(1, T - R + 1)  # must be <=22 with T=64,R=43

        # 
        # feats_tail_in = Lambda(lambda tt: tt[:, -(input_timepoints+K_MAX):, :], name='slice_tail_T_plus_K')(nets)        
        # feats_tail_in = tf.stop_gradient(feats_tail_in) if 'StopGrad' in params['TYPE_ARCH'] else feats_tail_in

        # # 2) Tail (all causal)
        # k_list = [2,4,8,16]  # later filter by <=K_MAX
        # C = 16
        
        # branches = [
        #     tf.keras.layers.SeparableConv1D(
        #         filters=C, kernel_size=k, padding='causal', kernel_initializer=this_kernel_initializer,
        #         depth_multiplier=1, use_bias=False, name=f'tail_dw_k{k:02d}'
        #     )(feats_tail_in) for k in k_list
        # ]
        # tail = tf.keras.layers.Concatenate(axis=-1, name='tail_ms_concat')(branches)
        # tail = tf.keras.layers.LayerNormalization(name='tail_ln')(tail)

        # glu = tf.keras.layers.Conv1D(64, 1, use_bias=True, kernel_initializer='glorot_uniform',
        #                                name='tail_glu_in')(tail)
        # lin, gate = tf.split(glu, 2, axis=-1)
        # h_tail = tf.keras.layers.Multiply(name='tail_glu_mul')(
        #     [lin, tf.keras.layers.Activation('sigmoid', name='tail_glu_sig')(gate)]
        # )


        # # Crop to last T so the classifier head is length T (aligns with emb)
        # cls_logits_in = Lambda(lambda tt: tt[:, -(input_timepoints):, :], name='slice_pre_logits')(h_tail)        
        cls_logits_in = tf.stop_gradient(feats) if 'StopGrad' in params['TYPE_ARCH'] else feats

        cls_logits = tf.keras.layers.Conv1D(
            1, 1, use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation=None,
            name='cls_logits'
        )(cls_logits_in)
        cls_prob = tf.keras.layers.Activation('sigmoid', name='cls_prob')(cls_logits)

        
        # ---- projection head for contrastive (L2N ONLY here) ----
        emb = LayerNormalization(name='emb_ln')(feats)
        emb = Dense(n_filters * 2, activation=this_activation,
                    kernel_initializer=this_kernel_initializer, name='emb_p1')(emb)
        emb = Dense(n_filters, activation=None,
                    kernel_initializer=this_kernel_initializer, name='emb_p2')(emb)
        emb = Lambda(lambda z: tf.math.l2_normalize(z, axis=-1), name='emb_l2n')(emb)

        # ---- outputs (prob first, then emb) ----
        out = Concatenate(axis=-1, name='concat_cls_emb')([cls_logits, cls_prob, emb])

        tcn_mod = Model(inputs=signal_input, outputs=out, name='tcn_triplet')
                
        tcn_mod._cls_logit_index  = 0   # channel of logits
        tcn_mod._cls_prob_index   = 1   # channel of probabilities
        tcn_mod._emb_start_index  = 2   # start of embedding slice
        # tcn_mod.trainable = True
        tcn_mod._is_classification_only = True

        return tcn_mod

    # Create the shared TCN model for triplet learning
    tcn_backbone = create_tcn_model(input_timepoints)

    # For training with triplet loss
    if params['mode'] == 'train':
        # Define inputs for anchor, positive, and negative samples
        # anchor_input = Input(shape=(None, input_chans), name='anchor')
        # positive_input = Input(shape=(None, input_chans), name='positive')
        # negative_input = Input(shape=(None, input_chans), name='negative')
        all_inputs = Input(shape=(None, input_chans), name='all_inputs')


        batch_size = tf.shape(all_inputs)[0] // 3

        # Split tensors using tf.split instead of unpacking
        anchor_input, positive_input, negative_input = tf.split(all_inputs, 3, axis=0)
        # anchor_true, positive_true, negative_true = tf.split(y_true, 3, axis=0)

        # Process inputs through shared TCN model
        anchor_output = tcn_backbone(anchor_input)
        positive_output = tcn_backbone(positive_input)
        negative_output = tcn_backbone(negative_input)

        all_outputs = Concatenate(axis=0)([anchor_output, positive_output, negative_output])
        # Create the training model with dictionary inputs and outputs
        model = Model(
            inputs = all_inputs,
            outputs = all_outputs
        )

        # Metrics - only applied to anchor output for metric tracking
        model._is_classification_only = True 
        metrics = [
            SamplePRAUC(model=model),
            SampleMaxMCC(model=model),
            SampleMaxF1(model=model),
            LatencyScore(thresholds=tf.linspace(0.5, 0.99, 6), min_run=5, tau=16.0, model=model),
            RecallAt0p7(model=model),  # <-- new
            FPperMinAt0p3(
                win_sec=params['NO_STRIDES']/params['SRATE'],
                mode="consec", consec_k=3,
                model=model
            ),                          # <-- new (name default 'fp_per_min')
        ]          

        # Create loss function and compile model
        # loss_fn = triplet_loss(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model)
        # loss_fn = mixed_latent_loss(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model)
        # loss_fn = mixed_circle_loss(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model)
        # loss_fn = mixed_mpnFocal_loss(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model)
        loss_fn = mixed_hybrid_loss_final(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model)

        model.compile(
            optimizer=this_optimizer,
            loss=loss_fn,
            metrics=metrics
        )

    elif params['mode'] == 'fine_tune':
        
        all_inputs = Input(shape=(None, input_chans), name='all_inputs')

        batch_size = tf.shape(all_inputs)[0] // 3

        # Split tensors using tf.split instead of unpacking
        anchor_input, positive_input, negative_input = tf.split(all_inputs, 3, axis=0)
        # anchor_true, positive_true, negative_true = tf.split(y_true, 3, axis=0)

        # Process inputs through shared TCN model
        anchor_output = tcn_backbone(anchor_input)
        positive_output = tcn_backbone(positive_input)
        negative_output = tcn_backbone(negative_input)

        all_outputs = Concatenate(axis=0)([anchor_output, positive_output, negative_output])
        # Create the training model with dictionary inputs and outputs
        model = Model(
            inputs = all_inputs,
            outputs = all_outputs
        )

        # Metrics - only applied to anchor output for metric tracking
        model._is_classification_only = True
        metrics = [
            SamplePRAUC(model=model),
            SampleMaxMCC(model=model),
            SampleMaxF1(model=model),
            LatencyScore(thresholds=tf.linspace(0.5, 0.99, 6), min_run=5, tau=16.0, model=model),
            RecallAt0p7(model=model),  # <-- new
            FPperMinAt0p3(
                win_sec=params['NO_STRIDES']/params['SRATE'],
                mode="consec", consec_k=3,
                model=model
            ),                          # <-- new (name default 'fp_per_min')
        ]        
        # Create loss function and compile model
        # loss_fn = triplet_loss(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model)
        # loss_fn = mixed_latent_loss(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model)
        loss_fn = class_finetune(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model)
        model.load_weights(params['WEIGHT_FILE'], by_name=True)
        print('Loaded weights for fine-tuning from', params['WEIGHT_FILE'])
        model.compile(
            optimizer=this_optimizer,
            loss=loss_fn,
            metrics=metrics
        ) 
    else:
        all_inputs = Input(shape=(None, input_chans), name='all_inputs')


        batch_size = tf.shape(all_inputs)[0] // 3

        # Split tensors using tf.split instead of unpacking
        # anchor_input, positive_input, negative_input = tf.split(all_inputs, 3, axis=0)
        # anchor_true, positive_true, negative_true = tf.split(y_true, 3, axis=0)
        # anchor_input = all_inputs
        # Process inputs through shared TCN model
        anchor_output = tcn_backbone(all_inputs)
        # positive_output = tcn_backbone(positive_input)
        # negative_output = tcn_backbone(negative_input)


        if params['mode']=='embedding':
            all_outputs = Lambda(lambda tt: tt[:, -1:, :], name='Last_Output')(anchor_output)
        elif params['mode']=='predict':
            # all_outputs = Lambda(lambda tt: tt[:, -1:, 0], name='Last_Output')(anchor_output)
            all_outputs = Lambda(lambda tt: tt[:, -1:, tcn_backbone._cls_prob_index], name='Last_Output')(anchor_output)
        else:
            all_outputs = Lambda(lambda tt: tt[:, -1:, tcn_backbone._cls_prob_index], name='Last_Output')(anchor_output)


        # all_outputs = Concatenate(axis=0)([anchor_output, positive_output, negative_output])
        # Create the training model with dictionary inputs and outputs
        model = Model(
            inputs = all_inputs,
            outputs = all_outputs
        )

        # Metrics - only applied to anchor output for metric tracking
        model._is_classification_only = True

        # f1_metric = MaxF1MetricHorizon(model=model)
        # r1_metric = RobustF1Metric(model=model)
        # latency_metric = LatencyMetric(model=model)

        model.compile(
            optimizer=this_optimizer,
            loss='mse',
            # metrics=[f1_metric, r1_metric, latency_metric]
        )

    if params['WEIGHT_FILE']:
        print('load model')
        model.load_weights(params['WEIGHT_FILE'], by_name=True)#, skip_mismatch=False, by_name=False)

    return model

#################### Custom Layers ####################
class CSDLayer(Layer):
    """
    A custom Keras layer that computes Current Source Density (CSD)
    from the input signal.
    """

    def __init__(self, **kwargs):
        super(CSDLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Apply CSD transformation on the inputs.

        Args:
        - inputs: Tensor of shape [batch_size, num_samples, num_channels].

        Returns:
        - csd: Tensor of the same shape as inputs with CSD applied.
        """
        # Apply the CSD computation (second spatial derivative)
        num_channels = tf.shape(inputs)[-1]

        # Compute second spatial derivative (CSD)
        csd = inputs[:, :, 2:] - 2 * inputs[:, :, 1:-1] + inputs[:, :, :-2]

        # # Pad to retain original shape
        # csd = tf.pad(csd, [[0, 0], [0, 0], [1, 1]])
        csd = tf.pad(csd, [[0, 0], [0, 0], [1, 1]], "SYMMETRIC")

        return csd

    def get_config(self):
        config = super(CSDLayer, self).get_config()
        return config

#################### METRICS and LOSSES ####################

def custom_binary_accuracy_mixer(y_true, y_pred):
    # Calculate Binary Accuracy for the last channel
    y_true_exp = tf.expand_dims(y_true[:, :, 8], axis=-1)
    y_pred_exp = tf.math.sigmoid(tf.expand_dims(y_pred[:, :, 8], axis=-1))
    return tf.keras.metrics.binary_accuracy(y_true_exp, y_pred_exp)

def custom_mse_metric(y_true, y_pred):
    # Calculate MSE for the first 8 channels
    prediction_out = y_pred[:, :, :8]
    y_true_exp = y_true[:, :, :8]
    mse_metric = tf.reduce_mean(tf.square(prediction_out - y_true_exp))
    return mse_metric

def custom_binary_accuracy(y_true, y_pred):
    # Calculate Binary Accuracy for the last channel
    y_true_exp = tf.expand_dims(y_true[:, :, 8], axis=-1)
    y_pred_exp = tf.expand_dims(y_pred[:, :, 8], axis=-1)
    return tf.keras.metrics.binary_accuracy(y_true_exp, y_pred_exp)

def custom_l1_regularization_loss(model, l1_lambda=1e-5):
    # Calculate the L1 loss on all trainable weights
    l1_loss = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in model.trainable_weights])
    return l1_lambda * l1_loss

def custom_l2_regularization_loss(model, l2_lambda=1e-5):
    # Calculate the L2 loss on all trainable weights
    l2_loss = tf.add_n([tf.reduce_sum(tf.square(w)) for w in model.trainable_weights])
    return l2_lambda * l2_loss

def truncated_mse_loss(y_true, y_pred, tau=4.0):
    # Log probabilities
    log_probs = tf.math.log(y_pred + 1e-8)  # Adding epsilon to avoid log(0)

    # Frame-wise log differences
    delta = tf.abs(log_probs[:, 1:, :] - log_probs[:, :-1, :])

    # Apply truncation with tau = 4.0
    truncated_delta = tf.where(delta > tau, tau, delta)

    # Average over frames and classes
    t_mse = tf.reduce_mean(truncated_delta)
    return t_mse

#####################
##### METRICS #######
#####################
# Fix for MaxF1MetricHorizon
class MaxF1MetricHorizon(tf.keras.metrics.Metric):
    def __init__(self, name='f1', thresholds=tf.linspace(0.0, 1.0, 11), model=None, **kwargs):
        super(MaxF1MetricHorizon, self).__init__(name=name, **kwargs)
        self.thresholds = thresholds
        self.tp = self.add_weight(shape=(self.thresholds.shape[0],), initializer='zeros', name='tp')
        self.fp = self.add_weight(shape=(self.thresholds.shape[0],), initializer='zeros', name='fp')
        self.fn = self.add_weight(shape=(self.thresholds.shape[0],), initializer='zeros', name='fn')
        self.model = model

    def update_state(self, y_true, y_pred, **kwargs):
        # Get classification mode from model property
        is_classification_only = getattr(self.model, '_is_classification_only', True)

        # Extract labels based on mode - using tf.cond for graph compatibility
        y_true_labels = tf.cond(
            tf.constant(is_classification_only),
            # lambda: tf.slice(y_true, [0, 0, 0], [-1, -1, 1]),
            lambda: tf.slice(y_true, [0, 0, 0], [-1, -1, 1]),
            lambda: tf.slice(y_true, [0, 0, 8], [-1, -1, 1])
        )
        y_pred_labels = tf.cond(
            tf.constant(is_classification_only),
            # lambda: tf.slice(y_pred, [0, 0, 0], [-1, -1, 1]),
            lambda: tf.slice(y_pred, [0, 0, 1], [-1, -1, 1]),
            lambda: tf.slice(y_pred, [0, 0, 8], [-1, -1, 1])
        )

        # Reshape to ensure consistent dimensions
        y_true_labels = tf.reshape(y_true_labels, [tf.shape(y_true)[0], tf.shape(y_true)[1]])
        y_pred_labels = tf.reshape(y_pred_labels, [tf.shape(y_pred)[0], tf.shape(y_pred)[1]])

        def process_threshold(threshold):
            pred_events = tf.cast(y_pred_labels >= threshold, tf.float32)
            true_events = tf.cast(y_true_labels >= 0.5, tf.float32)

            tp = tf.reduce_sum(pred_events * true_events)
            fp = tf.reduce_sum(pred_events * (1 - true_events))
            fn = tf.reduce_sum((1 - pred_events) * true_events)

            return tp, fp, fn

        # Use tf.map_fn to process all thresholds
        metrics = tf.map_fn(
            process_threshold,
            self.thresholds,
            fn_output_signature=(tf.float32, tf.float32, tf.float32)
        )

        tp_all, fp_all, fn_all = metrics

        # Update accumulators
        self.tp.assign_add(tp_all)
        self.fp.assign_add(fp_all)
        self.fn.assign_add(fn_all)

    def result(self):
        # Calculate F1 scores for all thresholds
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f1_scores = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

        # Return maximum F1 score
        return tf.reduce_max(f1_scores)

    def reset_state(self):
        self.tp.assign(tf.zeros_like(self.tp))
        self.fp.assign(tf.zeros_like(self.fp))
        self.fn.assign(tf.zeros_like(self.fn))

EPS = tf.keras.backend.epsilon()

# ──────────────────────────────────────────────────────────────────────
# 1.  Helper: ≥ k consecutive TRUE anywhere in the sequence
#    pred_bin: bool [..., T]
#    returns  : bool [...]  (True if any run ≥ k)
# ──────────────────────────────────────────────────────────────────────
def _has_run_k(pred_bin, k):
    if k <= 1:
        return tf.reduce_any(pred_bin, axis=-1)          # same shape [...]
    x   = tf.cast(pred_bin, tf.float32)                  # [...,T]
    shp = tf.shape(x)
    T   = shp[-1]
    flat = tf.reshape(x, [-1, T, 1])                     # [N,T,1]
    filt = tf.ones((k, 1, 1), flat.dtype)                # [k,1,1]
    runsum = tf.nn.conv1d(flat, filt,
                          stride=1, padding="SAME")      # [N,T,1]
    has_run = tf.reduce_max(runsum, axis=1)[:, 0] >= float(k)  # [N]
    return tf.reshape(has_run, shp[:-1])                 # [...]

# ──────────────────────────────────────────────────────────────────────
# 2.  Helper: ≥ ratio·T positives in the sequence
# ──────────────────────────────────────────────────────────────────────
def _has_majority(pred_bin, ratio):
    if ratio <= 0.0:
        return tf.reduce_any(pred_bin, axis=-1)
    T = tf.shape(pred_bin)[-1]
    m = tf.cast(tf.math.round(tf.cast(T, tf.float32)*ratio), tf.int32)
    cnt = tf.reduce_sum(tf.cast(pred_bin, tf.int32), axis=-1)
    return cnt >= m

# --------------------------------------------------------------------- #
# Voting helpers (time axis = 1)
# --------------------------------------------------------------------- #
def _vote_consecutive(pred_bin, k):          # [B,T] bool
    if k <= 1:
        return pred_bin
    # causal SAME padding keeps length T
    pool = tf.nn.max_pool1d(tf.cast(pred_bin, tf.float32)[..., None],
                            ksize=k, strides=1, padding="SAME",
                            data_format="NWC")
    return tf.cast(pool[..., 0] >= 1.0, tf.bool)                # [B,T]

def _vote_majority(pred_bin, ratio):
    if ratio <= 0.0:
        return pred_bin
    m = tf.cast(tf.math.round(
            tf.cast(tf.shape(pred_bin)[1] * ratio, tf.float32)), tf.int32)
    count = tf.reduce_sum(tf.cast(pred_bin, tf.int32), axis=1, keepdims=True)
    return count >= m                                           # [B,1]  broadcast later
# --------------------------------------------------------------------- #
# Channel helper (unchanged)
# --------------------------------------------------------------------- #
def _ch(t, is_cls):          # [B,T,?] -> [B,T]
    # return t[..., 0] if is_cls else t[..., 8]
    return t[..., 1] if is_cls else t[..., 8]
def _ch_lab(t, is_cls):          # [B,T,?] -> [B,T]
    return t[..., 0] if is_cls else t[..., 8]
# ===================================================================== #
# 1) EVENT-LEVEL PR-AUC with vote rule
# ===================================================================== #
class EventPRAUC(tf.keras.metrics.Metric):
    def __init__(self, thresholds=tf.linspace(0., 1., 11),
                 mode="consec", consec_k=3, majority_ratio=0.0,
                 name="event_pr_auc", model=None, **kw):
        super().__init__(name=name, **kw)
        self.tau = tf.cast(thresholds, tf.float32)
        k = len(thresholds)
        self.tp = self.add_weight("tp", shape=(k,), initializer="zeros")
        self.fp = self.add_weight("fp", shape=(k,), initializer="zeros")
        self.fn = self.add_weight("fn", shape=(k,), initializer="zeros")
        self.model = model
        self.mode = mode
        self.consec_k = int(consec_k)
        self.maj_ratio = float(majority_ratio)

    def update_state(self, y_true, y_pred, **_):
        is_cls = getattr(self.model, "_is_classification_only", True)
        yt, yp = _ch_lab(y_true, is_cls), _ch(y_pred, is_cls)            # [B,T]
        true_evt_b = tf.reduce_max(yt, 1) >= 0.5                     # [B]
        pred_bin_kbt = yp[None,...] >= self.tau[:,None,None]         # [K,B,T]

        if self.mode == "consec":
            pred_evt_bk = _has_run_k(pred_bin_kbt, self.consec_k)    # [K,B]
        elif self.mode == "majority":
            pred_evt_bk = _has_majority(pred_bin_kbt, self.maj_ratio)
        else:  # "any"
            pred_evt_bk = tf.reduce_any(pred_bin_kbt, axis=-1)

        true_bk = true_evt_b[None,:]
        self.tp.assign_add(tf.reduce_sum(tf.cast(pred_evt_bk &  true_bk, tf.float32), 1))
        self.fp.assign_add(tf.reduce_sum(tf.cast(pred_evt_bk & ~true_bk, tf.float32), 1))
        self.fn.assign_add(tf.reduce_sum(tf.cast(~pred_evt_bk &  true_bk, tf.float32), 1))

    def result(self):
        prec = self.tp / (self.tp + self.fp + EPS)
        reca = self.tp / (self.tp + self.fn + EPS)
        idx  = tf.argsort(reca)
        reca = tf.gather(reca, idx); prec = tf.gather(prec, idx)
        return tf.reduce_sum((reca[1:] - reca[:-1]) *
                             (prec[1:] + prec[:-1]) * 0.5)

    def reset_state(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(tf.zeros_like(v))


# ===================================================================== #
# 1) SAMPLE PR-AUC with vote rule
# ===================================================================== #
# ---------- helpers (consistent with your indexing) ----------
def _lab_timepoints(y_true):
    # y_true is [B,T,1] in your setup → always take channel 0
    return y_true[..., 0]

def _pred_prob_timepoints(y_pred, model):
    # use model._cls_prob_index (you set it to 1)
    return y_pred[..., getattr(model, "_cls_prob_index", 1)]

# ---------- Sample-level PR-AUC (timepoint-wise) ----------
class SamplePRAUC(tf.keras.metrics.Metric):
    def __init__(self, thresholds=tf.linspace(0., 1., 11),
                 name="sample_pr_auc", model=None, **kw):
        super().__init__(name=name, **kw)
        k = int(thresholds.shape[0])
        self.tau = tf.cast(thresholds, tf.float32)
        self.tp = self.add_weight("tp", shape=(k,), initializer="zeros")
        self.fp = self.add_weight("fp", shape=(k,), initializer="zeros")
        self.fn = self.add_weight("fn", shape=(k,), initializer="zeros")
        self.model = model

    def update_state(self, y_true, y_pred, **_):
        yt = _lab_timepoints(y_true)                         # [B,T]
        yp = _pred_prob_timepoints(y_pred, self.model)       # [B,T]
        yt_bin = yt >= 0.5
        pred_bin_kbt = yp[None, ...] >= self.tau[:, None, None]  # [K,B,T]
        tp_k = tf.reduce_sum(tf.cast(pred_bin_kbt &  yt_bin[None, ...], tf.float32), axis=[1,2])
        fp_k = tf.reduce_sum(tf.cast(pred_bin_kbt & ~yt_bin[None, ...], tf.float32), axis=[1,2])
        fn_k = tf.reduce_sum(tf.cast(~pred_bin_kbt &  yt_bin[None, ...], tf.float32), axis=[1,2])
        self.tp.assign_add(tp_k); self.fp.assign_add(fp_k); self.fn.assign_add(fn_k)

    def result(self):
        eps = tf.keras.backend.epsilon()
        prec = self.tp / (self.tp + self.fp + eps)
        reca = self.tp / (self.tp + self.fn + eps)
        idx  = tf.argsort(reca)
        return tf.reduce_sum((tf.gather(reca, idx)[1:] - tf.gather(reca, idx)[:-1]) *
                             (tf.gather(prec, idx)[1:] + tf.gather(prec, idx)[:-1]) * 0.5)

    def reset_state(self):
        self.tp.assign(tf.zeros_like(self.tp))
        self.fp.assign(tf.zeros_like(self.fp))
        self.fn.assign(tf.zeros_like(self.fn))

# ---- helpers consistent with your current setup ----
def _yt_bt(y_true):               # [B,T,1] -> [B,T]
    return y_true[..., 0]

def _yp_bt(y_pred, model):        # [B,T,C] -> [B,T] probs
    return y_pred[..., getattr(model, "_cls_prob_index", 1)]

# ================= Sample-level MAX MCC =================
class SampleMaxMCC(tf.keras.metrics.Metric):
    """
    Timepoint-wise MCC computed across a threshold grid; returns max MCC.
    Robust to class imbalance; no voting or event logic.
    """
    def __init__(self, thresholds=tf.linspace(0., 1., 11),
                 name="sample_max_mcc", model=None, **kw):
        super().__init__(name=name, **kw)
        k = int(thresholds.shape[0])
        self.tau = tf.cast(thresholds, tf.float32)
        self.tp  = self.add_weight("tp",  shape=(k,), initializer="zeros")
        self.fp  = self.add_weight("fp",  shape=(k,), initializer="zeros")
        self.tn  = self.add_weight("tn",  shape=(k,), initializer="zeros")
        self.fn  = self.add_weight("fn",  shape=(k,), initializer="zeros")
        self.model = model

    def update_state(self, y_true, y_pred, **_):
        yt = _yt_bt(y_true)                         # [B,T]
        yp = _yp_bt(y_pred, self.model)             # [B,T]
        yt_bin = yt >= 0.5                          # [B,T]

        pred_kbt = yp[None, ...] >= self.tau[:, None, None]  # [K,B,T]

        tp = tf.reduce_sum(tf.cast(pred_kbt &  yt_bin[None, ...], tf.float32), axis=[1,2])
        fp = tf.reduce_sum(tf.cast(pred_kbt & ~yt_bin[None, ...], tf.float32), axis=[1,2])
        tn = tf.reduce_sum(tf.cast(~pred_kbt & ~yt_bin[None, ...], tf.float32), axis=[1,2])
        fn = tf.reduce_sum(tf.cast(~pred_kbt &  yt_bin[None, ...], tf.float32), axis=[1,2])

        self.tp.assign_add(tp); self.fp.assign_add(fp)
        self.tn.assign_add(tn); self.fn.assign_add(fn)

    def result(self):
        eps = tf.keras.backend.epsilon()
        num = self.tp * self.tn - self.fp * self.fn
        den = tf.sqrt((self.tp + self.fp) * (self.tp + self.fn) *
                      (self.tn + self.fp) * (self.tn + self.fn) + eps)
        mcc = num / den
        return tf.reduce_mean(mcc)

    def reset_state(self):
        for v in (self.tp, self.fp, self.tn, self.fn):
            v.assign(tf.zeros_like(v))


class SampleMaxF1(tf.keras.metrics.Metric):
    def __init__(self, thresholds=tf.linspace(0., 1., 11), name="sample_max_f1", model=None, **kw):
        super().__init__(name=name, **kw)
        k = int(thresholds.shape[0])  # OK for tf.linspace
        self.tau = tf.cast(thresholds, tf.float32)
        self.tp = self.add_weight("tp", shape=(k,), initializer="zeros")
        self.fp = self.add_weight("fp", shape=(k,), initializer="zeros")
        self.fn = self.add_weight("fn", shape=(k,), initializer="zeros")
        self.model = model

    def update_state(self, y_true, y_pred, **_):
        yt = _lab_timepoints(y_true)                             # [B,T]
        yp = _pred_prob_timepoints(y_pred, self.model)           # [B,T]
        yt_bin = yt >= 0.5

        pred_kbt = yp[None, ...] >= self.tau[:, None, None]      # [K,B,T]
        tp = tf.reduce_sum(tf.cast(pred_kbt &  yt_bin[None, ...], tf.float32), axis=[1,2])
        fp = tf.reduce_sum(tf.cast(pred_kbt & ~yt_bin[None, ...], tf.float32), axis=[1,2])
        fn = tf.reduce_sum(tf.cast(~pred_kbt &  yt_bin[None, ...], tf.float32), axis=[1,2])

        self.tp.assign_add(tp); self.fp.assign_add(fp); self.fn.assign_add(fn)

    def result(self):
        eps = tf.keras.backend.epsilon()
        prec = self.tp / (self.tp + self.fp + eps)
        reca = self.tp / (self.tp + self.fn + eps)
        f1   = 2.0 * prec * reca / (prec + reca + eps)
        return tf.reduce_mean(f1)

    def reset_state(self):
        for v in (self.tp, self.fp, self.fn): v.assign(tf.zeros_like(v))

# ================= Sample-level LATENCY SCORE =================
class LatencyScore(tf.keras.metrics.Metric):
    """
    Early-OK latency: score = exp(-max(0, det_idx - onset_idx)/tau).
    - Early detections (det_idx < onset_idx) get full credit (score=1).
    - Higher is better; 1.0 means zero (or negative) latency.
    - Uses thresholds in (0,1) to avoid the 'always 1' degeneracy.
    - Optional min_run to ignore single-frame spikes.
    """
    def __init__(self, thresholds=tf.linspace(0.05, 0.95, 11),
                 tau=16.0, min_run=1, name="latency_score", model=None, **kw):
        super().__init__(name=name, **kw)
        self.tau = tf.constant(float(tau), tf.float32)      # in samples
        self.tau_grid = tf.cast(thresholds, tf.float32)
        self.min_run = int(min_run)
        k = int(thresholds.shape[0])
        self.sum_score = self.add_weight("sum_score", shape=(k,), initializer="zeros")
        self.count     = self.add_weight("count",     shape=(k,), initializer="zeros")
        self.model = model

    @staticmethod
    def _first_true_idx(mat_bool):              # [B,T] -> (idx[B], has[B])
        idx = tf.argmax(tf.cast(mat_bool, tf.int32), axis=-1, output_type=tf.int32)
        has = tf.reduce_any(mat_bool, axis=-1)
        return idx, has

    @staticmethod
    def _enforce_min_run(pred_bt, k):           # pred_bt: [B,T] bool
        if k <= 1: 
            return pred_bt
        x = tf.cast(pred_bt, tf.float32)[..., None]      # [B,T,1]
        filt = tf.ones((k, 1, 1), x.dtype)               # [k,1,1]
        runsum = tf.nn.conv1d(x, filt, stride=1, padding="SAME")
        return runsum[..., 0] >= float(k)                # [B,T] bool

    def update_state(self, y_true, y_pred, **_):
        yt = y_true[..., 0]                                        # [B,T]
        yp = y_pred[..., getattr(self.model, "_cls_prob_index", 1)]# [B,T]
        yt_bin = yt >= 0.5

        onset_idx, has_ev = self._first_true_idx(yt_bin)           # [B], [B]
        pred_kbt = yp[None, ...] >= self.tau_grid[:, None, None]   # [K,B,T]

        def one_k(pred_bt):                                        # pred_bt: [B,T]
            if self.min_run > 1:
                pred_bt = self._enforce_min_run(pred_bt, self.min_run)
            det_idx, has_det = self._first_true_idx(pred_bt)       # first anywhere
            valid = has_ev & has_det                               # only windows with event & a detection
            lat = tf.cast(det_idx - onset_idx, tf.float32)         # can be negative (early)
            lat_pos = tf.nn.relu(lat)                              # early -> 0 penalty
            score = tf.exp(-lat_pos / self.tau)                    # [B]
            score = tf.where(valid, score, 0.0)
            return tf.reduce_sum(score), tf.reduce_sum(tf.cast(valid, tf.float32))

        sums, cnts = tf.map_fn(one_k, pred_kbt, fn_output_signature=(tf.float32, tf.float32))
        self.sum_score.assign_add(sums)
        self.count.assign_add(cnts)

    def result(self):
        eps = tf.keras.backend.epsilon()
        avg = self.sum_score / (self.count + eps)   # [K]
        return tf.reduce_max(avg)

    def reset_state(self):
        self.sum_score.assign(tf.zeros_like(self.sum_score))
        self.count.assign(tf.zeros_like(self.count))

# --------------------------------------------------------------------- #
# FALSE-POSITIVES PER MINUTE  (negative windows only)
# --------------------------------------------------------------------- #
class FPperMinMetric(tf.keras.metrics.Metric):
    def __init__(self, thresh=0.5, win_sec=64/2500,
                 mode="consec", consec_k=3, majority_ratio=0.0,
                 name="fp_per_min", model=None, **kw):
        super().__init__(name=name, **kw)
        self.tau  = float(thresh)
        self.wsec = tf.constant(win_sec, tf.float32)
        self.fp   = self.add_weight("fp",   initializer="zeros")
        self.nwin = self.add_weight("nwin", initializer="zeros")
        self.model = model
        self.mode = mode
        self.consec_k = int(consec_k)
        self.maj_ratio = float(majority_ratio)
        self.eps = tf.keras.backend.epsilon()

    def update_state(self, y_true, y_pred, **_):
        is_cls = getattr(self.model, "_is_classification_only", True)
        yt, yp = _ch_lab(y_true, is_cls), _ch(y_pred, is_cls)

        neg_win = tf.reduce_max(yt, 1) < 0.5                       # [B]
        pred_bin = yp >= self.tau                                  # [B,T]

        if self.mode == "consec":
            has_det = _has_run_k(pred_bin, self.consec_k)
        elif self.mode == "majority":
            has_det = _has_majority(pred_bin, self.maj_ratio)
        else:
            has_det = tf.reduce_any(pred_bin, 1)

        self.fp.assign_add(tf.reduce_sum(tf.cast(has_det & neg_win, tf.float32)))
        self.nwin.assign_add(tf.cast(tf.shape(yt)[0], tf.float32))

    def result(self):
        minutes = (self.nwin * self.wsec) / 60.0 + self.eps
        return self.fp / minutes

    def reset_state(self):
        self.fp.assign(0.); self.nwin.assign(0.)

# ================= Recall @ τ = 0.7 (sample-level) =================
class RecallAt0p7(tf.keras.metrics.Metric):
    """
    Sample-level recall at fixed threshold τ=0.7.
    Uses your helpers `_yt_bt` and `_yp_bt` and model._cls_prob_index.
    Monitored name: 'recall_at_0p7' -> 'val_recall_at_0p7'
    """
    def __init__(self, tau=0.7, name="recall_at_0p7", model=None, **kw):
        super().__init__(name=name, **kw)
        self.tau = float(tau)
        self.tp  = self.add_weight("tp", initializer="zeros", dtype=tf.float32)
        self.fn  = self.add_weight("fn", initializer="zeros", dtype=tf.float32)
        self.model = model

    def update_state(self, y_true, y_pred, **_):
        yt = _yt_bt(y_true)                     # [B,T] labels
        yp = _yp_bt(y_pred, self.model)         # [B,T] probs
        yt_pos  = yt >= 0.5
        predpos = yp >= self.tau
        tp = tf.reduce_sum(tf.cast(predpos & yt_pos, tf.float32))
        fn = tf.reduce_sum(tf.cast(~predpos & yt_pos, tf.float32))
        self.tp.assign_add(tp)
        self.fn.assign_add(fn)

    def result(self):
        eps = tf.keras.backend.epsilon()
        return self.tp / (self.tp + self.fn + eps)

    def reset_state(self):
        self.tp.assign(0.0); self.fn.assign(0.0)


# ============ FP per minute @ τ = 0.3 (window-level on negatives) ============
class FPperMinAt0p3(tf.keras.metrics.Metric):
    """
    False positives per minute at fixed threshold τ=0.3.
    Drop-in twin of your FPperMinMetric, using your helpers:
      - _ch_lab, _ch (to extract [B,T] labels/probs for classification)
      - _has_run_k / _has_majority (to enforce detection rule)
    Count a window as FP if any detection (per your rule) occurs in a negative window.
    Monitored name default: 'fp_per_min' -> 'val_fp_per_min'
    If you prefer a different monitor key, set name="fp_per_min_at_0p3".
    """
    def __init__(self,
                 win_sec,
                 name="fp_per_min",
                 model=None,
                 consec_k=3,
                 majority_ratio=0.0,
                 mode="consec",
                 tau=0.3,
                 **kw):
        super().__init__(name=name, **kw)
        self.tau  = float(tau)
        self.wsec = tf.constant(float(win_sec), tf.float32)
        self.fp   = self.add_weight("fp",   initializer="zeros", dtype=tf.float32)
        self.nwin = self.add_weight("nwin", initializer="zeros", dtype=tf.float32)
        self.model = model
        self.mode = str(mode)
        self.consec_k = int(consec_k)
        self.maj_ratio = float(majority_ratio)
        self.eps = tf.keras.backend.epsilon()

    def update_state(self, y_true, y_pred, **_):
        # Extract [B,T] labels/probs via your classification helpers.
        is_cls = getattr(self.model, "_is_classification_only", True)
        yt = _ch_lab(y_true, is_cls)   # [B,T]
        yp = _ch(y_pred, is_cls)       # [B,T]

        neg_win  = tf.reduce_max(yt, axis=1) < 0.5       # [B] windows with no positives
        pred_bin = yp >= self.tau                         # [B,T] binarized at τ=0.3

        if self.mode == "consec":
            has_det = _has_run_k(pred_bin, self.consec_k)     # [B]
        elif self.mode == "majority":
            has_det = _has_majority(pred_bin, self.maj_ratio) # [B]
        else:
            has_det = tf.reduce_any(pred_bin, axis=1)         # [B]

        # Count FP windows among negatives
        self.fp.assign_add(tf.reduce_sum(tf.cast(has_det & neg_win, tf.float32)))
        self.nwin.assign_add(tf.cast(tf.shape(yt)[0], tf.float32))

    def result(self):
        minutes = (self.nwin * self.wsec) / 60.0 + self.eps
        return self.fp / minutes

    def reset_state(self):
        self.fp.assign(0.0); self.nwin.assign(0.0)
        
# --------------------------------------------------------------------- #
# LOSS FN 
# --------------------------------------------------------------------- #
def mixed_latent_loss(horizon=0, loss_weight=1, params=None, model=None, this_op=None):
    import math
    import tensorflow as tf

    # ---------- helpers ----------
    def _ramp(step, delay, dur):
        step  = tf.cast(step,  tf.float32)
        delay = tf.cast(delay, tf.float32)
        dur   = tf.maximum(tf.cast(dur, tf.float32), 1.0)
        x = tf.clip_by_value((step - delay) / dur, 0.0, 1.0)
        return 0.5 - 0.5 * tf.cos(tf.constant(math.pi, tf.float32) * x)

    def _temporal_deltas(x_bt):  # [B,T] -> [B,T-1]
        return x_bt[:, 1:] - x_bt[:, :-1]

    def _temporal_penalty(deltas, kind="tMSE", tau=3.5):
        absd = tf.abs(deltas)
        if kind == "TV":
            return tf.reduce_mean(absd)                       # classic TV (L1)
        return tf.reduce_mean(tf.minimum(absd, tf.cast(tau, absd.dtype)))  # truncated-L1

    def _smooth_gate(x, k=8.0, eps=0.02):  # ≥0, smooth; avoids silent gradients
        return tf.nn.softplus(k * x) / k + eps

    # ---------- metric losses on time-averaged similarities ----------
    def circle_timeavg(a_btd, p_btd, n_btd, m=0.32, gamma=20.0):
        """
        Cosine per time, then mean over time:
          s_ap = mean_t <a_t, p_t>
          s_an[i,j] = mean_t <a_i,t, n_j,t>, s_pn similar.
        Circle is applied on these time-averaged sims (no feature pooling).
        """
        a = tf.math.l2_normalize(a_btd, axis=-1)  # [B,T,D]
        p = tf.math.l2_normalize(p_btd, axis=-1)
        n = tf.math.l2_normalize(n_btd, axis=-1)

        s_ap_bt = tf.reduce_sum(a * p, axis=-1)           # [B,T]
        s_ap = tf.reduce_mean(s_ap_bt, axis=1)            # [B]

        s_an_bbt = tf.einsum('btd,ktd->bkt', a, n)        # [B,B,T]
        s_pn_bbt = tf.einsum('btd,ktd->bkt', p, n)        # [B,B,T]
        s_an = tf.reduce_mean(s_an_bbt, axis=-1)          # [B,B]
        s_pn = tf.reduce_mean(s_pn_bbt, axis=-1)          # [B,B]

        alpha_p = _smooth_gate(m + 1.0 - s_ap)            # [B]
        alpha_n = _smooth_gate(s_an + m)                  # [B,B]
        delta_p = 1.0 - m

        neg_part_a = tf.reduce_logsumexp(gamma * alpha_n * (s_an - m), axis=1)  # [B]
        pos_part_a = gamma * alpha_p * (s_ap - delta_p)                          # [B]
        L_a = tf.nn.softplus(neg_part_a + pos_part_a)

        alpha_n_p = _smooth_gate(s_pn + m)
        neg_part_p = tf.reduce_logsumexp(gamma * alpha_n_p * (s_pn - m), axis=1)
        pos_part_p = gamma * alpha_p * (s_ap - delta_p)
        L_p = tf.nn.softplus(neg_part_p + pos_part_p)

        return tf.reduce_mean(0.5 * (L_a + L_p))

    def supcon_timeavg(a_btd, p_btd, n_btd, temperature=0.1):
        """
        SupCon over a similarity matrix built from time-averaged cosine sims.
        Class 1: A&P blocks; Class 0: N block.
        """
        B = tf.shape(a_btd)[0]
        z_all = tf.concat([a_btd, p_btd, n_btd], axis=0)         # [3B,T,D]
        z_all = tf.math.l2_normalize(z_all, axis=-1)

        sim_nmt = tf.einsum('ntd,mtd->nmt', z_all, z_all)        # [3B,3B,T]
        sim_nm = tf.reduce_mean(sim_nmt, axis=-1)                 # [3B,3B]

        M = tf.shape(sim_nm)[0]
        logits = sim_nm / tf.cast(temperature, sim_nm.dtype)
        eye = tf.eye(M, dtype=logits.dtype)
        logits = logits - 1e9 * eye

        labels = tf.concat([tf.ones([2*B], tf.int32), tf.zeros([B], tf.int32)], axis=0)
        pos = tf.cast(tf.equal(labels[:, None], labels[None, :]), logits.dtype) - eye

        log_prob = logits - tf.reduce_logsumexp(logits, axis=1, keepdims=True)
        pos_cnt = tf.reduce_sum(pos, axis=1)
        loss_vec = -tf.reduce_sum(pos * log_prob, axis=1) / (pos_cnt + 1e-8)
        return tf.reduce_mean(loss_vec)

    ema_metric = tf.Variable(1.0, trainable=False, dtype=tf.float32, name="ema_metric_loss")
    ema_clf    = tf.Variable(1.0, trainable=False, dtype=tf.float32, name="ema_clf_loss")

    @tf.function
    def loss_fn(y_true, y_pred):
        # --- triplet split ---
        a_out, p_out, n_out = tf.split(y_pred, 3, axis=0)
        a_true, p_true, n_true = tf.split(y_true, 3, axis=0)

        # --- heads ---
        logit_idx = getattr(model, '_cls_logit_index', 0)
        prob_idx  = getattr(model, '_cls_prob_index',  1)
        emb_start = getattr(model, '_emb_start_index', 2)

        a_logit = a_out[..., logit_idx];  p_logit = p_out[..., logit_idx];  n_logit = n_out[..., logit_idx]  # [B,T]
        a_prob  = a_out[..., prob_idx];   p_prob  = p_out[..., prob_idx];   n_prob  = n_out[..., prob_idx]   # [B,T]
        a_emb   = a_out[..., emb_start:]; p_emb   = p_out[..., emb_start:]; n_emb   = n_out[..., emb_start:] # [B,T,D]

        a_lab = tf.cast(a_true[..., 0], tf.float32)
        p_lab = tf.cast(p_true[..., 0], tf.float32)
        n_lab = tf.cast(n_true[..., 0], tf.float32)

        # --- ramps (same keys as before) ---
        it = tf.cast(model.optimizer.iterations, tf.float32)
        total_steps = float(params.get('TOTAL_STEPS', 100000))

        r = _ramp(it,
                  float(params.get('RAMP_DELAY', 0.01 * total_steps)),
                  float(params.get('RAMP_STEPS', 0.25 * total_steps)))

        # Metric weights
        w_circle   = tf.cast(params.get('LOSS_Circle', 60.0), tf.float32)
        w_supcon_t = tf.cast(params.get('LOSS_SupCon', 0.5),  tf.float32)  # ramp to target and stay
        w_supcon   = r * w_supcon_t

        # Negatives ramp
        neg_min    = float(params.get('LOSS_NEGATIVES_MIN', 4.0))
        neg_target = float(params.get('LOSS_NEGATIVES', 26.0))
        r_neg = _ramp(it,
                      float(params.get('NEG_RAMP_DELAY', 0.05 * total_steps)),
                      float(params.get('NEG_RAMP_STEPS', 0.45 * total_steps)))
        w_neg = tf.cast(neg_min + r_neg * max(0.0, neg_target - neg_min), tf.float32)

        # --- metric (NO pooling) ---
        m = float(params.get('CIRCLE_m', 0.32))
        g = float(params.get('CIRCLE_gamma', 20.0))
        L_circle_raw = circle_timeavg(a_emb, p_emb, n_emb, m=m, gamma=g)
        L_sup_raw    = supcon_timeavg(a_emb, p_emb, n_emb, temperature=float(params.get('SUPCON_T', 0.1)))
        metric_loss  = w_circle * L_circle_raw + w_supcon * L_sup_raw

        # --- BCE (separate A/P/N weights) ---
        def bce_logits(y, z):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z))

        w_a = float(params.get('BCE_ANC_ALPHA', 2.0))   # anchor
        w_p = float(params.get('BCE_POS_ALPHA', 2.0))   # positive
        cls_a = bce_logits(a_lab, a_logit)
        cls_p = bce_logits(p_lab, p_logit)
        cls_n = bce_logits(n_lab, n_logit)
        classification_loss = w_a * cls_a + w_p * cls_p + w_neg * cls_n

        # --- temporal smoothing (truncated change on logits) ---
        lam_tv   = tf.cast(params.get('LOSS_TV', 0.30), tf.float32)
        r_tv     = _ramp(it,
                         float(params.get('TV_DELAY', 0.10 * total_steps)),
                         float(params.get('TV_DUR',   0.30 * total_steps)))
        # fixed, robust defaults
        smooth_type  = params.get('SMOOTH_TYPE',  'tMSE')     # "tMSE" or "TV"
        smooth_space = params.get('SMOOTH_SPACE', 'logit')    # "logit" | "prob" | "logprob"
        tau_fixed    = float(params.get('SMOOTH_TAU', 3.5))

        if smooth_space == 'logit':
            a_seq, p_seq, n_seq = a_logit, p_logit, n_logit
        elif smooth_space == 'prob':
            eps = tf.constant(1e-8, tf.float32)
            a_seq = tf.clip_by_value(a_prob, eps, 1.0 - eps)
            p_seq = tf.clip_by_value(p_prob, eps, 1.0 - eps)
            n_seq = tf.clip_by_value(n_prob, eps, 1.0 - eps)
        else:  # 'logprob'
            eps = tf.constant(1e-8, tf.float32)
            a_seq = tf.math.log(tf.clip_by_value(a_prob, eps, 1.0))
            p_seq = tf.math.log(tf.clip_by_value(p_prob, eps, 1.0))
            n_seq = tf.math.log(tf.clip_by_value(n_prob, eps, 1.0))

        da = _temporal_deltas(a_seq); dp = _temporal_deltas(p_seq); dn = _temporal_deltas(n_seq)
        tpen = _temporal_penalty(da, kind=smooth_type, tau=tau_fixed) \
             + _temporal_penalty(dp, kind=smooth_type, tau=tau_fixed) \
             + _temporal_penalty(dn, kind=smooth_type, tau=tau_fixed)
        classification_loss = classification_loss + (lam_tv * r_tv) * tpen

        # --- ratio trick + mild throttle (names unchanged) ---
        alpha = tf.constant(0.01, tf.float32)  # slow & stable
        ema_metric.assign((1.0 - alpha) * ema_metric + alpha * tf.stop_gradient(L_circle_raw + L_sup_raw))
        ema_clf.assign(   (1.0 - alpha) * ema_clf    + alpha * tf.stop_gradient(classification_loss))

        ratio = tf.clip_by_value(ema_metric / (ema_clf + 1e-6), 0.1, 1.0)  # never > 1


        # classfication 
        clf_scale = tf.cast(params.get("CLF_SCALE", 0.30), tf.float32)
        clf_delay = float(params.get("CLF_RAMP_DELAY", params.get("RAMP_DELAY", 0.01 * total_steps)))
        clf_dur   = float(params.get("CLF_RAMP_STEPS", params.get("RAMP_STEPS", 0.25 * total_steps)))
        r_clf     = _ramp(it, clf_delay, clf_dur)   # 0 → 1
        w_clf     = clf_scale * r_clf

        total = tf.reduce_mean(metric_loss) + (w_clf * ratio) * tf.reduce_mean(classification_loss)
        return total

    return loss_fn

def class_finetune(horizon=0, loss_weight=1, params=None, model=None, this_op=None):

    def pair_mask(m_bt):                 # [B,T] -> [B,T-1], mask for adjacent pairs
        return tf.minimum(m_bt[:,1:], m_bt[:,:-1])

    def log_sigmoid(z):                  # stable log σ(z) == log-prob
        return -tf.nn.softplus(-z)

    def tv_trunc_logp_masked(p_bt, m_bt, tau=4.0):
        """Symmetric TV on log-prob, masked, with truncation."""
        lp = tf.math.log(tf.clip_by_value(p_bt, 1e-6, 1.0))   # [B,T]
        d  = tf.abs(lp[:,1:] - lp[:,:-1])                     # [B,T-1]
        d  = tf.minimum(d, tau)
        pm = pair_mask(m_bt)
        num = tf.reduce_sum(d * pm)
        den = tf.reduce_sum(pm) + EPS
        return num / den

    def tv_up_logp_from_logits_masked(z_bt, m_bt, tau=4.0):
        """Up-only TV on log-prob from logits (stable at tiny p), masked, truncated."""
        lp = log_sigmoid(z_bt)                                 # log p(z)
        d  = lp[:,1:] - lp[:,:-1]                              # ↑ means p going up
        d  = tf.minimum(tf.nn.relu(d), tau)                    # up-only + trunc
        pm = pair_mask(m_bt)
        num = tf.reduce_sum(d * pm)
        den = tf.reduce_sum(pm) + EPS
        return num / den
    
    def bce_logits_weighted(y_bt, logit_bt, w_bt):
        # per-timepoint BCE
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_bt, logits=logit_bt)  # [B,T]
        w   = tf.cast(w_bt, tf.float32)
        num = tf.reduce_sum(bce * w)
        den = tf.reduce_sum(w) + tf.keras.backend.epsilon()
        return num / den    
    
    def focal_logits_weighted(y_bt, logit_bt, w_bt, alpha=0.25, gamma=2.0):
        # y_bt, logit_bt, w_bt: [B, T]
        focal = tf.keras.losses.BinaryFocalCrossentropy(
            alpha=alpha, gamma=gamma, from_logits=True,
            apply_class_balancing=True,
            reduction=tf.keras.losses.Reduction.NONE, axis=-1
        )
        # Add a singleton channel so Keras doesn't try to squeeze weird dims
        y = tf.cast(y_bt, tf.float32)[..., None]      # [B, T, 1]
        z = tf.cast(logit_bt, tf.float32)[..., None]  # [B, T, 1]
        # Per-element loss -> [B, T]
        loss_bt = focal(y, z)
        w = tf.cast(w_bt, tf.float32)                 # [B, T]
        
        # Safe masked mean
        num = tf.reduce_sum(loss_bt * w)
        den = tf.reduce_sum(w) + tf.keras.backend.epsilon()
        return num / den
    
    # --- post-onset mask for A/P (fixes dtype mismatch robustly) ---
    def post_onset_mask(y_bt):
        """
        y_bt: [B,T] float in {0,1}
        returns: [B,T] float mask in {0,1}
        - For windows with any positive (>=0.5): mask=1 from onset onward, else 0 before.
        - For windows with no positive: return all ones (do NOT drop them).
        """
        yb = y_bt >= 0.5                        # bool [B,T]
        # First positive index and has-event flag
        idx = tf.argmax(tf.cast(yb, tf.int32), axis=1, output_type=tf.int32)   # [B] int32
        has = tf.reduce_any(yb, axis=1)                                        # [B] bool

        B = tf.shape(y_bt)[0]
        T = tf.shape(y_bt)[1]
        rng = tf.range(T, dtype=tf.int32)[None, :]                              # [1,T] int32
        mpos = tf.cast(rng >= idx[:, None], tf.float32)                         # [B,T] 1 from onset onward

        # no-onset windows -> keep whole window (all ones)
        return tf.where(has[:, None], mpos, tf.ones((B, T), tf.float32))


    @tf.function
    def loss_fn(y_true, y_pred):
        # ---- split triplet on batch axis ----
        a_out, p_out, n_out = tf.split(y_pred, 3, axis=0)
        a_true, p_true, n_true = tf.split(y_true, 3, axis=0)

        # ---- head indices from model ----
        logit_idx = getattr(model, '_cls_logit_index', 0)
        prob_idx  = getattr(model, '_cls_prob_index',  1)
        emb_start = getattr(model, '_emb_start_index', 2)

        # ---- slice logits / embeddings ----
        a_logit = a_out[..., logit_idx]      # [B,T]
        p_logit = p_out[..., logit_idx]      # [B,T]
        n_logit = n_out[..., logit_idx]      # [B,T]
        a_prob = a_out[..., prob_idx]        # [B,T]
        p_prob = p_out[..., prob_idx]        # [B,T]
        n_prob = n_out[..., prob_idx]        # [B,T]
        a_emb   = a_out[..., emb_start:]     # [B,T,D]
        p_emb   = p_out[..., emb_start:]     # [B,T,D]
        n_emb   = n_out[..., emb_start:]     # [B,T,D]

        # ---- labels ----
        a_lab = tf.cast(a_true[..., 0], tf.float32)  # [B,T] (0/1)
        p_lab = tf.cast(p_true[..., 0], tf.float32)  # [B,T]
        n_lab = tf.cast(n_true[..., 0], tf.float32)  # [B,T] (all zeros by design)

        # ---- ramps ----
        it = tf.cast(model.optimizer.iterations, tf.float32)
        total_steps = float(params.get('TOTAL_STEPS', 100000))

        # ---- Classification BCE with post-onset masking ----
        a_mask = post_onset_mask(a_lab)                 # [B,T]
        p_mask = post_onset_mask(p_lab)                 # [B,T]
        n_mask = tf.ones_like(n_lab, tf.float32)        # keep all negatives

        # cls_a = bce_logits_weighted(a_lab, a_logit, a_mask)
        # cls_p = bce_logits_weighted(p_lab, p_logit, p_mask)
        # cls_n = bce_logits_weighted(n_lab, n_logit, n_mask)

        alpha = float(params.get('FOCAL_ALPHA', 0.25))
        gamma = float(params.get('FOCAL_GAMMA', 2.0))
        cls_a = focal_logits_weighted(a_lab, a_logit, a_mask,  alpha=alpha, gamma=gamma)
        cls_p = focal_logits_weighted(p_lab, p_logit, p_mask,  alpha=alpha, gamma=gamma)
        cls_n = focal_logits_weighted(n_lab, n_logit, n_mask,  alpha=alpha, gamma=gamma)

        alpha_pos = float(params.get('BCE_POS_ALPHA', 2.0))
        print('alpha_pos:', alpha_pos)
        alpha_neg = float(params.get('LOSS_NEGATIVES', 20.0))
        print('alpha_neg:', alpha_neg)

        lam_tv = float(params.get('LOSS_TV', 0.3))
        print('loss_tv:', lam_tv)

        tv_term = 2 * lam_tv * (
                    tv_trunc_logp_masked(a_prob, n_mask, tau=4.0) +         # A: smooth after onset
                    tv_trunc_logp_masked(p_prob, n_mask, tau=4.0) +         # P: same
                    tv_trunc_logp_masked(n_logit, n_mask, tau=4.0) # N: suppress upward drift
                )
        classification_loss = cls_a + alpha_pos * cls_p + alpha_neg * cls_n + tv_term
        return tf.reduce_mean(classification_loss)

    return loss_fn

def mixed_mpn_loss(horizon=0, loss_weight=1, params=None, model=None, this_op=None):
    # ----- helpers -----
    def _mean_pool(x): return tf.reduce_mean(x, axis=1)

    # cosine ramp in [0,1]
    def _ramp(step, delay, dur):
        step  = tf.cast(step, tf.float32)
        delay = tf.cast(delay, tf.float32)
        dur   = tf.maximum(tf.cast(dur, tf.float32), 1.0)
        x = tf.clip_by_value((step - delay) / dur, 0.0, 1.0)
        return 0.5 - 0.5 * tf.cos(tf.constant(math.pi, tf.float32) * x)  # 0→1

    def tv_on_logits(z):  # z: [B,T]
        return tf.reduce_mean(tf.abs(z[:, 1:] - z[:, :-1]))

    # ----- metric losses (unchanged math) -----
    def mpn_tuple_loss(z_a_raw, z_p_raw, z_n_raw, *, margin_hard=1.0, margin_weak=0.1, lambda_pull=0.1, exclude_self=True):
        z_a = tf.reduce_mean(tf.cast(z_a_raw, tf.float32), axis=1)
        z_p = tf.reduce_mean(tf.cast(z_p_raw, tf.float32), axis=1)
        z_n = tf.reduce_mean(tf.cast(z_n_raw, tf.float32), axis=1)
        B   = tf.shape(z_a)[0]

        # pull
        d_pair = tf.reduce_sum(tf.square(z_a - z_p), axis=1) + 1e-8
        L_pull = lambda_pull * d_pair

        def _mask_self(mat):
            if exclude_self:
                mask = tf.eye(B, dtype=tf.bool)
                return tf.where(mask, tf.fill(tf.shape(mat), tf.constant(1e9, tf.float32)), mat)
            return mat

        d_ap = tf.reduce_sum(tf.square(z_a[:,None,:] - z_p[None,:,:]), axis=2) + 1e-8
        d_ap = _mask_self(d_ap)
        L_weak = tf.reduce_mean(tf.nn.relu(margin_weak + d_pair[:,None] - d_ap))
        d_an = tf.reduce_sum(tf.square(z_a[:,None,:] - z_n[None,:,:]), axis=2) + 1e-8
        d_an = _mask_self(d_an)
        lifted = tf.reduce_logsumexp(margin_hard - d_an, axis=1)
        L_hard = tf.nn.relu(d_pair + lifted)

        return tf.reduce_mean(L_pull + L_weak + L_hard)

    def supcon_ripple(z_a_raw, z_p_raw, z_n_raw, *, temperature=0.1):
        z_a = _mean_pool(z_a_raw); z_p = _mean_pool(z_p_raw); z_n = _mean_pool(z_n_raw)
        z_all = tf.math.l2_normalize(tf.concat([z_a, z_p, z_n], axis=0), axis=1)
        M = tf.shape(z_all)[0]
        sim = tf.matmul(z_all, z_all, transpose_b=True) / temperature

        B = tf.shape(z_a)[0]
        labels = tf.concat([tf.ones(2*B, tf.int32), tf.zeros(B, tf.int32)], axis=0)
        pos = tf.cast(tf.equal(labels[:,None], labels[None,:]), tf.float32) - tf.eye(M, dtype=tf.float32)

        logits = sim - 1e9 * tf.eye(M)
        log_prob = logits - tf.reduce_logsumexp(logits, axis=1, keepdims=True)

        pos_cnt = tf.reduce_sum(pos, axis=1)
        loss_vec = -tf.reduce_sum(pos * log_prob, axis=1) / (pos_cnt + 1e-8)
        return tf.reduce_mean(loss_vec)

    @tf.function
    def loss_fn(y_true, y_pred):
        a_out, p_out, n_out = tf.split(y_pred, 3, axis=0)
        a_true, p_true, n_true = tf.split(y_true, 3, axis=0)

        logit_idx = getattr(model, '_cls_logit_index', 0)
        prob_idx  = getattr(model, '_cls_prob_index',  1)
        emb_start = getattr(model, '_emb_start_index', 2)

        a_logit = a_out[..., logit_idx];  p_logit = p_out[..., logit_idx];  n_logit = n_out[..., logit_idx]
        a_prob  = a_out[..., prob_idx];   p_prob  = p_out[..., prob_idx];   n_prob  = n_out[..., prob_idx]
        a_emb   = a_out[..., emb_start:]; p_emb   = p_out[..., emb_start:]; n_emb   = n_out[..., emb_start:]

        a_lab = tf.cast(a_true[..., 0], tf.float32)
        p_lab = tf.cast(p_true[..., 0], tf.float32)
        n_lab = tf.cast(n_true[..., 0], tf.float32)

        # --------- ramps (READ-ONLY) ----------
        it = tf.cast(model.optimizer.iterations, tf.float32)
        total_steps = float(params.get('TOTAL_STEPS', 100000))

        # Metric ramps
        ramp_delay = tf.cast(params.get('RAMP_DELAY', int(0.01*total_steps)), tf.float32)
        ramp_steps = tf.cast(params.get('RAMP_STEPS', int(0.25*total_steps)), tf.float32)

        w_sup_tgt = tf.cast(params.get('LOSS_SupCon', 1.0), tf.float32)
        w_mpn_tgt = tf.cast(params.get('LOSS_TupMPN', 1.0), tf.float32)
        w_neg_tgt = tf.cast(params.get('LOSS_NEGATIVES', 2.0), tf.float32)
        
        r = _ramp(it, ramp_delay, ramp_steps)  # 0→1 after delay
        w_supcon = r * w_sup_tgt
        w_tupMPN = r * w_mpn_tgt

        # Negatives ramp (MIN → target)
        # Negatives ramp: gentle 25–40% of run
        neg_min = params.setdefault('LOSS_NEGATIVES_MIN', 4.0)  # softer start to reduce FP early
        neg_delay = params.setdefault('NEG_RAMP_DELAY', int(0.05* total_steps))
        neg_steps = params.setdefault('NEG_RAMP_STEPS', int(0.45 * total_steps))
        neg_steps = tf.cast(neg_steps if neg_steps is not None else params.get('RAMP_STEPS', 1), tf.float32)
        neg_steps = tf.maximum(neg_steps, 1.0)

        r_neg = _ramp(it, neg_delay, neg_steps)
        loss_fp_weight = neg_min + r_neg * tf.maximum(0.0, w_neg_tgt - neg_min)

        # --------- metric learning ----------
        L_mpn_raw = mpn_tuple_loss(a_emb, p_emb, n_emb, margin_hard=1.0, margin_weak=0.1, lambda_pull=0.1)
        L_sup_raw = supcon_ripple(a_emb, p_emb, n_emb, temperature=params.get('SUPCON_T', 0.1))
        metric_loss = w_tupMPN * L_mpn_raw + w_supcon * L_sup_raw

        # --------- BCE on logits ----------
        def bce_logits(y, z):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z))

        cls_a = bce_logits(a_lab, a_logit)
        cls_p = bce_logits(p_lab, p_logit)
        cls_n = bce_logits(n_lab, n_logit)

        alpha_pos = tf.cast(params.get('BCE_POS_ALPHA', 2.0), tf.float32)
        classification_loss = cls_a + alpha_pos * cls_p + loss_fp_weight * cls_n

        # --------- logit-TV smoothing ----------
        lam_tv = tf.cast(params.get('LOSS_TV', 0.02), tf.float32)
        print(f"TV weight: {lam_tv}")
        tv_term = tv_on_logits(a_logit) + tv_on_logits(p_logit) + tv_on_logits(n_logit)
        classification_loss = classification_loss + lam_tv * tv_term

        # --------- re-scaling (ratio trick) ----------
        ratio = tf.stop_gradient(
            (tf.reduce_mean(L_mpn_raw + L_sup_raw)) /
            (tf.reduce_mean(classification_loss) + 1e-6)
        )
        ratio = tf.clip_by_value(ratio, 0.1, 10.0)

        total = tf.reduce_mean(metric_loss) + 0.5 * ratio * tf.reduce_mean(classification_loss)

        return total
    return loss_fn


def mixed_mpnFocal_loss(horizon=0, loss_weight=1, params=None, model=None, this_op=None):
    # ----- helpers -----
    def _mean_pool(x): return tf.reduce_mean(x, axis=1)

    # cosine ramp in [0,1]
    def _ramp(step, delay, dur):
        step  = tf.cast(step, tf.float32)
        delay = tf.cast(delay, tf.float32)
        dur   = tf.maximum(tf.cast(dur, tf.float32), 1.0)
        x = tf.clip_by_value((step - delay) / dur, 0.0, 1.0)
        return 0.5 - 0.5 * tf.cos(tf.constant(math.pi, tf.float32) * x)  # 0→1

    def tv_on_logits(z):  # z: [B,T]
        return tf.reduce_mean(tf.abs(z[:, 1:] - z[:, :-1]))

    # ----- metric losses (unchanged math) -----
    def mpn_tuple_loss(z_a_raw, z_p_raw, z_n_raw, *, margin_hard=1.0, margin_weak=0.1, lambda_pull=0.1, exclude_self=True):
        z_a = tf.reduce_mean(tf.cast(z_a_raw, tf.float32), axis=1)
        z_p = tf.reduce_mean(tf.cast(z_p_raw, tf.float32), axis=1)
        z_n = tf.reduce_mean(tf.cast(z_n_raw, tf.float32), axis=1)
        B   = tf.shape(z_a)[0]

        # pull
        d_pair = tf.reduce_sum(tf.square(z_a - z_p), axis=1) + 1e-8
        L_pull = lambda_pull * d_pair

        def _mask_self(mat):
            if exclude_self:
                mask = tf.eye(B, dtype=tf.bool)
                return tf.where(mask, tf.fill(tf.shape(mat), tf.constant(1e9, tf.float32)), mat)
            return mat

        d_ap = tf.reduce_sum(tf.square(z_a[:,None,:] - z_p[None,:,:]), axis=2) + 1e-8
        d_ap = _mask_self(d_ap)
        L_weak = tf.reduce_mean(tf.nn.relu(margin_weak + d_pair[:,None] - d_ap))
        d_an = tf.reduce_sum(tf.square(z_a[:,None,:] - z_n[None,:,:]), axis=2) + 1e-8
        d_an = _mask_self(d_an)
        lifted = tf.reduce_logsumexp(margin_hard - d_an, axis=1)
        L_hard = tf.nn.relu(d_pair + lifted)

        return tf.reduce_mean(L_pull + L_weak + L_hard)

    def supcon_ripple(z_a_raw, z_p_raw, z_n_raw, *, temperature=0.1):
        z_a = _mean_pool(z_a_raw); z_p = _mean_pool(z_p_raw); z_n = _mean_pool(z_n_raw)
        z_all = tf.math.l2_normalize(tf.concat([z_a, z_p, z_n], axis=0), axis=1)
        M = tf.shape(z_all)[0]
        sim = tf.matmul(z_all, z_all, transpose_b=True) / temperature

        B = tf.shape(z_a)[0]
        labels = tf.concat([tf.ones(2*B, tf.int32), tf.zeros(B, tf.int32)], axis=0)
        pos = tf.cast(tf.equal(labels[:,None], labels[None,:]), tf.float32) - tf.eye(M, dtype=tf.float32)

        logits = sim - 1e9 * tf.eye(M)
        log_prob = logits - tf.reduce_logsumexp(logits, axis=1, keepdims=True)

        pos_cnt = tf.reduce_sum(pos, axis=1)
        loss_vec = -tf.reduce_sum(pos * log_prob, axis=1) / (pos_cnt + 1e-8)
        return tf.reduce_mean(loss_vec)

    @tf.function
    def loss_fn(y_true, y_pred):
        a_out, p_out, n_out = tf.split(y_pred, 3, axis=0)
        a_true, p_true, n_true = tf.split(y_true, 3, axis=0)

        logit_idx = getattr(model, '_cls_logit_index', 0)
        prob_idx  = getattr(model, '_cls_prob_index',  1)
        emb_start = getattr(model, '_emb_start_index', 2)

        a_logit = a_out[..., logit_idx];  p_logit = p_out[..., logit_idx];  n_logit = n_out[..., logit_idx]
        a_prob  = a_out[..., prob_idx];   p_prob  = p_out[..., prob_idx];   n_prob  = n_out[..., prob_idx]
        a_emb   = a_out[..., emb_start:]; p_emb   = p_out[..., emb_start:]; n_emb   = n_out[..., emb_start:]

        a_lab = tf.cast(a_true[..., 0], tf.float32)
        p_lab = tf.cast(p_true[..., 0], tf.float32)
        n_lab = tf.cast(n_true[..., 0], tf.float32)

        # --------- ramps (READ-ONLY) ----------
        it = tf.cast(model.optimizer.iterations, tf.float32)
        total_steps = float(params.get('TOTAL_STEPS', 100000))

        # Metric ramps
        ramp_delay = tf.cast(params.get('RAMP_DELAY', int(0.01*total_steps)), tf.float32)
        ramp_steps = tf.cast(params.get('RAMP_STEPS', int(0.25*total_steps)), tf.float32)

        w_sup_tgt = tf.cast(params.get('LOSS_SupCon', 1.0), tf.float32)
        w_mpn_tgt = tf.cast(params.get('LOSS_TupMPN', 1.0), tf.float32)
        w_neg_tgt = tf.cast(params.get('LOSS_NEGATIVES', 2.0), tf.float32)
        
        r = _ramp(it, ramp_delay, ramp_steps)  # 0→1 after delay
        w_supcon = r * w_sup_tgt
        w_tupMPN = r * w_mpn_tgt

        # Negatives ramp (MIN → target)
        # Negatives ramp: gentle 25–40% of run
        neg_min = params.setdefault('LOSS_NEGATIVES_MIN', 4.0)  # softer start to reduce FP early
        neg_delay = params.setdefault('NEG_RAMP_DELAY', int(0.05* total_steps))
        neg_steps = params.setdefault('NEG_RAMP_STEPS', int(0.45 * total_steps))
        neg_steps = tf.cast(neg_steps if neg_steps is not None else params.get('RAMP_STEPS', 1), tf.float32)
        neg_steps = tf.maximum(neg_steps, 1.0)

        r_neg = _ramp(it, neg_delay, neg_steps)
        loss_fp_weight = neg_min + r_neg * tf.maximum(0.0, w_neg_tgt - neg_min)

        # --------- metric learning ----------
        L_mpn_raw = mpn_tuple_loss(a_emb, p_emb, n_emb, margin_hard=1.0, margin_weak=0.1, lambda_pull=0.1)
        L_sup_raw = supcon_ripple(a_emb, p_emb, n_emb, temperature=params.get('SUPCON_T', 0.1))
        metric_loss = w_tupMPN * L_mpn_raw + w_supcon * L_sup_raw

        # --------- BCE on logits ----------
        def bce_logits(y, z):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z))

        cls_a = bce_logits(a_lab, a_logit)
        cls_p = bce_logits(p_lab, p_logit)
        cls_n = bce_logits(n_lab, n_logit)

        alpha_pos = tf.cast(params.get('BCE_POS_ALPHA', 2.0), tf.float32)
        classification_loss = cls_a + alpha_pos * cls_p + loss_fp_weight * cls_n

        # --------- logit-TV smoothing ----------
        lam_tv = tf.cast(params.get('LOSS_TV', 0.02), tf.float32)
        print(f"TV weight: {lam_tv}")
        tv_term = tv_on_logits(a_logit) + tv_on_logits(p_logit) + tv_on_logits(n_logit)
        classification_loss = classification_loss + lam_tv * tv_term

        # --------- re-scaling (ratio trick) ----------
        ratio = tf.stop_gradient(
            (tf.reduce_mean(L_mpn_raw + L_sup_raw)) /
            (tf.reduce_mean(classification_loss) + 1e-6)
        )
        ratio = tf.clip_by_value(ratio, 0.1, 10.0)

        total = tf.reduce_mean(metric_loss) + 0.5 * ratio * tf.reduce_mean(classification_loss)

        return total
    return loss_fn

def mixed_circle_loss(horizon=0, loss_weight=1, params=None, model=None, this_op=None):
    import math
    import tensorflow as tf

    # ---------- helpers ----------
    def _mean_pool(x): return tf.reduce_mean(x, axis=1)
    
    def _mean_pool_bt(x):               # [B,T,D] -> [B,D]
        return tf.reduce_mean(tf.cast(x, tf.float32), axis=1)

    def _ramp(step, delay, dur):        # cosine 0→1
        step  = tf.cast(step,  tf.float32)
        delay = tf.cast(delay, tf.float32)
        dur   = tf.maximum(tf.cast(dur, tf.float32), 1.0)
        x = tf.clip_by_value((step - delay) / dur, 0.0, 1.0)
        return 0.5 - 0.5 * tf.cos(tf.constant(math.pi, tf.float32) * x)

    def tv_on_logits(z_bt):             # z: [B,T]
        return tf.reduce_mean(tf.abs(z_bt[:, 1:] - z_bt[:, :-1]))

    # TV on logits, masked as well
    def tv_on_logits_masked(z_bt, m_bt):
        z  = z_bt[:, 1:] - z_bt[:, :-1]      # [B,T-1]
        m  = tf.minimum(m_bt[:, 1:], m_bt[:, :-1])
        num = tf.reduce_sum(tf.abs(z) * m)
        den = tf.reduce_sum(m) + tf.keras.backend.epsilon()
        return num / den
    
    def pair_mask(m_bt):                 # [B,T] -> [B,T-1], mask for adjacent pairs
        return tf.minimum(m_bt[:,1:], m_bt[:,:-1])

    def log_sigmoid(z):                  # stable log σ(z) == log-prob
        return -tf.nn.softplus(-z)

    def tv_trunc_logp_masked(p_bt, m_bt, tau=4.0):
        """Symmetric TV on log-prob, masked, with truncation."""
        lp = tf.math.log(tf.clip_by_value(p_bt, 1e-6, 1.0))   # [B,T]
        d  = tf.abs(lp[:,1:] - lp[:,:-1])                     # [B,T-1]
        d  = tf.minimum(d, tau)
        pm = pair_mask(m_bt)
        num = tf.reduce_sum(d * pm)
        den = tf.reduce_sum(pm) + EPS
        return num / den

    def tv_up_logp_from_logits_masked(z_bt, m_bt, tau=4.0):
        """Up-only TV on log-prob from logits (stable at tiny p), masked, truncated."""
        lp = log_sigmoid(z_bt)                                 # log p(z)
        d  = lp[:,1:] - lp[:,:-1]                              # ↑ means p going up
        d  = tf.minimum(tf.nn.relu(d), tau)                    # up-only + trunc
        pm = pair_mask(m_bt)
        num = tf.reduce_sum(d * pm)
        den = tf.reduce_sum(pm) + EPS
        return num / den
    
    # --- post-onset mask for A/P (fixes dtype mismatch robustly) ---
    def post_onset_mask(y_bt):
        """
        y_bt: [B,T] float in {0,1}
        returns: [B,T] float mask in {0,1}
        - For windows with any positive (>=0.5): mask=1 from onset onward, else 0 before.
        - For windows with no positive: return all ones (do NOT drop them).
        """
        yb = y_bt >= 0.5                        # bool [B,T]
        # First positive index and has-event flag
        idx = tf.argmax(tf.cast(yb, tf.int32), axis=1, output_type=tf.int32)   # [B] int32
        has = tf.reduce_any(yb, axis=1)                                        # [B] bool

        B = tf.shape(y_bt)[0]
        T = tf.shape(y_bt)[1]
        rng = tf.range(T, dtype=tf.int32)[None, :]                              # [1,T] int32
        mpos = tf.cast(rng >= idx[:, None], tf.float32)                         # [B,T] 1 from onset onward

        # no-onset windows -> keep whole window (all ones)
        return tf.where(has[:, None], mpos, tf.ones((B, T), tf.float32))

    # --- weighted BCE (time-masked), safe reduction ---
    def bce_logits_weighted(y_bt, logit_bt, w_bt):
        # per-timepoint BCE
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_bt, logits=logit_bt)  # [B,T]
        w   = tf.cast(w_bt, tf.float32)
        num = tf.reduce_sum(bce * w)
        den = tf.reduce_sum(w) + tf.keras.backend.epsilon()
        return num / den

    def supcon_ripple(z_a_raw, z_p_raw, z_n_raw, *, temperature=0.1):
        z_a = _mean_pool(z_a_raw); z_p = _mean_pool(z_p_raw); z_n = _mean_pool(z_n_raw)
        z_all = tf.math.l2_normalize(tf.concat([z_a, z_p, z_n], axis=0), axis=1)
        M = tf.shape(z_all)[0]
        sim = tf.matmul(z_all, z_all, transpose_b=True) / temperature

        B = tf.shape(z_a)[0]
        labels = tf.concat([tf.ones(2*B, tf.int32), tf.zeros(B, tf.int32)], axis=0)
        pos = tf.cast(tf.equal(labels[:,None], labels[None,:]), tf.float32) - tf.eye(M, dtype=tf.float32)

        logits = sim - 1e9 * tf.eye(M)
        log_prob = logits - tf.reduce_logsumexp(logits, axis=1, keepdims=True)

        pos_cnt = tf.reduce_sum(pos, axis=1)
        loss_vec = -tf.reduce_sum(pos * log_prob, axis=1) / (pos_cnt + 1e-8)
        return tf.reduce_mean(loss_vec)

    # --- Circle loss (pooled, cosine sim, all negatives per anchor) ---
    def circle_loss(z_a_bt, z_p_bt, z_n_bt, m=0.25, gamma=32.0):
        """
        z_*_bt: [B,T,D] -> pooled to [B,D]
        Standard Circle loss (Sun et al., CVPR 2020):
          L_a = softplus( logsumexp(gamma * alpha_n * (s_an - m)) + gamma * alpha_p * (s_ap - (1-m)) )
          (and symmetric p-branch), averaged over batch and branches.
        """
        za = tf.math.l2_normalize(_mean_pool_bt(z_a_bt), axis=1)  # [B,D]
        zp = tf.math.l2_normalize(_mean_pool_bt(z_p_bt), axis=1)  # [B,D]
        zn = tf.math.l2_normalize(_mean_pool_bt(z_n_bt), axis=1)  # [B,D]

        # Similarities
        s_ap = tf.reduce_sum(za * zp, axis=1)                     # [B]
        s_an = tf.matmul(za, zn, transpose_b=True)                # [B,B]
        s_pn = tf.matmul(zp, zn, transpose_b=True)                # [B,B]

        # Circle weights
        delta_p = 1.0 - m
        # weights: hard pos/neg get larger weights
        alpha_p = tf.nn.relu(delta_p - s_ap)                  # [B]
        alpha_n_a = tf.nn.relu(s_an - m)                      # [B,B]
        alpha_n_p = tf.nn.relu(s_pn - m)                      # [B,B]

        # anchor branch
        neg_a = tf.reduce_logsumexp(gamma * alpha_n_a * (s_an - m), axis=1)   # [B]
        pos_a = gamma * alpha_p * (s_ap - delta_p)                            # [B]
        L_a = tf.nn.softplus(neg_a + pos_a)

        # positive-as-anchor (symmetry)
        neg_p = tf.reduce_logsumexp(gamma * alpha_n_p * (s_pn - m), axis=1)   # [B]
        L_p = tf.nn.softplus(neg_p + pos_a)  # pos term reuses s_ap

        return tf.reduce_mean(0.5 * (L_a + L_p))

    def focal_logits_weighted(y_bt, logit_bt, w_bt, alpha=0.25, gamma=2.0):
        # y_bt, logit_bt, w_bt: [B, T]
        focal = tf.keras.losses.BinaryFocalCrossentropy(
            alpha=alpha, gamma=gamma, apply_class_balancing=True, from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE, axis=-1
        )
        # Add a singleton channel so Keras doesn't try to squeeze weird dims
        y = tf.cast(y_bt, tf.float32)[..., None]      # [B, T, 1]
        z = tf.cast(logit_bt, tf.float32)[..., None]  # [B, T, 1]
        # Per-element loss -> [B, T]
        loss_bt = focal(y, z)
        
        w = tf.cast(w_bt, tf.float32)                 # [B, T]
        # Safe masked mean
        num = tf.reduce_sum(loss_bt * w)
        den = tf.reduce_sum(w) + tf.keras.backend.epsilon()
        return num / den

    @tf.function
    def loss_fn(y_true, y_pred):
        # ---- split triplet on batch axis ----
        a_out, p_out, n_out = tf.split(y_pred, 3, axis=0)
        a_true, p_true, n_true = tf.split(y_true, 3, axis=0)

        # ---- head indices from model ----
        logit_idx = getattr(model, '_cls_logit_index', 0)
        prob_idx  = getattr(model, '_cls_prob_index',  1)
        emb_start = getattr(model, '_emb_start_index', 2)

        # ---- slice logits / embeddings ----
        a_logit = a_out[..., logit_idx]      # [B,T]
        p_logit = p_out[..., logit_idx]      # [B,T]
        n_logit = n_out[..., logit_idx]      # [B,T]
        a_prob = a_out[..., prob_idx]        # [B,T]
        p_prob = p_out[..., prob_idx]        # [B,T]
        n_prob = n_out[..., prob_idx]        # [B,T]
        a_emb   = a_out[..., emb_start:]     # [B,T,D]
        p_emb   = p_out[..., emb_start:]     # [B,T,D]
        n_emb   = n_out[..., emb_start:]     # [B,T,D]

        # ---- labels ----
        a_lab = tf.cast(a_true[..., 0], tf.float32)  # [B,T] (0/1)
        p_lab = tf.cast(p_true[..., 0], tf.float32)  # [B,T]
        n_lab = tf.cast(n_true[..., 0], tf.float32)  # [B,T] (all zeros by design)

        # ---- ramps ----
        it = tf.cast(model.optimizer.iterations, tf.float32)
        total_steps = float(params.get('TOTAL_STEPS', 100000))

        # ramp_delay = float(params.get('RAMP_DELAY', 0.01 * total_steps))
        # ramp_steps = float(params.get('RAMP_STEPS', 0.25 * total_steps))
        # r = _ramp(it, ramp_delay, ramp_steps)                       # 0→1

        # Circle weight (use LOSS_Circle if present; else fall back to your old TupMPN weight)
        w_circle_tgt = tf.cast(params.get('LOSS_Circle',
                                   params.get('LOSS_TupMPN', 30.0)), tf.float32)
        w_supcon = tf.cast(params.get('LOSS_SupCon', 1.0), tf.float32)
        
        # Negatives ramp (controls FP/min pressure)
        neg_min    = float(params.get('LOSS_NEGATIVES_MIN', 4.0))
        neg_target = float(params.get('LOSS_NEGATIVES', 20.0))
        neg_delay  = float(params.get('NEG_RAMP_DELAY', 0.05 * total_steps))
        neg_steps  = float(params.get('NEG_RAMP_STEPS', 0.45 * total_steps))
        r_neg = _ramp(it, neg_delay, max(1.0, neg_steps))
        loss_fp_weight = tf.cast(neg_min + r_neg * max(0.0, neg_target - neg_min), tf.float32)

        # ---- Circle metric loss ----
        m     = float(params.get('CIRCLE_m', 0.25))
        gamma = float(params.get('CIRCLE_gamma', 32.0))
        L_circle_raw = circle_loss(a_emb, p_emb, n_emb, m=m, gamma=gamma)

        # L_sup_raw = supcon_ripple(a_emb, p_emb, n_emb, temperature=params.get('SUPCON_T', 0.1))
        metric_loss = w_circle_tgt * L_circle_raw# + w_supcon * L_sup_raw


        # ---- Classification BCE with post-onset masking ----
        a_mask = post_onset_mask(a_lab)                 # [B,T]
        p_mask = post_onset_mask(p_lab)                 # [B,T]
        n_mask = tf.ones_like(n_lab, tf.float32)        # keep all negatives

        alpha_pos = float(params.get('BCE_POS_ALPHA', 2.0))

        # cls_a = bce_logits_weighted(a_lab, a_logit, a_mask)
        # cls_p = bce_logits_weighted(p_lab, p_logit, p_mask)
        # cls_n = bce_logits_weighted(n_lab, n_logit, n_mask)


        alpha = float(params.get('FOCAL_ALPHA', 0.25))
        gamma = float(params.get('FOCAL_GAMMA', 2.00))
        
        # print("Using Focal Loss with alpha =", alpha, "gamma =", gamma)
        cls_a = focal_logits_weighted(a_lab, a_logit, a_mask,  alpha=alpha, gamma=gamma)
        cls_p = focal_logits_weighted(p_lab, p_logit, p_mask,  alpha=alpha, gamma=gamma)
        cls_n = focal_logits_weighted(n_lab, n_logit, n_mask,  alpha=alpha, gamma=gamma)

        classification_loss = cls_a + alpha_pos * cls_p + loss_fp_weight * cls_n

        # ---- TV smoothing on logits ----
        lam_tv = tf.cast(params.get('LOSS_TV', 0.02), tf.float32)

        # tv_term = lam_tv * (
        #     tv_trunc_logp_masked(a_prob, n_mask, tau=4.0) +         # A: smooth after onset
        #     tv_trunc_logp_masked(p_prob, n_mask, tau=4.0) +         # P: same
        #     tv_trunc_logp_masked(n_prob, n_mask, tau=4.0) # N: suppress upward drift
        # )
        tv_term = lam_tv * (
                    tv_on_logits_masked(a_logit, n_mask) +  # Smooth logits A
                    tv_on_logits_masked(p_logit, n_mask) +  # Smooth logits P
                    tv_on_logits_masked(n_logit, n_mask)    # Smooth logits N
                )        

        classification_loss = classification_loss + tv_term

        # ---- magnitude balancing (same idea as before) ----
        ratio = tf.stop_gradient(
            tf.reduce_mean(metric_loss) / (tf.reduce_mean(classification_loss) + 1e-6)
        )
        ratio = tf.clip_by_value(ratio, 0.1, 10.0)
        
        total = tf.reduce_mean(metric_loss) + 0.5 * ratio * tf.reduce_mean(classification_loss)
        return total

    return loss_fn

def mixed_hybrid_loss_final(horizon=0, loss_weight=1, params=None, model=None, this_op=None):
    import math
    import tensorflow as tf

    # ==========================
    # 1. HELPERS
    # ==========================
    def _mean_pool_bt(x):               
        # Flatten time: [B,T,D] -> [B,D]
        return tf.reduce_mean(tf.cast(x, tf.float32), axis=1)

    def _ramp(step, delay, dur):
        step  = tf.cast(step, tf.float32)
        delay = tf.cast(delay, tf.float32)
        dur   = tf.maximum(tf.cast(dur, tf.float32), 1.0)
        x = tf.clip_by_value((step - delay) / dur, 0.0, 1.0)
        return 0.5 - 0.5 * tf.cos(tf.constant(math.pi, tf.float32) * x)

    def tv_on_logits(z):  # z: [B,T]
        return tf.reduce_mean(tf.abs(z[:, 1:] - z[:, :-1]))

    # ==========================
    # 2. METRIC LOSS A: TupMPN (Symmetric)
    # ==========================
    def mpn_tuple_loss(z_a_raw, z_p_raw, z_n_raw, *, margin_hard=1.0, margin_weak=0.1, lambda_pull=0.1):
        # Pool to [B, D]
        z_a = tf.reduce_mean(tf.cast(z_a_raw, tf.float32), axis=1)
        z_p = tf.reduce_mean(tf.cast(z_p_raw, tf.float32), axis=1)
        z_n = tf.reduce_mean(tf.cast(z_n_raw, tf.float32), axis=1)
        
        # A. Pull (A <-> P) - The Identity Constraint
        d_pair = tf.reduce_sum(tf.square(z_a - z_p), axis=1) + 1e-8
        L_pull = lambda_pull * d_pair

        # B. Weak Push (A <-> Other P) - The Cluster Constraint
        # "Stay closer to your match than to other ripples, but allows clustering."
        d_ap = tf.reduce_sum(tf.square(z_a[:,None,:] - z_p[None,:,:]), axis=2) + 1e-8
        mask = tf.eye(tf.shape(z_a)[0], dtype=tf.bool)
        d_ap_masked = tf.where(mask, tf.fill(tf.shape(d_ap), 1e9), d_ap)
        
        L_weak = tf.reduce_mean(tf.nn.relu(margin_weak + d_pair[:,None] - d_ap_masked))
        
        # C. Hard Push (A <-> N) AND (P <-> N) - The Repulsion Constraint
        d_an = tf.reduce_sum(tf.square(z_a[:,None,:] - z_n[None,:,:]), axis=2) + 1e-8
        d_pn = tf.reduce_sum(tf.square(z_p[:,None,:] - z_n[None,:,:]), axis=2) + 1e-8 # Symmetric
        
        # LogSumExp (Lifted)
        lifted_an = tf.reduce_logsumexp(margin_hard - d_an, axis=1)
        lifted_pn = tf.reduce_logsumexp(margin_hard - d_pn, axis=1)
        
        L_hard = 0.5 * (tf.nn.relu(d_pair + lifted_an) + tf.nn.relu(d_pair + lifted_pn))

        return tf.reduce_mean(L_pull + L_weak + L_hard)

    # ==========================
    # 3. METRIC LOSS B: Circle Loss (The Fix)
    # ==========================
    def circle_loss_fixed(z_a_bt, z_p_bt, z_n_bt, m=0.25, gamma=64.0):
        # Normalize
        za = tf.math.l2_normalize(_mean_pool_bt(z_a_bt), axis=1)
        zp = tf.math.l2_normalize(_mean_pool_bt(z_p_bt), axis=1)
        zn = tf.math.l2_normalize(_mean_pool_bt(z_n_bt), axis=1)

        # Similarities
        s_ap = tf.reduce_sum(za * zp, axis=1)       
        s_an = tf.matmul(za, zn, transpose_b=True)  
        s_pn = tf.matmul(zp, zn, transpose_b=True)  

        # --- THE CRITICAL FIX (s + m) ---
        # Guarantees gradients at initialization (s=0 -> alpha=m)
        alpha_p   = tf.nn.relu((1.0 + m) - tf.stop_gradient(s_ap)) 
        alpha_n_a = tf.nn.relu(tf.stop_gradient(s_an) + m) 
        alpha_n_p = tf.nn.relu(tf.stop_gradient(s_pn) + m) 

        # Margins & Logits
        Delta_p = 1.0 - m
        Delta_n = m

        neg_a = tf.reduce_logsumexp(gamma * alpha_n_a * (s_an - Delta_n), axis=1)
        neg_p = tf.reduce_logsumexp(gamma * alpha_n_p * (s_pn - Delta_n), axis=1)
        pos_a = gamma * alpha_p * (s_ap - Delta_p)

        # Softplus( Neg - Pos )
        L_a = tf.nn.softplus(neg_a - pos_a)
        L_p = tf.nn.softplus(neg_p - pos_a)

        return tf.reduce_mean(0.5 * (L_a + L_p))

    @tf.function
    def loss_fn(y_true, y_pred):
        # Split & Slice
        a_out, p_out, n_out = tf.split(y_pred, 3, axis=0)
        a_true, p_true, n_true = tf.split(y_true, 3, axis=0)

        logit_idx = getattr(model, '_cls_logit_index', 0)
        emb_start = getattr(model, '_emb_start_index', 2)

        a_logit = a_out[..., logit_idx]; p_logit = p_out[..., logit_idx]; n_logit = n_out[..., logit_idx]
        a_emb   = a_out[..., emb_start:]; p_emb   = p_out[..., emb_start:]; n_emb   = n_out[..., emb_start:]
        
        a_lab = tf.cast(a_true[..., 0], tf.float32)
        p_lab = tf.cast(p_true[..., 0], tf.float32)
        n_lab = tf.cast(n_true[..., 0], tf.float32)

        # Ramps
        it = tf.cast(model.optimizer.iterations, tf.float32)
        total_steps = float(params.get('TOTAL_STEPS', 100000))
        r = _ramp(it, params.get('RAMP_DELAY', 0.01*total_steps), params.get('RAMP_STEPS', 0.25*total_steps))

        neg_min = params.setdefault('LOSS_NEGATIVES_MIN', 4.0)
        r_neg   = _ramp(it, params.setdefault('NEG_RAMP_DELAY', 0.05*total_steps), params.setdefault('NEG_RAMP_STEPS', 0.45*total_steps))
        loss_fp_weight = neg_min + r_neg * tf.maximum(0.0, params.get('LOSS_NEGATIVES', 24.0) - neg_min)

        # --- 1. METRIC LOSS (Backbone Training) ---
        L_mpn = mpn_tuple_loss(a_emb, p_emb, n_emb, 
                               margin_weak=float(params.get('MARGIN_WEAK', 0.1))) # Low margin for clusters
        
        L_circ = circle_loss_fixed(a_emb, p_emb, n_emb, 
                                   m=float(params.get('CIRCLE_m', 0.25)), 
                                   gamma=float(params.get('CIRCLE_gamma', 64.0)))
        
        w_tupMPN = r * float(params.get('LOSS_TupMPN', 80.0)) # High weight
        w_circle = r * float(params.get('LOSS_Circle', 40.0))
        metric_loss = w_tupMPN * L_mpn + w_circle * L_circ

        # --- 2. CLASSIFICATION LOSS (Head Training) ---
        smoothing = float(params.get('LABEL_SMOOTHING', 0.05)) # Jitter protection
        
        def bce_logits_smooth(y, z):
            if smoothing > 0:
                y = y * (1.0 - smoothing) + 0.5 * smoothing
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z))

        cls_a = bce_logits_smooth(a_lab, a_logit)
        cls_p = bce_logits_smooth(p_lab, p_logit)
        cls_n = bce_logits_smooth(n_lab, n_logit)

        alpha_pos = float(params.get('BCE_POS_ALPHA', 2.0))
        
        classification_loss = cls_a + alpha_pos * cls_p + loss_fp_weight * cls_n

        # TV Smoothness
        lam_tv = float(params.get('LOSS_TV', 0.15))
        classification_loss += lam_tv * (tv_on_logits(a_logit) + tv_on_logits(p_logit) + tv_on_logits(n_logit))

        # --- 3. FINAL AGGREGATION ---
        ratio = tf.stop_gradient(
            tf.reduce_mean(metric_loss) / (tf.reduce_mean(classification_loss) + 1e-6)
        )
        ratio = tf.clip_by_value(ratio, 0.1, 10.0)

        return tf.reduce_mean(metric_loss) + 0.5 * ratio * tf.reduce_mean(classification_loss)

    return loss_fn