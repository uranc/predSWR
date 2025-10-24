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
    # logit or sigmoid
    flag_sigmoid = 'SigmoidFoc' in params['TYPE_LOSS']
    print('Using Sigmoid:', flag_sigmoid)
    # # LR schedure 
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
        epochs = 100#int(params.get('NO_EPOCHS', 100))
        total_steps = max(1, steps_per_epoch * epochs)

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
                gate_bias=(-3.0 if params['mode']=="train" else -2.0),  # tighter in pretrain
                l1_gate=1e-5,                                   # mild sparsity on gate logits (reduces FP)
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
            feats = tcn_op(inputs_nets)  # [B, T, C_tcn]
            feats = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='slice_last_T')(feats)
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
        h = tf.stop_gradient(feats) if 'StopGrad' in params['TYPE_ARCH'] else feats
        h = LayerNormalization()(h)
        h = Conv1D(64, 
                    1, 
                    kernel_initializer='glorot_uniform', 
                    kernel_regularizer=tf.keras.regularizers.L1(1e-4), 
                    name="cls_pw1")(h)
        h = Activation('gelu')(h)
        cls_in = Dropout(0.1)(h)
        # cls_in = h
        cls_logits = Conv1D(1, 1,
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=tf.keras.regularizers.L1(1e-4),
                        bias_initializer='zeros',
                        activation=None, name='cls_logits')(cls_in)
        cls_prob = Activation('sigmoid', name='cls_prob')(cls_logits)

        # ---- projection head for contrastive (L2N ONLY here) ----
        emb = LayerNormalization(name='emb_ln')(feats)
        emb = Dense(n_filters * 2, activation='gelu',
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
 
        # f1_metric = MaxF1MetricHorizon(model=model)
        # # r1_metric = RobustF1Metric(model=model)
        # # latency_metric = LatencyMetric(model=model)
        # event_f1_metric = EventAwareF1(model=model)
        # fp_event_metric = EventFalsePositiveRateMetric(model=model)
        # metrics = [
        #     EventPRAUC(mode="consec", consec_k=5, model=model),
        #     MaxF1MetricHorizon(model=model),
        #     # LatencyWeightedF1Metric(tau=16, mode="consec", consec_k=3, model=model),
        #     FPperMinMetric(thresh=0.5, win_sec=64/2500, mode="consec", consec_k=3, model=model),
        # ]       
        # metrics = [
        #     SamplePRAUC(model=model),             # sample-level PR-AUC
        #     MaxF1MetricHorizon(model=model),      # your sample-level Max-F1
        #     FPperMinMetric(thresh=0.5, win_sec=64/2500,
        #                 mode="consec", consec_k=3, model=model),  # keep this for realism
        # ] 
        metrics = [
            SamplePRAUC(model=model),                  # timepoint PR-AUC
            SampleMaxMCC(model=model),                 # timepoint max MCC
            SampleMaxF1(model=model),                  # timepoint max F1
            LatencyScore(thresholds=tf.linspace(0.5, 0.99, 6), min_run = 5, tau=16.0, model=model),    
            FPperMinMetric(
                thresh=0.3,
                win_sec=params['NO_STRIDES']/params['SRATE'],  # <- stride/fs (NOT window)
                mode="consec", consec_k=3,
                model=model
            ),
        ]        
        # metrics = [
        #     EventPRAUC(consec_k=5, model=model),          # consecutive rule
        #     LatencyWeightedF1Metric(tau=16,                # τ in samples
        #                             consec_k=5,
        #                             model=model),
        #     FPperMinMetric(thresh=0.5,           # your operating τ
        #                 win_sec=64/2500,
        #                 consec_k=3,           # or majority_ratio=0.5
        #                 model=model),            
        # ]
        # Create loss function and compile model
        # loss_fn = triplet_loss(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model)
        loss_fn = mixed_latent_loss(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model)

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
        # metrics = [
        #     EventPRAUC(model=model),
        #     LatencyWeightedF1Metric(tau=32, model=model),
        #     TPRatFixedFPMetric(1., params['NO_TIMEPOINTS']/params['SRATE'],  # adjust to your sampling rate
        #                     model=model),
        # ]
        # metrics = [
        #     EventPRAUC(mode="consec", consec_k=5, model=model),
        #     MaxF1MetricHorizon(model=model),
        #     # LatencyWeightedF1Metric(tau=16, mode="consec", consec_k=3, model=model),
        #     FPperMinMetric(thresh=0.5, win_sec=64/2500, mode="consec", consec_k=3, model=model),
        # ]  
        metrics = [
                    EventPRAUC(mode="consec", consec_k=5, model=model),
                    MaxF1MetricHorizon(model=model),
                    # LatencyWeightedF1Metric(tau=16, mode="consec", consec_k=3, model=model),
                    FPperMinMetric(thresh=0.5, win_sec=64/2500, mode="consec", consec_k=3, model=model),
                ]              
        # Create loss function and compile model
        # loss_fn = triplet_loss(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model)
        loss_fn = mixed_latent_loss(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model)

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
            all_outputs = Lambda(lambda tt: tt[:, -1:, 1], name='Last_Output')(anchor_output)

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
        model.load_weights(params['WEIGHT_FILE'])

    return model

def build_DBI_Patch(params):
    """Builds model with concatenated outputs based on actual PatchAD returns."""

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
        initial_lr = params['LEARNING_RATE']
        this_optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_lr, weight_decay=1e-4, clipnorm=1.0)
        # Optional schedule (else keep constant LR)
        if params.get("USE_LR_SCHEDULE", False):
            sched = WarmStableCool(
                base_lr=params["LEARNING_RATE"],
                warmup_steps=int(0.02 * T_steps),
                cool_start=int(0.80 * T_steps),
                total_steps=T_steps,
                final_scale=0.1
            )
            this_optimizer.learning_rate = sched
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

    # Input layer
    input_layer = Input(shape=(params['seq_length'], params['input_channels']))

    # Create PatchAD model
    patch_ad = PatchAD(
        input_channels=params['input_channels'],
        seq_length=params['seq_length'],
        patch_sizes=params['patch_sizes'],
        d_model=params['NO_FILTERS'],
        e_layer=params['NO_KERNELS'],
        dropout=params.get('dropout', 0.1),
        activation=params.get('activation', 'elu'), # Ensure this matches MLPBlock's activ_name if needed
        norm=params.get('patch_ad_norm', 'ln')
    )

    # Get all PatchAD outputs
    num_dists, size_dists, num_mx, size_mx, reconstruction = patch_ad(input_layer)

    # Get target sequence length from reconstruction
    target_seq_length = tf.shape(reconstruction)[1]
    d_model = params['NO_FILTERS']
    batch_size = tf.shape(input_layer)[0]

    def upsample_and_combine_repeat(tensor_list, target_len_tensor, name_prefix):
        """Upsamples using tf.repeat (inter-patch style) and concatenates."""
        upsampled_tensors = []
        if not tensor_list: # Handle empty list case
            return None
        for i, tensor in enumerate(tensor_list):
            B_ = tf.shape(tensor)[0]
            current_len = tf.shape(tensor)[1]
            D_ = tf.shape(tensor)[2]

            repeats = tf.cast(tf.math.ceil(tf.cast(target_len_tensor, tf.float32) / tf.cast(current_len, tf.float32)), tf.int32)
            repeated = tf.repeat(tensor, repeats=repeats, axis=1)

            sliced_tensor = tf.slice(repeated, [0, 0, 0], tf.stack([B_, target_len_tensor, D_]))
            upsampled_tensors.append(sliced_tensor)

        if not upsampled_tensors: # Should not happen if tensor_list was not empty initially
             return None
        # Concatenate along the feature dimension
        return Concatenate(axis=2, name=f'{name_prefix}_concat')(upsampled_tensors)

    def upsample_and_combine_tile(tensor_list, target_len_tensor, name_prefix):
        """Upsamples using tf.tile (intra-patch style) and concatenates."""
        upsampled_tensors = []
        if not tensor_list: # Handle empty list case
            return None
        for i, tensor in enumerate(tensor_list):
            B_ = tf.shape(tensor)[0]
            current_len = tf.shape(tensor)[1]
            D_ = tf.shape(tensor)[2]

            repeats_factor = tf.cast(tf.math.ceil(tf.cast(target_len_tensor, tf.float32) / tf.cast(current_len, tf.float32)), tf.int32)

            reshaped = tf.reshape(tensor, tf.stack([B_, current_len, 1, D_]))
            multiples = tf.stack([1, 1, repeats_factor, 1]) # Multiples for tf.tile
            tiled = tf.tile(reshaped, multiples)

            # Reshape back: current_len * repeats_factor can be a tensor
            # Ensure the shape for reshape uses tensor multiplication
            tiled_flat = tf.reshape(tiled, tf.stack([B_, current_len * repeats_factor, D_]))

            sliced_tensor = tf.slice(tiled_flat, [0, 0, 0], [B_, target_len_tensor, D_])
            upsampled_tensors.append(sliced_tensor)

        if not upsampled_tensors: # Should not happen if tensor_list was not empty initially
            return None
        # Concatenate along the feature dimension
        return Concatenate(axis=2, name=f'{name_prefix}_concat')(upsampled_tensors)


    # pdb.set_trace()
    # Upsample and combine features from different scales using repeat/tile
    combined_num_dists = upsample_and_combine_repeat(num_dists, target_seq_length, 'num_dists')
    combined_size_dists = upsample_and_combine_tile(size_dists, target_seq_length, 'size_dists')
    combined_num_mx = upsample_and_combine_repeat(num_mx, target_seq_length, 'num_mx')
    combined_size_mx = upsample_and_combine_tile(size_mx, target_seq_length, 'size_mx')

    # # Create upsampling layers
    # upsample_repeat = UpsampleAndCombineRepeat(name_prefix='num_dists')
    # upsample_tile = UpsampleAndCombineTile(name_prefix='size_dists')

    # # Apply upsampling
    # combined_num_dists = upsample_repeat([num_dists, target_seq_length])
    # combined_size_dists = upsample_tile([size_dists, target_seq_length])
    # combined_num_mx = upsample_repeat([num_mx, target_seq_length])
    # combined_size_mx = upsample_tile([size_mx, target_seq_length])

    # pdb.set_trace()
    # --- Classification Head ---
    # Combine features relevant for classification (e.g., distributions)
    cls_features = tf.concat([
        combined_num_dists,
        combined_size_dists,
        combined_num_mx, # Optional
        combined_size_mx
    ], axis=-1, name='cls_features_concat') # Shape [B, L, D*2 or D*4]

    # Apply L2 Normalization if specified
    if params.get('TYPE_ARCH', '').find('L2N') > -1:
        cls_features = Lambda(lambda t: tf.math.l2_normalize(t, axis=-1), name='l2_norm_cls')(cls_features)

    # Final classification layer
    classification_output = Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(),
                                  use_bias=True,
                                  activation='sigmoid', # Use sigmoid for probability
                                  name='classification_output')(cls_features) # Shape [B, L, 1]

    # --- Final Model Output ---
    if params['mode'] == 'predict':
        final_output = tf.concat([reconstruction, classification_output], axis=-1)
        final_output = Lambda(lambda tt: tt[:, -1:, :], name='Last_Output')(final_output) # [B, 1, C+1]
    else:
        final_output = tf.concat([
            reconstruction,                    # [B, L, C]
            classification_output,            # [B, L, 1]
            combined_num_dists,              # [B, L, D*(Patch*Levels)]
            combined_size_dists,             # [B, L, D*(Patch*Levels)]
            combined_num_mx,                 # [B, L, D*(Patch*Levels)]
            combined_size_mx                 # [B, L, D*(Patch*Levels)]
        ], axis=-1)

    # Create the Model
    model = Model(inputs=input_layer, outputs=final_output)

    # Set attribute for metrics (indicates reconstruction is present)
    model._is_classification_only = False
    hori_shift = 0 # Set horizon shift for metrics
    loss_weight = params.get('LOSS_WEIGHT', 1.0) # Default to 1.0 if not specified
    loss_fn = custom_fbfce(horizon=hori_shift, loss_weight=loss_weight, params=params)
    # loss_fn = combined_mse_fbfce_loss(params)
    # Compile the model
    # Ensure `combined_mse_fbfce_loss` is adapted for this output structure
    model.compile(
        optimizer=tf.keras.optimizers.Adam(params.get('LEARNING_RATE', 1e-3)),
        # loss=combined_mse_fbfce_loss(params), # Use the updated loss function below
        loss = loss_fn,
        metrics=[MaxF1MetricHorizon(model=model),
                 EventAwareF1(model=model),
                 EventFalsePositiveRateMetric(model=model)]
    )

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
    def __init__(self, thresholds=tf.linspace(0., 1., 51),
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
    def __init__(self, thresholds=tf.linspace(0., 1., 101),
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
        return tf.reduce_max(mcc)

    def reset_state(self):
        for v in (self.tp, self.fp, self.tn, self.fn):
            v.assign(tf.zeros_like(v))


class SampleMaxF1(tf.keras.metrics.Metric):
    def __init__(self, thresholds=tf.linspace(0., 1., 101), name="sample_max_f1", model=None, **kw):
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
        return tf.reduce_max(f1)

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
    def __init__(self, thresholds=tf.linspace(0.05, 0.95, 19),
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


def custom_fbfce(loss_weight=1, horizon=0, params=None, model=None, this_op=None):
    flag_sigmoid = 'SigmoidFoc' in params['TYPE_LOSS']
    is_classification_only = 'Only' in  params['TYPE_ARCH']
    is_cad = 'CAD' in params['TYPE_ARCH']
    is_patch = 'Patch' in params['TYPE_ARCH']
    """
    Custom loss function that combines focal binary cross-entropy (FBFCE) with additional terms.
    Args:
    - y_true: Ground truth labels.
    - y_pred: Predicted labels.
    - loss_weight: Weight for the additional loss terms.
    - horizon: Horizon for prediction.
    - params: Dictionary of parameters.
    - model: Keras model.
    - this_op: Optional additional operation.
    Returns:
    - Combined loss value.
    """
    @tf.function
    def loss_fn(y_true, y_pred, loss_weight=loss_weight, horizon=horizon, flag_sigmoid=flag_sigmoid, is_classification_only=is_classification_only, is_cad=is_cad):
        # Get the last dimension size and determine mode
        # print(y_true.shape)
        # print(y_pred.shape)
        # pdb.set_trace()

        def extract_param_from_loss_type(loss_type, param_prefix, default_value):
            """Helper function to extract parameters from loss type string"""
            try:
                if param_prefix in loss_type:
                    idx = loss_type.find(param_prefix) + len(param_prefix)
                    param_str = loss_type[idx:idx+3]
                    return float(param_str) / 100.0
                return default_value
            except:
                return default_value

        def compute_classification_loss(y_true_exp, y_pred_exp, sample_weight, params):
            loss_type = params['TYPE_LOSS']

            # Initialize the loss variable
            classification_loss = 0.0

            # FocalSmooth loss
            if 'FocalSmooth' in loss_type:
                # Extract alpha and gamma parameters
                alpha = extract_param_from_loss_type(loss_type, 'Ax', 0.65)
                gamma = extract_param_from_loss_type(loss_type, 'Gx', 2.0)

                focal_loss = tf.keras.losses.BinaryFocalCrossentropy(
                    apply_class_balancing=True,
                    alpha=alpha,
                    gamma=gamma,
                    label_smoothing=0.1,
                    reduction=tf.keras.losses.Reduction.NONE
                )
                # pdb.set_trace()
                loss_values = focal_loss(y_true_exp, y_pred_exp)
                classification_loss = tf.reduce_mean(loss_values * sample_weight)

            # SigmoidFoc loss
            elif 'SigmoidFoc' in loss_type:
                # Extract alpha and gamma parameters
                alpha = extract_param_from_loss_type(loss_type, 'Ax', 0.65)
                gamma = extract_param_from_loss_type(loss_type, 'Gx', 2.0)

                focal_loss = tf.keras.losses.BinaryFocalCrossentropy(
                    apply_class_balancing=True,
                    alpha=alpha,
                    gamma=gamma,
                    from_logits=flag_sigmoid,
                    reduction=tf.keras.losses.Reduction.NONE
                )
                loss_values = focal_loss(y_true_exp, y_pred_exp)
                classification_loss = tf.reduce_mean(loss_values * sample_weight)

            # Focal loss
            elif 'Focal' in loss_type:
                # Extract alpha and gamma parameters
                alpha = extract_param_from_loss_type(loss_type, 'Ax', 0.65)
                gamma = extract_param_from_loss_type(loss_type, 'Gx', 2.0)

                # print(f"Using Focal loss with alpha={alpha}, gamma={gamma}")

                use_class_balancing = alpha > 0
                focal_loss = tf.keras.losses.BinaryFocalCrossentropy(
                    apply_class_balancing=use_class_balancing,
                    alpha=alpha if alpha > 0 else None,
                    gamma=gamma,
                    reduction=tf.keras.losses.Reduction.NONE
                )
                # pdb.set_trace()
                loss_values = focal_loss(y_true_exp, y_pred_exp)
                # pdb.set_trace()
                classification_loss = tf.reduce_mean(loss_values)# * sample_weight)
                # classification_loss = tf.reduce_mean(loss_values * sample_weight)

            # Default to binary cross-entropy
            else:
                bce_loss = tf.keras.losses.BinaryCrossentropy(
                    reduction=tf.keras.losses.Reduction.NONE
                )
                loss_values = bce_loss(y_true_exp, y_pred_exp)
                classification_loss = tf.reduce_mean(loss_values * sample_weight)

            return classification_loss

        def add_extra_losses(base_loss, y_true_exp, y_pred_exp, sample_weight, params, model,
                            prediction_targets, prediction_out, this_op):
            # Start with the base loss
            total_loss = base_loss
            loss_type = params['TYPE_LOSS']

            # Add TV Loss if specified (Total Variation - encourages smoothness)
            if 'TV' in loss_type:
                tv_factor = 1e-5
                tv_loss = tv_factor * tf.reduce_sum(
                    tf.image.total_variation(tf.expand_dims(y_pred_exp, axis=-1))
                )
                total_loss += tv_loss

            # Add L2 smoothness loss - penalizes large changes between consecutive predictions
            if 'L2' in loss_type:
                l2_factor = 1e-5
                l2_smooth_loss = l2_factor * tf.reduce_mean(
                    tf.square(y_pred_exp[:, 1:] - y_pred_exp[:, :-1])
                )
                total_loss += l2_smooth_loss

            # Add margin loss - encourages confident predictions
            if 'Margin' in loss_type:
                margin_factor = 1e-4
                margin_loss = margin_factor * tf.reduce_mean(
                    tf.squeeze(y_pred_exp * (1 - y_pred_exp))
                )
                total_loss += margin_loss

            # Add entropy loss - encourages confident predictions for binary labels
            if 'Entropy' in loss_type and 'HYPER_ENTROPY' in params:
                entropy_factor = params['HYPER_ENTROPY']

                # Calculate prediction entropy: -p*log(p) - (1-p)*log(1-p)
                # Add small epsilon to avoid log(0)
                epsilon = 1e-8
                y_pred_squeezed = tf.squeeze(y_pred_exp, axis=-1)
                y_true_squeezed = tf.squeeze(y_true_exp, axis=-1)

                # Calculate binary entropy of predictions
                entropy = -(y_pred_squeezed * tf.math.log(y_pred_squeezed + epsilon) +
                        (1 - y_pred_squeezed) * tf.math.log(1 - y_pred_squeezed + epsilon))

                # For binary labels (0/1), we want to minimize entropy for all predictions
                # Higher penalty for incorrect low-confidence predictions, lower penalty for correct confident predictions
                confidence_weight = tf.abs(y_true_squeezed - y_pred_squeezed) + 0.1  # Higher weight for incorrect predictions

                # Weight entropy by confidence - penalize uncertainty more for incorrect predictions
                weighted_entropy = entropy * confidence_weight

                entropy_loss = tf.reduce_mean(weighted_entropy)
                total_loss += entropy_factor * entropy_loss

            # add false positive loss
            if 'FalsePositive' in loss_type and 'HYPER_FALSEPOSITIVE' in params:
                false_positive_factor = params['HYPER_FALSEPOSITIVE']
                # False positive loss implementation
                # FP = tf.reduce_sum(tf.cast((y_pred > threshold) & (y_true < 0.5), tf.float32))
                # FP_loss = FP / (tf.reduce_sum(1 - y_true) + epsilon)
                # FP_loss = tf.reduce_mean(tf.square(y_pred * (1 - y_true))) * lambda_fp

                total_loss += false_positive_factor * false_positive_loss

            # Add truncated MSE loss
            if 'TMSE' in loss_type and 'HYPER_TMSE' in params:
                tmse_factor = params['HYPER_TMSE']
                # Truncated MSE loss implementation
                tmse_loss = truncated_mse_loss(y_true_exp, y_pred_exp)
                total_loss += tmse_factor * tmse_loss

            # Early onset preference loss - encourages early detection
            if 'Early' in loss_type:
                early_factor = 0.1
                if 'HYPER_EARLY' in params:
                    early_factor = params['HYPER_EARLY']

                def early_onset_loss():
                    threshold = 0.5
                    early_onset_threshold = 5
                    penalty_factor = 0.1
                    reward_factor = 0.05

                    # Reshape for compatibility if needed
                    y_true_sq = tf.squeeze(y_true_exp, axis=-1)
                    y_pred_sq = tf.squeeze(y_pred_exp, axis=-1)

                    # Calculate delay in detection - find first time predictions cross threshold
                    detected_times = tf.argmax(tf.cast(y_pred_sq >= threshold, tf.int32), axis=1)
                    true_event_times = tf.argmax(tf.cast(y_true_sq, tf.int32), axis=1)
                    delay = detected_times - true_event_times

                    # Penalty for late detections
                    late_penalty = tf.where(delay > early_onset_threshold,
                                            penalty_factor * tf.cast(delay - early_onset_threshold, tf.float32),
                                            0.0)

                    # Reward for slightly early detections
                    early_reward = tf.where((delay < 0) & (delay >= -early_onset_threshold),
                                        reward_factor * tf.cast(-delay, tf.float32),
                                        0.0)

                    # Combine penalty and reward terms
                    onset_loss = tf.reduce_mean(late_penalty - early_reward)
                    return onset_loss

                # Only calculate early onset loss if there are actual events in the batch
                has_events = tf.reduce_any(y_true_exp > 0.5)
                early_loss = tf.cond(
                    has_events,
                    early_onset_loss,
                    lambda: 0.0
                )
                total_loss += early_factor * early_loss

            # Monotonicity loss - encourages predictions to rise toward events
            if 'Mono' in loss_type and 'HYPER_MONO' in params:
                mono_factor = params['HYPER_MONO']

                def monotonicity_loss():
                    """Loss to encourage early rising predictions before onset"""
                    batch_size = tf.shape(y_true_exp)[0]
                    time_steps = tf.shape(y_true_exp)[1]
                    gap_margin = 40  # Default max time steps before onset to apply monotonicity

                    # Find onset indices (first 0 to 1 transition)
                    y_true_flat = tf.squeeze(y_true_exp, axis=-1)
                    shifted_y_true = tf.concat([tf.zeros((batch_size, 1)), y_true_flat[:, :-1]], axis=1)
                    onset_mask = tf.logical_and(shifted_y_true < 0.5, y_true_flat >= 0.5)
                    onset_indices = tf.argmax(tf.cast(onset_mask, tf.int32), axis=1)

                    # Create gap mask relative to onset indices
                    gap_start = tf.maximum(onset_indices - gap_margin, 0)
                    gap_end = onset_indices
                    gap_mask = tf.sequence_mask(gap_end, maxlen=time_steps, dtype=tf.float32) - \
                            tf.sequence_mask(gap_start, maxlen=time_steps, dtype=tf.float32)
                    gap_mask = tf.expand_dims(gap_mask, axis=-1)

                    # Differences between consecutive predictions
                    y_pred_flat = tf.squeeze(y_pred_exp, axis=-1)
                    diff = y_pred_flat[:, :-1] - y_pred_flat[:, 1:]
                    diff = tf.pad(diff, [[0, 0], [0, 1]])  # Pad to match original shape

                    # Penalize non-monotonic increases in the gap period
                    monotonicity_penalty = tf.nn.relu(diff) * gap_mask
                    return tf.reduce_mean(monotonicity_penalty)

                # Only calculate monotonicity loss if there are actual events in the batch
                has_events = tf.reduce_any(y_true_exp > 0.5)
                mono_loss = tf.cond(
                    has_events,
                    monotonicity_loss,
                    lambda: 0.0
                )

                total_loss += mono_factor * mono_loss

            # Add Barlow Twins loss for representation learning
            if 'Bar' in loss_type and 'HYPER_BARLOW' in params:
                barlow_factor = params['HYPER_BARLOW']

                def barlow_twins_loss(embedding1, embedding2, lambda_param=0.005):
                    # Normalize embeddings
                    embedding1 = tf.nn.l2_normalize(embedding1, axis=1)
                    embedding2 = tf.nn.l2_normalize(embedding2, axis=1)

                    # Cross-correlation matrix
                    c = tf.matmul(embedding1, embedding2, transpose_a=True)
                    c /= tf.cast(tf.shape(embedding1)[0], tf.float32)  # Normalize by batch size

                    # On-diagonal: minimize difference from 1
                    on_diag = tf.reduce_sum(tf.square(tf.linalg.diag_part(c) - 1))

                    # Off-diagonal: minimize correlation
                    off_diag = tf.reduce_sum(tf.square(c)) - on_diag

                    # Total loss
                    return on_diag + lambda_param * off_diag

                if this_op is not None:
                    # Extract intermediate tensor representations
                    # This is a simplified implementation that would need adjustment
                    total_loss += barlow_factor * barlow_twins_loss(this_op(prediction_out), this_op(prediction_targets))
            return total_loss

        def prediction_mode_branch():
            if horizon == 0:
                prediction_targets = y_true[:, :, :8]  # LFP targets (8 channels)
                prediction_out = y_pred[:, :, :8]     # LFP predictions (8 channels)
            else:
                prediction_targets = y_true[:, horizon:, :8]  # LFP targets (8 channels)
                prediction_out = y_pred[:, :-horizon, :8]     # LFP predictions (8 channels)
            y_true_exp = tf.expand_dims(y_true[:, :, 8], axis=-1)  # Probability at index 8
            y_pred_exp = tf.expand_dims(y_pred[:, :, 8], axis=-1)  # Predicted probability at index 8
            return prediction_targets, prediction_out, y_true_exp, y_pred_exp

        def patch_mode_branch():
            # Assumes d_model and proj_dim are correctly retrieved from params or scope
            # Example: d_model = params.get('D_MODEL', 32) # Make sure D_MODEL is in params
            # Example: proj_dim = params.get('PROJ_DIM', d_model) # Often same as d_model
            input_channels = params['input_channels']
            d_model = params['NO_FILTERS']
            n_patch = params['NO_DILATIONS']
            n_layer = params['NO_KERNELS']
            proj_dim = d_model # Assuming proj_dim is d_model

            # LFP Reconstruction part
            if horizon == 0:
                prediction_targets = y_true[:, :, :8]  # LFP targets (8 channels)
                prediction_out = y_pred[:, :, :8]     # LFP predictions (8 channels)
            else:
                prediction_targets = y_true[:, horizon:, :8]
                prediction_out = y_pred[:, :-horizon, :8]

            # Classification part
            y_true_exp = tf.expand_dims(y_true[:, :, 8], axis=-1)  # True probability at index 8
            y_pred_exp = tf.expand_dims(y_pred[:, :, 8], axis=-1)  # Predicted probability at index 8

            current_idx = 9  # Start index for additional features
            # Split tensor into n_layer*n_patch tensors of d_model dimension
            num_dists_full = y_pred[:, :, current_idx:current_idx + d_model*n_layer*n_patch]
            num_dists = []
            for i in range(n_layer*n_patch):
                start_idx = i * d_model
                end_idx = (i + 1) * d_model
                num_dists.append(num_dists_full[:, :, start_idx:end_idx])
            current_idx += d_model*n_layer*n_patch

            size_dists_full = y_pred[:, :, current_idx:current_idx + d_model*n_layer*n_patch]
            size_dists = []
            for i in range(n_layer*n_patch):
                start_idx = i * d_model
                end_idx = (i + 1) * d_model
                size_dists.append(size_dists_full[:, :, start_idx:end_idx])
            current_idx += d_model*n_layer*n_patch

            num_mx_full = y_pred[:, :, current_idx:current_idx + d_model*n_layer*n_patch]
            num_mx = []
            for i in range(n_layer*n_patch):
                start_idx = i * d_model
                end_idx = (i + 1) * d_model
                num_mx.append(num_mx_full[:, :, start_idx:end_idx])
            current_idx += d_model*n_layer*n_patch

            size_mx_full = y_pred[:, :, current_idx:current_idx + d_model*n_layer*n_patch]
            size_mx = []
            for i in range(n_layer*n_patch):
                start_idx = i * d_model
                end_idx = (i + 1) * d_model
                size_mx.append(size_mx_full[:, :, start_idx:end_idx])
            return prediction_targets, prediction_out, y_true_exp, y_pred_exp, num_dists, size_dists, num_mx, size_mx

        def classification_mode_branch():
            # For classification-only mode (1 or 2 channels)
            if tf.shape(y_true)[-1] > 1:
                y_true_exp = tf.expand_dims(y_true[:, :, 0], axis=-1)
                y_pred_exp = tf.expand_dims(y_pred[:, :, 0], axis=-1)
            else:
                # y_true_exp = tf.expand_dims(y_true, axis=-1)
                # y_pred_exp = tf.expand_dims(y_pred, axis=-1)
                y_true_exp = y_true#tf.expand_dims(y_true, axis=-1)
                y_pred_exp = y_pred#tf.expand_dims(y_pred, axis=-1)

            # Create dummy tensors for prediction targets and outputs
            dummy_shape = tf.concat([tf.shape(y_true)[:2], [8]], axis=0)
            dummy_tensor = tf.zeros(dummy_shape)
            return dummy_tensor, dummy_tensor, y_true_exp, y_pred_exp

        # print('Using classification only:', is_classification_only)
        if is_patch:
            prediction_targets, prediction_out, y_true_exp, y_pred_exp, num_dists, size_dists, num_mx, size_mx = patch_mode_branch()
            sample_weight =  tf.ones_like(y_true[:, :, 0])
        elif not is_classification_only:
            prediction_targets, prediction_out, y_true_exp, y_pred_exp = prediction_mode_branch()
            sample_weight = tf.cond(
                tf.greater(tf.shape(y_true)[-1], 9),
                lambda: y_true[:, :, 9],
                lambda: tf.ones_like(y_true[:, :, 0])
            )
        else:
            prediction_targets, prediction_out, y_true_exp, y_pred_exp = classification_mode_branch()
            sample_weight = tf.cond(
                tf.greater(tf.shape(y_true)[-1], 1),
                lambda: y_true[:, :, 1],
                lambda: tf.ones_like(y_true[:, :, 0])
            )

            # if is_cad:
            #     print('Using CAD mode and downsampling labels')
            #     pool = tf.keras.layers.MaxPooling1D(pool_size=12, strides=12, padding='valid')
            #     sample_weight_expanded = tf.expand_dims(sample_weight, axis=-1)
            #     sample_weight = pool(sample_weight_expanded)
            #     # y_true_exp_expanded = tf.expand_dims(y_true_exp, axis=-1)
            #     y_true_exp = pool(y_true_exp)
        # pdb.set_trace()

        # # Maybe zero weights during training (20% chance)
        # if params.get('TRAINING', False):  # Only if in training mode
        #     random_val = tf.random.uniform(())
        #     sample_weight = tf.cond(
        #         tf.less(random_val, 0.2),
        #         lambda: tf.zeros_like(sample_weight),
        #         lambda: sample_weight
        #     )

        # Calculate classification loss
        total_loss = compute_classification_loss(y_true_exp, y_pred_exp, sample_weight, params)

        # # Handle sigmoid activation if needed
        # if flag_sigmoid:
        #     y_pred_exp = tf.math.sigmoid(y_pred_exp)

        # Add extra loss terms
        # total_loss = add_extra_losses(total_loss, y_true_exp, y_pred_exp, sample_weight,
        #                              params, model, prediction_targets, prediction_out, this_op)

        # if is_patch:
        #     from model.patchAD.loss import tf_anomaly_score
        #     # total_loss += loss_weight*patch_loss(prediction_targets, prediction_out, proj_inter, proj_intra, x_inter, x_intra)


        #     loss_cont = tf.reduce_mean(tf_anomaly_score( # Renamed variable
        #         num_dists, size_dists,
        #         params['seq_length'],
        #         True,
        #         params.get('patch_ad_temp', 1.0)
        #     ))

        #     # Calculate patch loss
        #     Np_proj = tf.reduce_mean(tf_anomaly_score( # Renamed variable
        #         num_mx, size_dists,
        #         params['seq_length'],
        #         True,
        #         params.get('patch_ad_temp', 1.0)
        #     ))

        #     Pp_proj = tf.reduce_mean(tf_anomaly_score( # Renamed variable
        #         num_dists, size_mx,
        #         params['seq_length'],
        #         True,
        #         params.get('patch_ad_temp', 1.0)
        #     ))

        #     loss_proj = Np_proj + Pp_proj
        #     # Ensure patch_loss is a scalar tensor
        #     proj_weight = params.get('patch_ad_proj', 0.2)
        #     patch_loss = (1-proj_weight)*tf.reduce_mean(loss_cont)+proj_weight*tf.reduce_mean(loss_proj)
        #     weight_patch = tf.cast(params.get('WEIGHT_Patch', 1.0), dtype=tf.float32)
        #     total_loss += weight_patch * patch_loss
        # elif not is_classification_only:
        if not is_classification_only:
            mse_loss = tf.reduce_mean(tf.square(prediction_targets-prediction_out))
            total_loss += loss_weight * mse_loss

        return total_loss
    return loss_fn


def triplet_loss(horizon=0, loss_weight=1, params=None, model=None, this_op=None):
    """
    Custom triplet loss function for SWR prediction that works with tuple outputs.
    """
    def extract_param_from_loss_type(loss_type, param_prefix, default_value):
        """Helper function to extract parameters from loss type string"""
        try:
            if param_prefix in loss_type:
                idx = loss_type.find(param_prefix) + len(param_prefix)
                param_str = loss_type[idx:idx+3]
                return float(param_str) / 100.0
            return default_value
        except:
            return default_value
    
    @tf.function
    def loss_fn(y_true, y_pred):
        # Get batch size
        # pdb.set_trace()
        # batch_size = tf.shape(y_pred)[0] // 3
        # params['BATCH_SIZE'] = tf.shape(y_pred)[0] // 3

        # Split tensors using tf.split instead of unpacking
        anchor_out, positive_out, negative_out = tf.split(y_pred, num_or_size_splits=3, axis=0)
        anchor_true, positive_true, negative_true = tf.split(y_true, num_or_size_splits=3, axis=0)

        # Split predictions into classification and embedding parts using tf operations
        anchor_class = anchor_out[..., 0]
        positive_class = positive_out[..., 0]
        negative_class = negative_out[..., 0]

        anchor_emb = tf.reduce_mean(anchor_out[..., 1:], axis=1)
        positive_emb = tf.reduce_mean(positive_out[..., 1:], axis=1)
        negative_emb = tf.reduce_mean(negative_out[..., 1:], axis=1)

        # Extract labels using tf operations
        anchor_labels = tf.cast(anchor_true[..., 0], tf.float32)
        positive_labels = tf.cast(positive_true[..., 0], tf.float32)
        negative_labels = tf.cast(negative_true[..., 0], tf.float32)

        # Calculate embedding distances using tf operations
        pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor_emb - negative_emb), axis=-1)

        # Triplet loss with margin
        margin = tf.cast(params.get('TRIPLET_MARGIN', 1.0), tf.float32)
        triplet_loss_val = tf.maximum(0.0, pos_dist - neg_dist + margin)

        # Classification loss using focal loss

        loss_type = params['TYPE_LOSS']
        alpha = 0.5#extract_param_from_loss_type(loss_type, 'Ax', 0.65)
        gamma = 2#extract_param_from_loss_type(loss_type, 'Gx', 2.0)

        focal_loss = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=alpha,
            gamma=gamma,
            reduction=tf.keras.losses.Reduction.NONE
        )

        # Calculate classification loss for each part of the triplet
        class_loss_anchor = focal_loss(anchor_labels, anchor_class)
        class_loss_positive = focal_loss(positive_labels, positive_class)
        class_loss_negative = focal_loss(negative_labels, negative_class)

        # Combine losses using tf operations
        loss_fp_weight = params.get('LOSS_NEGATIVES', 2.0)
        print(f"Loss weight: {loss_weight}, FP weight: {loss_fp_weight}")
        total_class_loss = (class_loss_anchor + class_loss_positive + loss_fp_weight * class_loss_negative) / 3.0


        # Weight between triplet and classification loss
        total_loss = loss_weight*tf.reduce_mean(triplet_loss_val) + tf.reduce_mean(total_class_loss)

        if ('Entropy' in params['TYPE_LOSS']) and ('HYPER_ENTROPY' in params):
            entropy_factor = params['HYPER_ENTROPY']

            # Calculate prediction entropy: -p*log(p) - (1-p)*log(1-p)
            # Add small epsilon to avoid log(0)
            epsilon = 1e-8

            for y_pred_exp, y_true_exp in zip([anchor_class, positive_class, negative_class],
                                           [anchor_labels, positive_labels, negative_labels]):

                # y_pred_squeezed = tf.squeeze(y_pred_exp, axis=-1)
                # y_true_squeezed = tf.squeeze(y_true_exp, axis=-1)
                y_pred_squeezed = y_pred_exp
                y_true_squeezed = y_true_exp

                # Calculate binary entropy of predictions
                entropy = -(y_pred_squeezed * tf.math.log(y_pred_squeezed + epsilon) +
                        (1 - y_pred_squeezed) * tf.math.log(1 - y_pred_squeezed + epsilon))

                # For binary labels (0/1), we want to minimize entropy for all predictions
                # Higher penalty for incorrect low-confidence predictions, lower penalty for correct confident predictions
                confidence_weight = tf.abs(y_true_squeezed - y_pred_squeezed) + 0.1  # Higher weight for incorrect predictions

                # Weight entropy by confidence - penalize uncertainty more for incorrect predictions
                weighted_entropy = entropy * confidence_weight

                entropy_loss = tf.reduce_mean(weighted_entropy)
                total_loss += entropy_factor * entropy_loss
        return total_loss
    return loss_fn

# def mixed_latent_loss(horizon=0, loss_weight=1, params=None, model=None, this_op=None):
#     # ----- helpers -----
#     def _mean_pool(x): return tf.reduce_mean(x, axis=1)

#     # cosine ramp in [0,1]
#     def _ramp(step, delay, dur):
#         step  = tf.cast(step, tf.float32)
#         delay = tf.cast(delay, tf.float32)
#         dur   = tf.maximum(tf.cast(dur, tf.float32), 1.0)
#         x = tf.clip_by_value((step - delay) / dur, 0.0, 1.0)
#         return 0.5 - 0.5 * tf.cos(tf.constant(math.pi, tf.float32) * x)  # 0→1

#     def tv_on_logits(z):  # z: [B,T]
#         return tf.reduce_mean(tf.abs(z[:, 1:] - z[:, :-1]))

#     # ----- metric losses (unchanged math) -----
#     def mpn_tuple_loss(z_a_raw, z_p_raw, z_n_raw, *, margin_hard=1.0, margin_weak=0.1, lambda_pull=0.1, exclude_self=True):
#         z_a = tf.reduce_mean(tf.cast(z_a_raw, tf.float32), axis=1)
#         z_p = tf.reduce_mean(tf.cast(z_p_raw, tf.float32), axis=1)
#         z_n = tf.reduce_mean(tf.cast(z_n_raw, tf.float32), axis=1)
#         B   = tf.shape(z_a)[0]

#         # pull
#         d_pair = tf.reduce_sum(tf.square(z_a - z_p), axis=1) + 1e-8
#         L_pull = lambda_pull * d_pair

#         def _mask_self(mat):
#             if exclude_self:
#                 mask = tf.eye(B, dtype=tf.bool)
#                 return tf.where(mask, tf.fill(tf.shape(mat), tf.constant(1e9, tf.float32)), mat)
#             return mat

#         d_ap = tf.reduce_sum(tf.square(z_a[:,None,:] - z_p[None,:,:]), axis=2) + 1e-8
#         d_ap = _mask_self(d_ap)
#         L_weak = tf.reduce_mean(tf.nn.relu(margin_weak + d_pair[:,None] - d_ap))
#         d_an = tf.reduce_sum(tf.square(z_a[:,None,:] - z_n[None,:,:]), axis=2) + 1e-8
#         d_an = _mask_self(d_an)
#         lifted = tf.reduce_logsumexp(margin_hard - d_an, axis=1)
#         L_hard = tf.nn.relu(d_pair + lifted)

#         return tf.reduce_mean(L_pull + L_weak + L_hard)

#     def supcon_ripple(z_a_raw, z_p_raw, z_n_raw, *, temperature=0.1):
#         z_a = _mean_pool(z_a_raw); z_p = _mean_pool(z_p_raw); z_n = _mean_pool(z_n_raw)
#         z_all = tf.math.l2_normalize(tf.concat([z_a, z_p, z_n], axis=0), axis=1)
#         M = tf.shape(z_all)[0]
#         sim = tf.matmul(z_all, z_all, transpose_b=True) / temperature

#         B = tf.shape(z_a)[0]
#         labels = tf.concat([tf.ones(2*B, tf.int32), tf.zeros(B, tf.int32)], axis=0)
#         pos = tf.cast(tf.equal(labels[:,None], labels[None,:]), tf.float32) - tf.eye(M, dtype=tf.float32)

#         logits = sim - 1e9 * tf.eye(M)
#         log_prob = logits - tf.reduce_logsumexp(logits, axis=1, keepdims=True)

#         pos_cnt = tf.reduce_sum(pos, axis=1)
#         loss_vec = -tf.reduce_sum(pos * log_prob, axis=1) / (pos_cnt + 1e-8)
#         return tf.reduce_mean(loss_vec)

#     @tf.function
#     def loss_fn(y_true, y_pred):
#         a_out, p_out, n_out = tf.split(y_pred, 3, axis=0)
#         a_true, p_true, n_true = tf.split(y_true, 3, axis=0)

#         logit_idx = getattr(model, '_cls_logit_index', 0)
#         prob_idx  = getattr(model, '_cls_prob_index',  1)
#         emb_start = getattr(model, '_emb_start_index', 2)

#         a_logit = a_out[..., logit_idx];  p_logit = p_out[..., logit_idx];  n_logit = n_out[..., logit_idx]
#         a_prob  = a_out[..., prob_idx];   p_prob  = p_out[..., prob_idx];   n_prob  = n_out[..., prob_idx]
#         a_emb   = a_out[..., emb_start:]; p_emb   = p_out[..., emb_start:]; n_emb   = n_out[..., emb_start:]

#         a_lab = tf.cast(a_true[..., 0], tf.float32)
#         p_lab = tf.cast(p_true[..., 0], tf.float32)
#         n_lab = tf.cast(n_true[..., 0], tf.float32)

#         # --------- ramps (READ-ONLY) ----------
#         it = tf.cast(model.optimizer.iterations, tf.float32)

#         total_steps = 100*1000
#         # Metric ramps
#         ramp_delay = tf.cast(params.get('RAMP_DELAY', int(0.01*total_steps)), tf.float32)
#         ramp_steps = tf.cast(params.get('RAMP_STEPS', int(0.25*total_steps)), tf.float32)

#         w_sup_tgt = tf.cast(params.get('LOSS_SupCon', 1.0), tf.float32)
#         w_mpn_tgt = tf.cast(params.get('LOSS_TupMPN', 1.0), tf.float32)
#         w_neg_tgt = tf.cast(params.get('LOSS_NEGATIVES', 2.0), tf.float32)
        
#         r = _ramp(it, ramp_delay, ramp_steps)  # 0→1 after delay
#         w_supcon = r * w_sup_tgt
#         w_tupMPN = r * w_mpn_tgt

#         # Negatives ramp (MIN → target)
#         # Negatives ramp: gentle 25–40% of run
#         neg_min = params.setdefault('LOSS_NEGATIVES_MIN', 4.0)  # softer start to reduce FP early
#         neg_delay = params.setdefault('NEG_RAMP_DELAY', int(0.05* total_steps))
#         neg_steps = params.setdefault('NEG_RAMP_STEPS', int(0.45 * total_steps))
#         neg_steps = tf.cast(neg_steps if neg_steps is not None else params.get('RAMP_STEPS', 1), tf.float32)
#         neg_steps = tf.maximum(neg_steps, 1.0)

#         r_neg = _ramp(it, neg_delay, neg_steps)
#         loss_fp_weight = neg_min + r_neg * tf.maximum(0.0, w_neg_tgt - neg_min)

#         # --------- metric learning ----------
#         L_mpn_raw = mpn_tuple_loss(a_emb, p_emb, n_emb, margin_hard=1.0, margin_weak=0.1, lambda_pull=0.1)
#         L_sup_raw = supcon_ripple(a_emb, p_emb, n_emb, temperature=0.1)
#         metric_loss = w_tupMPN * L_mpn_raw + w_supcon * L_sup_raw

#         # --------- BCE on logits ----------
#         def bce_logits(y, z):
#             return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z))

#         cls_a = bce_logits(a_lab, a_logit)
#         cls_p = bce_logits(p_lab, p_logit)
#         cls_n = bce_logits(n_lab, n_logit)

#         alpha_pos = tf.cast(params.get('BCE_POS_ALPHA', 2.0), tf.float32)
#         classification_loss = cls_a + alpha_pos * cls_p + loss_fp_weight * cls_n

#         # --------- logit-TV smoothing ----------
#         lam_tv = tf.cast(params.get('LOSS_TV', 0.02), tf.float32)
#         print(f"TV weight: {lam_tv}")
#         tv_term = tv_on_logits(a_logit) + tv_on_logits(p_logit) + tv_on_logits(n_logit)
#         classification_loss = classification_loss + lam_tv * tv_term

#         # --------- re-scaling (ratio trick) ----------
#         ratio = tf.stop_gradient(
#             (tf.reduce_mean(L_mpn_raw + L_sup_raw)) /
#             (tf.reduce_mean(classification_loss) + 1e-6)
#         )
#         ratio = tf.clip_by_value(ratio, 0.1, 10.0)

#         total = tf.reduce_mean(metric_loss) + 0.5 * ratio * tf.reduce_mean(classification_loss)

#         # Optional entropy term
#         if ('Entropy' in params.get('TYPE_LOSS','')) and ('HYPER_ENTROPY' in params):
#             eps = tf.constant(1e-8, tf.float32)
#             ent_w = tf.cast(params['HYPER_ENTROPY'], tf.float32)
#             y_prob_all = tf.concat([a_prob, p_prob, n_prob], axis=0)
#             y_true_all = tf.concat([a_lab,  p_lab,  n_lab],  axis=0)
#             entropy = -y_prob_all*tf.math.log(y_prob_all+eps) -(1. - y_prob_all)*tf.math.log(1.-y_prob_all+eps)
#             conf = tf.abs(y_true_all - y_prob_all) + 0.1
#             total += ent_w * tf.reduce_mean(entropy * conf)
#         return total
#     return loss_fn

def mixed_latent_loss(horizon=0, loss_weight=1, params=None, model=None, this_op=None):
    import math
    import tensorflow as tf

    # ---------- helpers ----------
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
        ap = tf.nn.relu(m + s_ap)                                 # [B]
        an = tf.nn.relu(s_an + m)                                 # [B,B]
        pn = tf.nn.relu(s_pn + m)                                 # [B,B]
        delta_p = 1.0 - m

        # Anchor branch
        neg_part_a = tf.reduce_logsumexp(gamma * an * (s_an - m), axis=1)      # [B]
        pos_part_a = gamma * ap * (s_ap - delta_p)                              # [B]
        L_a = tf.nn.softplus(neg_part_a + pos_part_a)                           # [B]

        # Positive-as-anchor branch (symmetry)
        neg_part_p = tf.reduce_logsumexp(gamma * pn * (s_pn - m), axis=1)      # [B]
        pos_part_p = gamma * ap * (s_ap - delta_p)                              # [B] (same s_ap)
        L_p = tf.nn.softplus(neg_part_p + pos_part_p)

        return tf.reduce_mean(0.5 * (L_a + L_p))

    @tf.function
    def loss_fn(y_true, y_pred):
        # ---- split triplet on batch axis ----
        a_out, p_out, n_out = tf.split(y_pred, 3, axis=0)
        a_true, p_true, n_true = tf.split(y_true, 3, axis=0)

        # ---- head indices from model ----
        logit_idx = getattr(model, '_cls_logit_index', 0)
        emb_start = getattr(model, '_emb_start_index', 2)

        # ---- slice logits / embeddings ----
        a_logit = a_out[..., logit_idx]      # [B,T]
        p_logit = p_out[..., logit_idx]      # [B,T]
        n_logit = n_out[..., logit_idx]      # [B,T]
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

        ramp_delay = float(params.get('RAMP_DELAY', 0.01 * total_steps))
        ramp_steps = float(params.get('RAMP_STEPS', 0.25 * total_steps))
        r = _ramp(it, ramp_delay, ramp_steps)                       # 0→1

        # Circle weight (use LOSS_Circle if present; else fall back to your old TupMPN weight)
        w_circle_tgt = tf.cast(params.get('LOSS_Circle',
                                   params.get('LOSS_TupMPN', 30.0)), tf.float32)
        w_circle = r * w_circle_tgt

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
        metric_loss  = w_circle * L_circle_raw

        # ---- Classification BCE with post-onset masking ----
        a_mask = post_onset_mask(a_lab)                 # [B,T]
        p_mask = post_onset_mask(p_lab)                 # [B,T]
        n_mask = tf.ones_like(n_lab, tf.float32)        # keep all negatives

        alpha_pos = tf.cast(params.get('BCE_POS_ALPHA', 2.0), tf.float32)

        cls_a = bce_logits_weighted(a_lab, a_logit, a_mask)
        cls_p = bce_logits_weighted(p_lab, p_logit, p_mask)
        cls_n = bce_logits_weighted(n_lab, n_logit, n_mask)

        classification_loss = cls_a + alpha_pos * cls_p + loss_fp_weight * cls_n

        # ---- TV smoothing on logits ----
        lam_tv = tf.cast(params.get('LOSS_TV', 0.02), tf.float32)
        tv_term = tv_on_logits(a_logit) + tv_on_logits(p_logit) + tv_on_logits(n_logit)
        classification_loss = classification_loss + lam_tv * tv_term

        # ---- magnitude balancing (same idea as before) ----
        ratio = tf.stop_gradient(
            tf.reduce_mean(metric_loss) / (tf.reduce_mean(classification_loss) + 1e-6)
        )
        ratio = tf.clip_by_value(ratio, 0.1, 10.0)

        total = tf.reduce_mean(metric_loss) + 0.5 * ratio * tf.reduce_mean(classification_loss)
        return total

    return loss_fn


def combined_mse_fbfce_loss(params):
    def loss(y_true, y_pred):
        # Get dimensions from params
        input_channels = params['input_channels']
        d_model = params['NO_FILTERS']
        n_patch = params['NO_DILATIONS']
        n_layer = params['NO_KERNELS']

        # Slice tensors
        current_idx = 0
        reconstruction = y_pred[:, :, current_idx:current_idx + input_channels]
        current_idx += input_channels

        classification_output = y_pred[:, :, current_idx:current_idx + 1]
        current_idx += 1

        # Split tensor into n_layer*n_patch tensors of d_model dimension
        num_dists_full = y_pred[:, :, current_idx:current_idx + d_model*n_layer*n_patch]
        num_dists = []
        for i in range(n_layer*n_patch):
            start_idx = i * d_model
            end_idx = (i + 1) * d_model
            num_dists.append(num_dists_full[:, :, start_idx:end_idx])
        current_idx += d_model*n_layer*n_patch

        size_dists_full = y_pred[:, :, current_idx:current_idx + d_model*n_layer*n_patch]
        size_dists = []
        for i in range(n_layer*n_patch):
            start_idx = i * d_model
            end_idx = (i + 1) * d_model
            size_dists.append(size_dists_full[:, :, start_idx:end_idx])
        current_idx += d_model*n_layer*n_patch

        num_mx_full = y_pred[:, :, current_idx:current_idx + d_model*n_layer*n_patch]
        num_mx = []
        for i in range(n_layer*n_patch):
            start_idx = i * d_model
            end_idx = (i + 1) * d_model
            num_mx.append(num_mx_full[:, :, start_idx:end_idx])
        current_idx += d_model*n_layer*n_patch

        size_mx_full = y_pred[:, :, current_idx:current_idx + d_model*n_layer*n_patch]
        size_mx = []
        for i in range(n_layer*n_patch):
            start_idx = i * d_model
            end_idx = (i + 1) * d_model
            size_mx.append(size_mx_full[:, :, start_idx:end_idx])

        # Slice y_true
        true_signal = y_true[:, :, :input_channels]
        true_labels = y_true[:, :, input_channels:input_channels + 1]

        # Calculate reconstruction loss
        recon_loss = tf.reduce_mean(tf.square(true_signal - reconstruction))

        loss_cont = tf.reduce_mean(tf_anomaly_score( # Renamed variable
            num_dists, size_dists,
            params['seq_length'],
            True,
            params.get('patch_ad_temp', 1.0)
        ))

        # Calculate patch loss
        Np_proj = tf.reduce_mean(tf_anomaly_score( # Renamed variable
            num_mx, size_dists,
            params['seq_length'],
            True,
            params.get('patch_ad_temp', 1.0)
        ))

        Pp_proj = tf.reduce_mean(tf_anomaly_score( # Renamed variable
            num_dists, size_mx,
            params['seq_length'],
            True,
            params.get('patch_ad_temp', 1.0)
        ))

        loss_proj = Np_proj + Pp_proj
        # Ensure patch_loss is a scalar tensor
        proj_weight = params.get('patch_ad_proj', 0.2)
        patch_loss = (1-proj_weight)*tf.reduce_mean(loss_cont)+proj_weight*tf.reduce_mean(loss_proj)

        # Extract alpha and gamma parameters
        alpha = params['focal_alpha'] if 'focal_alpha' in params else 0.25
        gamma = params['focal_gamma'] if 'focal_gamma' in params else 3.50

        use_class_balancing = alpha > 0
        focal_loss = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=use_class_balancing,
            alpha=alpha if alpha > 0 else None,
            gamma=gamma,
            reduction=tf.keras.losses.Reduction.NONE
        )
        loss_values = focal_loss(true_labels, classification_output)
        cls_loss = tf.reduce_mean(loss_values)


        # Convert weights to tensors and ensure correct dtype
        weight_recon = tf.cast(params.get('WEIGHT_Recon', 1.0), dtype=tf.float32)
        weight_patch = tf.cast(params.get('WEIGHT_Patch', 1.0), dtype=tf.float32)
        weight_class = tf.cast(params.get('WEIGHT_Class', 1.0), dtype=tf.float32)

        # Calculate total loss using tensor operations
        total_loss = patch_loss#cls_loss#recon_loss#cls_loss#weight_class * cls_loss#weight_recon * recon_loss
        #+ weight_class * cls_loss#* recon_loss + weight_patch * patch_loss + weight_class * cls_loss
        return total_loss

    return loss