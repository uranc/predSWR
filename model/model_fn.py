import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, ELU, Input, LSTM, Dense, Dropout, Layer, MultiHeadAttention, LayerNormalization, Normalization
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, Concatenate, Lambda
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
from tensorflow_addons.layers import WeightNormalization
from tensorflow_addons.activations import gelu
# from tensorflow_addons.losses import SigmoidFocalCrossEntropy, LiftedStructLoss, TripletSemiHardLoss, TripletHardLoss, ContrastiveLoss
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import applications
from model.patchAD.patch_ad import PatchAD
from model.patchAD.loss import PatchADLoss
import pdb

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
        # pdb.set_trace()

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


def build_DBI_TCN_TripletOnly(input_timepoints, input_chans=8, params=None):
    # logit or sigmoid
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

    # Function to create the TCN backbone for processing LFP signals
    def create_tcn_model():
        signal_input = Input(shape=(None, input_chans))

        # Apply normalization if needed
        if params['TYPE_ARCH'].find('ZNorm')>-1:
            print('Using ZNorm')
            normalized = Lambda(lambda x: (x - tf.reduce_mean(x, axis=1, keepdims=True)) /
                            (tf.math.reduce_std(x, axis=1, keepdims=True) + 1e-6))(signal_input)
        else:
            normalized = signal_input

        # Apply CSD if needed
        if params['TYPE_ARCH'].find('CSD')>-1:
            csd_inputs = CSDLayer()(normalized)
            inputs_nets = Concatenate(axis=-1)([normalized, csd_inputs])
        else:
            inputs_nets = normalized

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
                ch_slice = Lambda(lambda x: x[:, :, c:c+1])(signal_input)
                single_outputs.append(tcn_op(ch_slice))
            nets = Concatenate(axis=-1)(single_outputs)
            nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)

        # Apply L2 normalization if needed
        if params['TYPE_ARCH'].find('L2N')>-1:
            tcn_output = Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(nets)
        else:
            tcn_output = nets

        if params['TYPE_ARCH'].find('StopGrad')>-1:
            print('Using Stop Gradient for Class. Branch')
            # Stop gradient flow from classification branch to TCN
            frozen_features = tf.stop_gradient(tcn_output)
            # Classification branch uses frozen features
            classification_output = Conv1D(1, kernel_size=1, kernel_initializer='glorot_uniform',
                                        use_bias=True, activation='sigmoid',
                                        name='tmp_class')(frozen_features)
        else:
            classification_output = Conv1D(1, kernel_size=1, kernel_initializer='glorot_uniform', use_bias=True, activation='sigmoid', name='tmp_class')(tcn_output)

        # linear projection for the triplet loss
        tmp_pred = Dense(32, activation=this_activation, use_bias=True, name='tmp_pred')(tcn_output)  # Output future values
        triplet_output = Dense(32, activation=None, use_bias=True, name='triplet_output')(tmp_pred)

        concatenate_output = Concatenate(axis=-1)([classification_output, triplet_output])
        return Model(inputs=signal_input, outputs=concatenate_output)

    # Create the shared TCN model for triplet learning
    tcn_backbone = create_tcn_model()

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

        f1_metric = MaxF1MetricHorizon(model=model)
        # r1_metric = RobustF1Metric(model=model)
        # latency_metric = LatencyMetric(model=model)
        event_f1_metric = EventAwareF1(model=model)
        fp_event_metric = EventFalsePositiveRateMetric(model=model)

        # Create loss function and compile model
        loss_fn = triplet_loss(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model)

        model.compile(
            optimizer=this_optimizer,
            loss=loss_fn,
            metrics=[f1_metric, event_f1_metric, fp_event_metric]
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
            all_outputs = Lambda(lambda tt: tt[:, -1:, 0], name='Last_Output')(anchor_output)

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

def build_model_PatchAD(input_timepoints, input_chans=8, patch_sizes=[2, 4, 8, 16],
                             d_model=256, num_layers=3, classifier_hidden_dim=64, params=None):
    """
    Builds a Pure PatchAD-based model.

    Architecture:
    Input -> Preprocessing -> Patch Extraction -> PatchAD Module ->
    Aggregation of patch-based representations -> Classifier Head (MLP)

    Returns a model with a single classification output.
    """
    inputs = Input(shape=(input_timepoints, input_chans), name='inputs')
    normalized = Lambda(lambda x: (x - tf.reduce_mean(x, axis=1, keepdims=True)) /
                        (tf.math.reduce_std(x, axis=1, keepdims=True) + 1e-6),
                        name='norm')(inputs)

    # Lists to collect outputs from different scales.
    proj_inter_list = []
    proj_intra_list = []
    rec_inter_list = []
    rec_intra_list = []
    inter_list = []   # Raw encoder outputs (inter view)
    intra_list = []   # Raw encoder outputs (intra view)

    patchad_module = PatchAD(
        8, input_timepoints, [2, 4, 8, 16, 32],
        d_model=d_model,
        num_layers=num_layers
    )
    outputs = patchad_module(normalized, training=True)


    import pdb
    pdb.set_trace()
    # get classification loss
    concat_feats = tf.concat((outputs['x_inter'], outputs['x_intra']), axis=-1)
    classification_output = Dense(classifier_hidden_dim, activation='sigmoid')(concat_feats)

    # Get horizon shift if specified
    is_classification_only = True
    hori_shift = 0
    if params and 'TYPE_ARCH' in params and params['TYPE_ARCH'].find('PatchAD') > -1:
        hori_shift = int(int(params['TYPE_ARCH'][params['TYPE_ARCH'].find('PatchAD')+4:params['TYPE_ARCH'].find('PatchAD')+6])/1000*params['SRATE'])
        print('Using Horizon Timesteps:', hori_shift)
        is_classification_only = False

    loss_weight = 1.0
    if params and 'LOSS_WEIGHT' in params:
        loss_weight = params['LOSS_WEIGHT']

    # Handle prediction mode
    if params and 'mode' in params and params['mode'] == 'predict':
        classification_output = Lambda(lambda tt: tt[:, -1:], name='Last_Output')(classification_output)

    # Setup metrics
    f1_metric = MaxF1MetricHorizon(is_classification_only=is_classification_only, thresholds=tf.linspace(0.0, 1.0, 11))
    r1_metric = RobustF1Metric(is_classification_only=is_classification_only, thresholds=tf.linspace(0.0, 1.0, 11))
    latency_metric = LatencyMetric(is_classification_only=is_classification_only, max_early_detection=25)

    # Optimizer
    if params and 'TYPE_REG' in params:
        if params['TYPE_REG'].find('AdamW') > -1:
            initial_lr = params['LEARNING_RATE'] * 2.0 if 'LEARNING_RATE' in params else 0.001 * 2.0
            this_optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_lr, clipnorm=1.0)
        elif params['TYPE_REG'].find('Adam') > -1:
            this_optimizer = tf.keras.optimizers.Adam(learning_rate=params.get('LEARNING_RATE', 0.001))
        else:
            this_optimizer = tf.keras.optimizers.Adam(learning_rate=params.get('LEARNING_RATE', 0.001))
    else:
        this_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # # Create loss function - use standard BCE for classification-only model
    patch_ad_loss = PatchADLoss()
    custom_fbfce = custom_fbfce(horizon=hori_shift, loss_weight=loss_weight, params=params) + patch_ad_loss(outputs)

    model.compile(optimizer=this_optimizer,
                  loss=custom_fbfce,
                  metrics=[f1_metric, r1_metric, latency_metric, FalsePositiveMonitorMetric(model=model)])

    if params and 'WEIGHT_FILE' in params and params['WEIGHT_FILE']:
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

class MaxF1MetricHorizon(tf.keras.metrics.Metric):
    def __init__(self, name='max_f1_metric_horizon', thresholds=tf.linspace(0.0, 1.0, 11), model=None, **kwargs):
        super(MaxF1MetricHorizon, self).__init__(name=name, **kwargs)
        self.thresholds = thresholds
        self.tp = self.add_weight(shape=(len(thresholds),), initializer='zeros', name='tp')
        self.fp = self.add_weight(shape=(len(thresholds),), initializer='zeros', name='fp')
        self.fn = self.add_weight(shape=(len(thresholds),), initializer='zeros', name='fn')
        self.model = model

    def update_state(self, y_true, y_pred, **kwargs):
        # Get classification mode from model property
        is_classification_only = getattr(self.model, '_is_classification_only', True)

        # Extract labels based on mode
        y_true_labels = y_true[..., 0] if is_classification_only else y_true[..., 8]
        y_pred_labels = y_pred[..., 0] if is_classification_only else y_pred[..., 8]

        # is_cad = getattr(self.model, "_is_cad", False)
        # print('Using CAD:', is_cad)
        # if is_cad:
        #     print('Using CAD mode and downsampling labels')
        #     pool = tf.keras.layers.MaxPooling1D(pool_size=12, strides=12, padding='valid')
        #     y_true_exp_expanded = tf.expand_dims(y_true_labels, axis=-1)
        #     y_true_labels = tf.squeeze(pool(y_true_exp_expanded), axis=-1)#_expanded)

            
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

class EventAwareF1(tf.keras.metrics.Metric):
    def __init__(self, name="event_f1_metric", thresholds=tf.linspace(0.0, 1.0, 11),
                 early_margin=40, late_margin=40, model=None, **kwargs):
        super(EventAwareF1, self).__init__(name=name, **kwargs)
        self.thresholds = thresholds
        self.early_margin = early_margin
        self.late_margin = late_margin
        self.model = model

        self.tp = self.add_weight(shape=(len(thresholds),), initializer="zeros", name="tp")
        self.fp = self.add_weight(shape=(len(thresholds),), initializer="zeros", name="fp")
        self.fn = self.add_weight(shape=(len(thresholds),), initializer="zeros", name="fn")

    def update_state(self, y_true, y_pred, **kwargs):
        # Determine which branch of outputs to use
        is_classification_only = getattr(self.model, "_is_classification_only", True)
        # print('Using classification only:', is_classification_only)

        # Assume the true labels and predictions have shape [B, T, channels]
        # and we take channel 0 for classification (or channel 8 if not)
        y_true_labels = y_true[..., 0] if is_classification_only else y_true[..., 8]
        y_pred_labels = y_pred[..., 0] if is_classification_only else y_pred[..., 8]
        
        # is_cad = getattr(self.model, "_is_cad", False)
        # print('Using CAD:', is_cad)
        # if is_cad:
        #     print('Using CAD mode and downsampling labels')
        #     pool = tf.keras.layers.MaxPooling1D(pool_size=12, strides=12, padding='valid')
        #     y_true_exp_expanded = tf.expand_dims(y_true_labels, axis=-1)
        #     y_true_labels = tf.squeeze(pool(y_true_exp_expanded), axis=-1)#_expanded)

        # Now y_true_labels and y_pred_labels are [B, T]

        B = tf.shape(y_true_labels)[0]
        T = tf.shape(y_true_labels)[1]

        # Create a vector of time indices for the entire sequence
        time_indices = tf.cast(tf.range(T), tf.int32)  # shape [T]
        # Expand to [B, T]
        time_indices_exp = tf.tile(tf.expand_dims(time_indices, 0), [B, 1])

        # Vectorized function for a single threshold:
        def process_threshold(thresh):
            thresh = tf.cast(thresh, tf.float32)
            # B x T binary predictions
            y_pred_bin = tf.cast(y_pred_labels >= thresh, tf.float32)
            y_true_bin = tf.cast(y_true_labels >= 0.5, tf.float32)

            # For each sample, determine if there is an event and get the first index if so.
            has_event = tf.reduce_max(y_true_bin, axis=1) > 0  # shape [B], bool
            # Note: tf.argmax returns 0 if all are zeros; we rely on has_event to mask these out.
            true_onset = tf.argmax(y_true_bin, axis=1, output_type=tf.int32)  # shape [B]

            # For samples with an event, create lower and upper bounds.
            early_bound = tf.maximum(true_onset - self.early_margin, 0)  # [B]
            late_bound = tf.minimum(true_onset + self.late_margin, T - 1)  # [B]
            # Expand bounds to shape [B, T] for comparison with time_indices_exp:
            early_bound_exp = tf.tile(tf.expand_dims(early_bound, 1), [1, T])
            late_bound_exp = tf.tile(tf.expand_dims(late_bound, 1), [1, T])

            # Create a mask: True where the time index is in [early_bound, late_bound]
            within_bounds = tf.logical_and(time_indices_exp >= early_bound_exp,
                                           time_indices_exp <= late_bound_exp)  # [B, T]

            # For each sample, check if there is any predicted event in the allowed interval.
            # This gives a boolean vector of shape [B]: True if a match is found.
            match = tf.reduce_any(tf.logical_and(tf.equal(y_pred_bin, 1.0), within_bounds), axis=1)
            match = tf.cast(match, tf.float32)  # 1 if match, 0 if not

            # For samples with an event:
            #   - True Positive (TP) = 1 if match, else 0.
            #   - False Negative (FN) = 1 - match.
            tp_event = tf.where(has_event, match, tf.zeros_like(match))
            fn_event = tf.where(has_event, 1 - match, tf.zeros_like(match))
            # For samples without an event:
            #   - False Positives (FP) = total number of predicted events in that sample.
            # fp_no_event = tf.where(~has_event, tf.reduce_sum(y_pred_bin, axis=1), tf.zeros_like(tf.reduce_sum(y_pred_bin, axis=1)))
            fp_no_event = tf.where(~has_event,
                                tf.cast(tf.reduce_any(tf.equal(y_pred_bin, 1.0), axis=1), tf.float32),
                                tf.zeros_like(tf.reduce_sum(y_pred_bin, axis=1)))
            # Sum across the batch for this threshold.
            tp_total = tf.reduce_sum(tp_event)
            fp_total = tf.reduce_sum(fp_no_event)
            fn_total = tf.reduce_sum(fn_event)

            return tp_total, fp_total, fn_total

        # Use tf.map_fn over thresholds (this loop is over a small number of thresholds)
        results = tf.map_fn(lambda t: process_threshold(t), self.thresholds,
                              fn_output_signature=(tf.float32, tf.float32, tf.float32))
        tp_all, fp_all, fn_all = results
        self.tp.assign_add(tp_all)
        self.fp.assign_add(fp_all)
        self.fn.assign_add(fn_all)

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f1_scores = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return tf.reduce_max(f1_scores)

    def reset_state(self):
        self.tp.assign(tf.zeros_like(self.tp))
        self.fp.assign(tf.zeros_like(self.fp))
        self.fn.assign(tf.zeros_like(self.fn))
import tensorflow as tf

class EventFalsePositiveRateMetric(tf.keras.metrics.Metric):
    def __init__(self, name="event_fp_rate", thresholds=tf.linspace(0.0, 1.0, 11), model=None, **kwargs):
        super(EventFalsePositiveRateMetric, self).__init__(name=name, **kwargs)
        self.thresholds = thresholds
        self.model = model
        # Accumulators for FP and TN per threshold.
        self.fp = self.add_weight(shape=(len(thresholds),), initializer="zeros", name="fp")
        self.tn = self.add_weight(shape=(len(thresholds),), initializer="zeros", name="tn")

    def update_state(self, y_true, y_pred, **kwargs):
        # Choose the proper channel based on model flag.
        is_classification_only = getattr(self.model, "_is_classification_only", True)
        y_true_labels = y_true[..., 0] if is_classification_only else y_true[..., 8]
        y_pred_labels = y_pred[..., 0] if is_classification_only else y_pred[..., 8]

        # is_cad = getattr(self.model, "_is_cad", False)
        # print('Using CAD:', is_cad)
        # if is_cad:
        #     print('Using CAD mode and downsampling labels')
        #     pool = tf.keras.layers.MaxPooling1D(pool_size=12, strides=12, padding='valid')
        #     y_true_exp_expanded = tf.expand_dims(y_true_labels, axis=-1)
        #     y_true_labels = tf.squeeze(pool(y_true_exp_expanded), axis=-1)#_expanded)

        # Determine which windows are negative (no event in ground-truth).
        neg_mask = tf.less(tf.reduce_max(y_true_labels, axis=1), 0.5)  # shape: [B], dtype bool

        B = tf.shape(y_true_labels)[0]
        T = tf.shape(y_true_labels)[1]

        # For each threshold (vectorized over the batch dimension)
        def process_threshold(thresh):
            thresh = tf.cast(thresh, tf.float32)
            # For each sample, we check whether any timepoint is predicted above thresh.
            pred_event = tf.greater(tf.reduce_max(y_pred_labels, axis=1), thresh)  # shape: [B], bool
            pred_event_float = tf.cast(pred_event, tf.float32)
            # For negative windows (where neg_mask==True), count FP as 1 if any prediction is above thresh,
            # and TN as 1 if no prediction is above thresh.
            fp_per_sample = tf.where(neg_mask, pred_event_float, tf.zeros_like(pred_event_float, dtype=tf.float32))
            tn_per_sample = tf.where(neg_mask, 1.0 - pred_event_float, tf.zeros_like(pred_event_float, dtype=tf.float32))
            # Sum across the batch for this threshold.
            fp_total = tf.reduce_sum(fp_per_sample)
            tn_total = tf.reduce_sum(tn_per_sample)
            return fp_total, tn_total

        # Map over the thresholds.
        fp_all, tn_all = tf.map_fn(
            lambda t: process_threshold(t),
            self.thresholds,
            fn_output_signature=(tf.float32, tf.float32)
        )

        self.fp.assign_add(fp_all)
        self.tn.assign_add(tn_all)

    def result(self):
        # Compute FP rate for each threshold.
        fp_rate = self.fp / (self.fp + self.tn + tf.keras.backend.epsilon())
        # Return the mean FP rate across thresholds (a scalar).
        return tf.reduce_mean(fp_rate)

    def reset_state(self):
        self.fp.assign(tf.zeros_like(self.fp))
        self.tn.assign(tf.zeros_like(self.tn))



class LatencyMetric(tf.keras.metrics.Metric):
    def __init__(self, name='latency_metric', threshold=0.5, max_early_detection=50, model=None, **kwargs):
        super(LatencyMetric, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.max_early_detection = max_early_detection
        # Initialize with a large positive value so any real latency will be an improvement
        self.total_latency = self.add_weight(name='total_latency', initializer=lambda shape, dtype: tf.constant(1000.0, dtype=dtype))
        self.valid_events = self.add_weight(name='valid_events', initializer='zeros')
        self.model = model

    def update_state(self, y_true, y_pred, **kwargs):
        # Get classification mode from model property
        is_classification_only = getattr(self.model, '_is_classification_only', True)

        # Extract labels based on mode
        y_true_labels = y_true[..., 0] if is_classification_only else y_true[..., 8]
        y_pred_labels = y_pred[..., 0] if is_classification_only else y_pred[..., 8]

        # Calculate onset times using cumsum trick instead of where
        true_shifts = tf.pad(y_true_labels[:, :-1], [[0, 0], [1, 0]], constant_values=0.0)
        onset_mask = tf.cast(tf.logical_and(true_shifts < 0.5, y_true_labels >= 0.5), tf.float32)

        # Calculate detection times
        detection_mask = tf.cast(y_pred_labels >= self.threshold, tf.float32)

        # Find first occurrence using cumsum trick
        onset_cumsum = tf.cumsum(onset_mask, axis=1)
        detection_cumsum = tf.cumsum(detection_mask, axis=1)

        # Get indices of first 1s
        onset_indices = tf.cast(tf.argmax(onset_cumsum > 0, axis=1), tf.float32)
        detection_indices = tf.cast(tf.argmax(detection_cumsum > 0, axis=1), tf.float32)

        # Calculate latency only for valid events
        has_event = tf.reduce_max(onset_cumsum, axis=1) > 0
        has_detection = tf.reduce_max(detection_cumsum, axis=1) > 0
        valid_mask = tf.cast(tf.logical_and(has_event, has_detection), tf.float32)

        # Calculate latency
        latency = detection_indices - onset_indices
        latency = tf.clip_by_value(latency, -self.max_early_detection, tf.cast(tf.shape(y_true_labels)[1], tf.float32))

        # Reset the accumulators for each batch to properly compute the average
        if tf.reduce_sum(valid_mask) > 0:
            self.total_latency.assign(tf.reduce_sum(latency * valid_mask))
            self.valid_events.assign(tf.reduce_sum(valid_mask))
        else:
            # If no valid events in this batch, keep the previous value
            # But for the first batch with no events, set to a large value
            self.total_latency.assign(1000.0 if self.valid_events == 0 else self.total_latency)

    def result(self):
        # Return large positive value if no valid events have been seen
        return tf.cond(
            self.valid_events > 0,
            lambda: self.total_latency / self.valid_events,
            lambda: tf.constant(1000.0, dtype=self.total_latency.dtype)
        )

    def reset_state(self):
        # Initialize to large positive value instead of zero
        self.total_latency.assign(1000.0)
        self.valid_events.assign(0.0)

class RobustF1Metric(tf.keras.metrics.Metric):
    def __init__(self, name='robust_f1', thresholds=tf.linspace(0.0, 1.0, 11), model=None, **kwargs):
        super(RobustF1Metric, self).__init__(name=name, **kwargs)
        self.thresholds = thresholds
        self.tp = self.add_weight(shape=(len(thresholds),), initializer='zeros', name='tp')
        self.fp = self.add_weight(shape=(len(thresholds),), initializer='zeros', name='fp')
        self.fn = self.add_weight(shape=(len(thresholds),), initializer='zeros', name='fn')
        self.pred_sum = self.add_weight(name='pred_sum', initializer='zeros')
        self.pred_count = self.add_weight(name='pred_count', initializer='zeros')
        self.temp_diff_sum = self.add_weight(name='temp_diff_sum', initializer='zeros')
        self.model = model

    def update_state(self, y_true, y_pred, **kwargs):
        # Get classification mode from model property
        is_classification_only = getattr(self.model, '_is_classification_only', True)

        # Extract labels based on mode
        y_true_labels = y_true[..., 0] if is_classification_only else y_true[..., 8]
        y_pred_labels = y_pred[..., 0] if is_classification_only else y_pred[..., 8]

        def process_threshold(threshold):
            pred_events = tf.cast(y_pred_labels >= threshold, tf.float32)
            true_events = tf.cast(y_true_labels >= 0.5, tf.float32)

            tp = tf.reduce_sum(pred_events * true_events)
            fp = tf.reduce_sum(pred_events * (1 - true_events))
            fn = tf.reduce_sum((1 - pred_events) * true_events)

            return tp, fp, fn

        # Process all thresholds using map_fn
        metrics = tf.map_fn(
            process_threshold,
            self.thresholds,
            fn_output_signature=(tf.float32, tf.float32, tf.float32)
        )
        tp_all, fp_all, fn_all = metrics

        # Update main F1 accumulators
        self.tp.assign_add(tp_all)
        self.fp.assign_add(fp_all)
        self.fn.assign_add(fn_all)

        # Update additional metrics
        non_event_pred = y_pred_labels * (1 - y_true_labels)
        self.pred_sum.assign_add(tf.reduce_sum(non_event_pred))
        self.pred_count.assign_add(tf.cast(tf.size(non_event_pred), tf.float32))

        # Temporal consistency using valid time steps only
        temp_diff = tf.abs(y_pred_labels[:, 1:] - y_pred_labels[:, :-1])
        self.temp_diff_sum.assign_add(tf.reduce_sum(temp_diff))

    def result(self):
        # Calculate F1 scores
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f1_scores = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

        # Calculate components
        mean_f1 = tf.reduce_mean(f1_scores)
        f1_stability = 1.0 - tf.math.reduce_std(f1_scores)
        noise_penalty = 1.0 - (self.pred_sum / (self.pred_count + tf.keras.backend.epsilon()))
        temp_consistency = 1.0 - (self.temp_diff_sum / (self.pred_count + tf.keras.backend.epsilon()))

        # Combine metrics with weights
        return (0.7 * mean_f1 +
                0.1 * f1_stability +
                0.1 * noise_penalty +
                0.1 * temp_consistency)

    def reset_state(self):
        self.tp.assign(tf.zeros_like(self.tp))
        self.fp.assign(tf.zeros_like(self.fp))
        self.fn.assign(tf.zeros_like(self.fn))
        self.pred_sum.assign(0.0)
        self.pred_count.assign(0.0)
        self.temp_diff_sum.assign(0.0)


def custom_fbfce(loss_weight=1, horizon=0, params=None, model=None, this_op=None):
    flag_sigmoid = 'SigmoidFoc' in params['TYPE_LOSS']
    is_classification_only = 'Only' in  params['TYPE_ARCH']
    is_cad = 'CAD' in params['TYPE_ARCH']
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
            # For training/validation with LFP predictions (9 or 10 channels)
            # Check if we have enough channels for prediction mode
            # def has_enough_channels():
            prediction_targets = y_true[:, horizon:, :8]  # LFP targets (8 channels)
            prediction_out = y_pred[:, :-horizon, :8]     # LFP predictions (8 channels)
            y_true_exp = tf.expand_dims(y_true[:, :, 8], axis=-1)  # Probability at index 8
            y_pred_exp = tf.expand_dims(y_pred[:, :, 8], axis=-1)  # Predicted probability at index 8
            return prediction_targets, prediction_out, y_true_exp, y_pred_exp

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
        if not is_classification_only:
            prediction_targets, prediction_out, y_true_exp, y_pred_exp = prediction_mode_branch()
            if (len(y_true.shape) == 3) and (y_true.shape[-1] > 9 ):
                sample_weight = y_true[:, :, 9]
            else:
                sample_weight = tf.ones_like(y_true[:, :, 0])
        else:
            # pdb.set_trace()
            prediction_targets, prediction_out, y_true_exp, y_pred_exp = classification_mode_branch()
            
            if (len(y_true.shape) == 3) and (y_true.shape[-1] > 1 ):
                sample_weight = y_true[:, :, 1]
            else:
                sample_weight = tf.ones_like(y_true[:, :, 0])
                
            # if is_cad:
            #     print('Using CAD mode and downsampling labels')
            #     pool = tf.keras.layers.MaxPooling1D(pool_size=12, strides=12, padding='valid')
            #     sample_weight_expanded = tf.expand_dims(sample_weight, axis=-1) 
            #     sample_weight = pool(sample_weight_expanded)
            #     # y_true_exp_expanded = tf.expand_dims(y_true_exp, axis=-1)
            #     y_true_exp = pool(y_true_exp)


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
        total_loss = add_extra_losses(total_loss, y_true_exp, y_pred_exp, sample_weight,
                                     params, model, prediction_targets, prediction_out, this_op)

        # Add prediction loss conditionally
        if not is_classification_only:
            mse_loss = tf.reduce_mean(tf.square(prediction_targets-prediction_out))
            total_loss += loss_weight * mse_loss

        return total_loss
    return loss_fn


def triplet_loss(horizon=0, loss_weight=1, params=None, model=None, this_op=None):
    """
    Custom triplet loss function for SWR prediction that works with tuple outputs.
    """
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

        anchor_emb = anchor_out[..., 1:]
        positive_emb = positive_out[..., 1:]
        negative_emb = negative_out[..., 1:]

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
        alpha = tf.cast(params.get('FOCAL_ALPHA', 0.65), tf.float32)
        gamma = tf.cast(params.get('FOCAL_GAMMA', 1.0), tf.float32)

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
        total_loss = tf.multiply(loss_weight, tf.reduce_mean(triplet_loss_val)) + tf.reduce_mean(total_class_loss)

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
