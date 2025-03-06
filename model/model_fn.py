from tensorflow.keras.layers import Conv1D, Conv2D, ELU, Input, LSTM, Dense, Dropout, Layer, MultiHeadAttention, LayerNormalization, Normalization
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, Concatenate, Lambda
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
from tensorflow_addons.layers import WeightNormalization
from tensorflow_addons.activations import gelu
# from tensorflow_addons.losses import SigmoidFocalCrossEntropy, LiftedStructLoss, TripletSemiHardLoss, TripletHardLoss, ContrastiveLoss
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import applications
import tensorflow as tf
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
    if params['mode']!='train':
        concat_outputs = Lambda(lambda tt: tt[:, -1:, :], name='Last_Output')(concat_outputs)

    model = Model(inputs=inputs, outputs=concat_outputs)

    if flag_sigmoid:
        # f1_metric = MaxF1MetricHorizonMixer()
        # this_binary_accuracy = custom_binary_accuracy_mixer
        f1_metric = MaxF1MetricHorizon(is_classification_only=is_classification_only)
        r1_metric = RobustF1Metric(is_classification_only=is_classification_only)
        # this_binary_accuracy = custom_binary_accuracy
    else:
        f1_metric = MaxF1MetricHorizon(is_classification_only=is_classification_only)
        r1_metric = RobustF1Metric(is_classification_only=is_classification_only)
        latency_metric = LatencyMetric(is_classification_only=is_classification_only)
        # this_binary_accuracy = custom_binary_accuracy

    model.compile(optimizer=this_optimizer,
                  loss=custom_fbfce(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model, this_op=tcn_op),
                  metrics=[MaxF1MetricHorizon(thresholds=tf.linspace(0.0, 1.0, 11)), # Increase threshold granularity
                           RobustF1Metric(), LatencyMetric(max_early_detection=25)])  # Reduce max early detection windowcustom_mse_metric, this_binary_accuracy,

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
    if params['mode']!='train':
        concat_outputs = Lambda(lambda tt: tt[:, -1:, :], name='Last_Output')(concat_outputs)

    model = Model(inputs=inputs, outputs=concat_outputs)

    if flag_sigmoid:
        # f1_metric = MaxF1MetricHorizonMixer()
        # this_binary_accuracy = custom_binary_accuracy_mixer
        f1_metric = MaxF1MetricHorizon(is_classification_only=is_classification_only)
        r1_metric = RobustF1Metric(is_classification_only=is_classification_only)
        this_binary_accuracy = custom_binary_accuracy
    else:
        f1_metric = MaxF1MetricHorizon(is_classification_only=is_classification_only)
        r1_metric = RobustF1Metric(is_classification_only=is_classification_only)
        latency_metric = LatencyMetric(is_classification_only=is_classification_only)
        this_binary_accuracy = custom_binary_accuracy

    model.compile(optimizer=this_optimizer,
                  loss=custom_fbfce(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model, this_op=tcn_op),
                  metrics=[custom_mse_metric, this_binary_accuracy, f1_metric, r1_metric, latency_metric])#, this_embd=tcn_output

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

    if params['mode']=='predict':
        concat_outputs = Lambda(lambda tt: tt[:, -1:, :], name='Last_Output')(concat_outputs)
    # Define model with both outputs
    f1_metric = MaxF1MetricHorizon(is_classification_only=is_classification_only)
    r1_metric = RobustF1Metric(is_classification_only=is_classification_only)
    latency_metric = LatencyMetric(is_classification_only=is_classification_only)

    model = Model(inputs=inputs, outputs=concat_outputs)
    model.compile(optimizer=this_optimizer,
                    loss=custom_fbfce(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model, this_op=tcn_op),
                    metrics=[custom_mse_metric, custom_binary_accuracy, f1_metric, r1_metric, latency_metric]
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

    if params['mode']=='predict':
        concat_outputs = Lambda(lambda tt: tt[:, -1:, :], name='Last_Output')(concat_outputs)
    model = Model(inputs=inputs, outputs=concat_outputs)

    f1_metric = MaxF1MetricHorizon(is_classification_only=is_classification_only)
    r1_metric = RobustF1Metric(is_classification_only=is_classification_only)
    latency_metric = LatencyMetric(is_classification_only=is_classification_only)

    model.compile(optimizer=this_optimizer,
                    loss=custom_fbfce(horizon=hori_shift, loss_weight=loss_weight, params=params, model=model, this_op=tcn_op),
                    metrics=[custom_mse_metric, custom_binary_accuracy, f1_metric, r1_metric, latency_metric]
                  )

    if params['WEIGHT_FILE']:
        print('load model')
        model.load_weights(params['WEIGHT_FILE'])

    return model

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

###### Loss Functions ######

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
    def __init__(self, name='max_f1_metric_horizon', thresholds=tf.linspace(0.0, 1.0, 11), is_classification_only=False, **kwargs):
        super(MaxF1MetricHorizon, self).__init__(name=name, **kwargs)
        self.thresholds = thresholds
        self.tp = self.add_weight(shape=(len(thresholds),), initializer='zeros', name='tp')
        self.fp = self.add_weight(shape=(len(thresholds),), initializer='zeros', name='fp')
        self.fn = self.add_weight(shape=(len(thresholds),), initializer='zeros', name='fn')

    def update_state(self, y_true, y_pred, sample_weight=None, is_classification_only=is_classification_only):
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

class LatencyMetric(tf.keras.metrics.Metric):
    def __init__(self, name='latency_metric', threshold=0.5, max_early_detection=50, is_classification_only=False, **kwargs):
        super(LatencyMetric, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.max_early_detection = max_early_detection
        self.total_latency = self.add_weight(name='total_latency', initializer='zeros')
        self.valid_events = self.add_weight(name='valid_events', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None, is_classification_only=is_classification_only):
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

        # Update metric states
        self.total_latency.assign_add(tf.reduce_sum(latency * valid_mask))
        self.valid_events.assign_add(tf.reduce_sum(valid_mask))

    def result(self):
        return self.total_latency / (self.valid_events + tf.keras.backend.epsilon())

    def reset_state(self):
        self.total_latency.assign(0.)
        self.valid_events.assign(0.)

class RobustF1Metric(tf.keras.metrics.Metric):
    def __init__(self, name='robust_f1', thresholds=tf.linspace(0.0, 1.0, 11), is_classification_only=False, **kwargs):
        super(RobustF1Metric, self).__init__(name=name, **kwargs)
        self.thresholds = thresholds
        self.tp = self.add_weight(shape=(len(thresholds),), initializer='zeros', name='tp')
        self.fp = self.add_weight(shape=(len(thresholds),), initializer='zeros', name='fp')
        self.fn = self.add_weight(shape=(len(thresholds),), initializer='zeros', name='fn')
        self.pred_sum = self.add_weight(name='pred_sum', initializer='zeros')
        self.pred_count = self.add_weight(name='pred_count', initializer='zeros')
        self.temp_diff_sum = self.add_weight(name='temp_diff_sum', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None, is_classification_only=is_classification_only):
        # Extract labels based on mode
        y_true_labels = tf.cast(y_true[..., 0] if is_classification_only else y_true[..., 8], tf.float32)
        y_pred_labels = tf.cast(y_pred[..., 0] if is_classification_only else y_pred[..., 8], tf.float32)

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
    def loss_fn(y_true, y_pred, loss_weight=loss_weight, horizon=horizon, flag_sigmoid=flag_sigmoid, is_classification_only=is_classification_only):
        # Get the last dimension size and determine mode
        print(y_true.shape)
        print(y_pred.shape)

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

        if not is_classification_only:
            prediction_targets, prediction_out, y_true_exp, y_pred_exp = prediction_mode_branch()
            if (len(y_true.shape) == 3) and (y_true.shape[-1] > 9 ):
                sample_weight = y_true[:, :, 9]
            else:
                sample_weight = tf.ones_like(y_true[:, :, 0])
        else:
            prediction_targets, prediction_out, y_true_exp, y_pred_exp = classification_mode_branch()
            if (len(y_true.shape) == 3) and (y_true.shape[-1] > 1 ):
                sample_weight = y_true[:, :, 1]
            else:
                sample_weight = tf.ones_like(y_true[:, :, 0])

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

        # Handle sigmoid activation if needed
        if flag_sigmoid:
            y_pred_exp = tf.math.sigmoid(y_pred_exp)

        # Add extra loss terms
        total_loss = add_extra_losses(total_loss, y_true_exp, y_pred_exp, sample_weight,
                                     params, model, prediction_targets, prediction_out, this_op)

        # Add prediction loss conditionally
        if not is_classification_only:
            mse_loss = tf.reduce_mean(tf.square(prediction_targets-prediction_out))
            total_loss += loss_weight * mse_loss

        return total_loss

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

            print(f"Using Focal loss with alpha={alpha}, gamma={gamma}")

            use_class_balancing = alpha > 0
            focal_loss = tf.keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=use_class_balancing,
                alpha=alpha if alpha > 0 else None,
                gamma=gamma,
                reduction=tf.keras.losses.Reduction.NONE
            )
            loss_values = focal_loss(y_true_exp, y_pred_exp)
            # pdb.set_trace()
            classification_loss = tf.reduce_mean(loss_values * sample_weight)

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
    return loss_fn



############################### LEGACY ###############################

def build_DBI_TCN_Horizon_Updated(input_timepoints, input_chans=8, embedding_dim=32, params=None):
    from tcn import TCN

    inputs = Input(shape=(input_timepoints * 2, input_chans), name='inputs')

    # TCN Backbone
    tcn_layer = TCN(
        nb_filters=params['NO_FILTERS'],
        kernel_size=params['NO_KERNELS'],
        dilations=[2 ** i for i in range(params['NO_DILATIONS'])],
        return_sequences=True,
        use_skip_connections=True,
        dropout_rate=params.get('DROPOUT', 0.0),
        # activation='relu',
        activation=ELU(alpha=1),
        use_weight_norm=True,
        padding='causal'
    )

    tcn_output = tcn_layer(inputs)
    tcn_crop = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Pred_Output')
    tcn_output = tcn_crop(tcn_output)

    # Embedding Layer for Prototype and Barlow Twins Losses
    embedding_layer = Dense(embedding_dim, activation='linear', name='embedding')
    embeddings = embedding_layer(tcn_output)

    # Learnable Prototypes for events and non-events
    p_event = tf.Variable(tf.random.normal([embedding_dim]), trainable=True, name="p_event")
    p_non_event = tf.Variable(tf.random.normal([embedding_dim]), trainable=True, name="p_non_event")

    # Prediction Head for Classification
    classification_output = Dense(1, activation='sigmoid', name='classification_output')(embeddings)
    classification_output = tf.squeeze(classification_output)
    # Build Model
    model = Model(inputs=inputs, outputs=classification_output)

    current_lr = params['LEARNING_RATE']
    optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr)

    # Prototype-Based Loss
    def prototype_loss(embeddings, labels):
        d_event = tf.norm(embeddings - p_event, axis=-1)
        d_non_event = tf.norm(embeddings - p_non_event, axis=-1)
        return tf.reduce_mean(labels * d_event + (1 - labels) * d_non_event)

    def focal_tversky_loss(y_true, y_pred, sample_weight=None, alpha=0.7, beta=0.3, gamma=0.75):
        # Apply sample weights if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=y_true.dtype)
            y_true = y_true * sample_weight
            y_pred = y_pred * sample_weight

        # Calculate Tversky components
        true_pos = tf.reduce_sum(y_true * y_pred)
        false_neg = tf.reduce_sum(y_true * (1 - y_pred))
        false_pos = tf.reduce_sum((1 - y_true) * y_pred)

        # Calculate Tversky index
        tversky_index = (true_pos + 1e-10) / (true_pos + alpha * false_neg + beta * false_pos + 1e-10)

        # Apply focal modulation
        return tf.pow((1 - tversky_index), gamma)

    # Combined Loss
    def combined_loss(y_true, y_pred, embeddings, labels, alpha=0.0, weights=None, params=None):

        total_loss = 0.0
        if params['TYPE_LOSS'].find('Focal')>-1:
            print('Using Focal Loss')
            # bfce = tf.keras.losses.BinaryFocalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            classification_loss = tf.keras.losses.binary_focal_crossentropy(tf.expand_dims(y_true, axis=-1), tf.expand_dims(y_pred, axis=-1))
            total_loss += tf.reduce_mean(classification_loss*weights)
        elif params['TYPE_LOSS'].find('Tversky')>-1:
            print('Using Tversky Loss')
            classification_loss = focal_tversky_loss(y_true, y_pred, sample_weight = weights)
            total_loss += classification_loss
        # pdb.set_trace()

        if params['TYPE_LOSS'].find('Proto')>-1:
            print('Using Proto Loss:')
            loss_weight = (params['TYPE_LOSS'][params['TYPE_LOSS'].find('Proto')+5:params['TYPE_LOSS'].find('Proto')+8])
            weight = 1 if int(loss_weight[0])==1 else -1
            alpha = float(loss_weight[1])*10**(weight*float(loss_weight[2]))
            print('Proto Loss Weight: {0}'.format(alpha))
            if alpha != 0:
                total_loss += prototype_loss(embeddings, labels) * alpha

        return total_loss

    # Training Step with Binary Accuracy Monitoring
    def train_step(x, y,sample_weight=None, params=None):
        y = tf.squeeze(y)
        with tf.GradientTape() as tape:
            # Forward pass
            embeddings = embedding_layer(tcn_crop(tcn_layer(x)))
            predictions = model(x, training=True)

            # Combined loss (classification and prototype)
            total_loss = combined_loss(y, predictions, embeddings, y, alpha=5.0, weights=sample_weight, params=params)

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Calculate binary accuracy
        binary_accuracy = tf.reduce_mean(tf.keras.metrics.binary_accuracy(y, predictions))
        return total_loss, binary_accuracy

    def train_model(train_dataset, val_dataset, params=None, save_best=True, patience=1, factor=0.5, min_lr=1e-5):

        # Track the best validation F1 score
        best_val_f1 = 0.0
        epochs = params['NO_EPOCHS']

        # learning rate scheduler
        no_improve_epochs = 0

        # Instantiate F1 metric outside of `evaluate`
        val_f1_metric = MaxF1Metric()
        val_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
        val_precision_metric = tf.keras.metrics.Precision()
        val_recall_metric = tf.keras.metrics.Recall()
        val_loss_metric = tf.keras.metrics.Mean()

        for epoch in range(epochs):

            current_lr = optimizer.learning_rate.numpy()
            # Initialize metrics for the current epoch
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy_avg = tf.keras.metrics.Mean()

            # Training loop
            for inputs, labels, weights in train_dataset:
                # Perform a training step
                loss, accuracy = train_step(inputs, labels, sample_weight=weights, params=params)
                epoch_loss_avg.update_state(loss)
                epoch_accuracy_avg.update_state(accuracy)

            # Run validation and get F1 score, accuracy, precision, recall, and loss
            val_f1, val_accuracy, val_precision, val_recall, val_loss = evaluate(val_dataset, val_f1_metric, val_accuracy_metric, val_precision_metric, val_recall_metric, val_loss_metric)

            # Print epoch results
            print(f"Epoch {epoch+1}, Loss: {epoch_loss_avg.result().numpy()}, "
                  f"Accuracy: {epoch_accuracy_avg.result().numpy()}, "
                  f"Validation Loss: {val_loss:.4f}, "
                  f"Validation F1 Score: {val_f1:.4f}, "
                  f"Validation Accuracy: {val_accuracy:.4f}, "
                  f"Validation Precision: {val_precision:.4f}, "
                  f"Validation Recall: {val_recall:.4f}")

            # Save model if the F1 score improved
            if save_best and val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improve_epochs = 0
                model.save(params['EXP_DIR'] + '/best_f1_model.h5')
                print(f"New best model saved with F1 Score: {best_val_f1:.4f}")
            else:
                no_improve_epochs += 1
            # print('No Improve Epochs:', no_improve_epochs)

            # Reset metrics for the next epoch
            epoch_loss_avg.reset_states()
            epoch_accuracy_avg.reset_states()
            val_f1_metric.reset_states()
            val_accuracy_metric.reset_states()
            val_precision_metric.reset_states()
            val_recall_metric.reset_states()
            val_loss_metric.reset_states()

            # Callbacks at the end of each epoch
            if no_improve_epochs >= patience:
                new_lr = max(current_lr * factor, min_lr)
                if new_lr < current_lr:
                    current_lr = new_lr
                    optimizer.learning_rate.assign(current_lr)
                    print(f"Reduced learning rate to {current_lr}")
                else:
                    print(f"Learning rate already at minimum value of {min_lr}, Early stopping")
                    break
                no_improve_epochs = 0  # Reset counter after reducing learning rate

    def evaluate(val_dataset, val_f1_metric, val_accuracy_metric, val_precision_metric, val_recall_metric, val_loss_metric):
        # Reset metric state
        val_f1_metric.reset_states()
        val_accuracy_metric.reset_states()
        val_precision_metric.reset_states()
        val_recall_metric.reset_states()
        val_loss_metric.reset_states()

        # List to store F1 scores at each threshold
        val_f1_scores = []

        # Iterate through validation dataset
        for inputs, labels in val_dataset:
            predictions = model(inputs, training=False)

            # Update metrics based on labels and predictions
            val_f1_score = val_f1_metric(labels, predictions)
            val_accuracy_metric.update_state(labels, predictions)
            val_precision_metric.update_state(labels, predictions)
            val_recall_metric.update_state(labels, predictions)
            val_loss_metric.update_state(labels, predictions)
            val_f1_scores.append(val_f1_score)

        # Ensure `f1_scores` contains values before calculating max F1
        if val_f1_scores:
            max_f1_score = tf.reduce_max(tf.stack(val_f1_scores))
        else:
            max_f1_score = 0.0  # or other default value

        val_accuracy = val_accuracy_metric.result().numpy()
        val_precision = val_precision_metric.result().numpy()
        val_recall = val_recall_metric.result().numpy()
        val_loss = val_loss_metric.result().numpy()

        return max_f1_score, val_accuracy, val_precision, val_recall, val_loss
    return model, train_model

# Prototype Update Function
def update_prototypes(embeddings, labels, p_event, p_non_event):
    event_embeddings = tf.boolean_mask(embeddings, labels)
    non_event_embeddings = tf.boolean_mask(embeddings, 1 - labels)

    if tf.reduce_sum(tf.cast(labels, tf.float32)) > 0:
        p_event.assign(tf.reduce_mean(event_embeddings, axis=0))
    if tf.reduce_sum(tf.cast(1 - labels, tf.float32)) > 0:
        p_non_event.assign(tf.reduce_mean(non_event_embeddings, axis=0))

def build_DBI_TCN_HorizonBarlow(input_timepoints, input_chans=8, params=None):

    def augment_signal(signal):
        # Example augmentations: noise, scaling, and temporal shifts
        noise = tf.random.normal(shape=tf.shape(signal), mean=0.0, stddev=0.1)
        scaled = signal * tf.random.uniform(shape=[], minval=0.1, maxval=10.1)
        # shifted = tf.roll(signal, shift=tf.random.uniform(shape=[], minval=-5, maxval=5, dtype=tf.int32), axis=1)
        return scaled + noise

    def create_augmented_views(inputs):
        view1 = augment_signal(inputs)
        view2 = augment_signal(inputs)
        return view1, view2

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

    # Combined Loss
    def total_barlow_loss(y_true_exp, y_pred_exp, prediction_targets, prediction_out, embeddings1, embeddings2, horizon=0, loss_weight=1, params=None, model=None, sample_weight=None):
        if params['TYPE_LOSS'].find('FocalSmooth')>-1:
            print('FocalSmooth')
            total_loss = tf.keras.losses.binary_focal_crossentropy(y_true_exp, y_pred_exp, apply_class_balancing=True, label_smoothing=0.1)
        elif params['TYPE_LOSS'].find('FocalSmoothless')>-1:
            print('FocalRegular')
            total_loss = tf.keras.losses.binary_focal_crossentropy(y_true_exp, y_pred_exp, apply_class_balancing=True)
        elif params['TYPE_LOSS'].find('SigmoidFoc')>-1:
            print('Sigmoid Focal Loss')
            aind = params['TYPE_LOSS'].find('Ax')+2
            alp = float(params['TYPE_LOSS'][aind:aind+3])/100
            gind = params['TYPE_LOSS'].find('Gx')+2
            gam = float(params['TYPE_LOSS'][gind:gind+3])/100
            print('Alpha: {0}, Gamma: {1}'.format(alp, gam))
            sigmoid_focal_loss = SigmoidFocalCrossEntropy(from_logits=True,alpha=alp,gamma=gam,reduction='none')
            total_loss = tf.reduce_mean(tf.multiply(sigmoid_focal_loss(y_true_exp, y_pred_exp), sample_weight))
        elif params['TYPE_LOSS'].find('Focal')>-1:
            print('Focal')
            aind = params['TYPE_LOSS'].find('Ax')+2
            alp = float(params['TYPE_LOSS'][aind:aind+3])/100
            gind = params['TYPE_LOSS'].find('Gx')+2
            gam = float(params['TYPE_LOSS'][gind:gind+3])/100
            print('Alpha: {0}, Gamma: {1}'.format(alp, gam))
            if alp == 0:
                focal_loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=False,gamma=gam,axis=-1,reduction='none')
                total_loss = tf.reduce_mean(tf.multiply(focal_loss(y_true_exp, y_pred_exp), sample_weight))
            else:
                focal_loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True,alpha=alp,gamma=gam,axis=-1,reduction='none')
                total_loss = tf.reduce_mean(tf.multiply(focal_loss(y_true_exp, y_pred_exp), sample_weight))
        # add prediction loss
        # mse_loss = tf.reduce_mean(tf.multiply(tf.square(prediction_targets-prediction_out), sample_weight))
        mse_loss = tf.reduce_mean(tf.square(prediction_targets-prediction_out))
        total_loss += loss_weight * mse_loss

        # barlow loss for embedding robustness
        barlow_weight = 5e-1
        barlos_loss = barlow_weight * (barlow_twins_loss(embeddings1, embeddings2, lambda_param=0.005))# +
                                    #    barlow_twins_loss(embeddings1, prediction_out, lambda_param=0.005))
        total_loss += barlos_loss
        return total_loss

    # Loss Function
    @tf.function
    def train_step(inputs, y_true, params=None, horizon=0, loss_weight=1):
        with tf.GradientTape() as tape:

            # Augmented Views
            view1, view2 = tf.keras.layers.Lambda(create_augmented_views)(inputs)
            out_view1 = tcn_op(view1)
            out_view2 = tcn_op(view2)
            embeddings1 = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output1')(out_view1)
            embeddings2 = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output2')(out_view2)
            embeddings1 = embeddings1[:, horizon:, :]
            embeddings2 = embeddings2[:, horizon:, :]
            # Predictions
            predictions = model(inputs, training=True)

            # split inputs outputs
            prediction_targets = y_true[:, horizon:, :8]                   # Targets starting from horizon
            prediction_out = predictions[:, :-horizon, :8]                 # Predicted outputs before the horizon
            y_true_exp = tf.expand_dims(y_true[:, :, 8], axis=-1)
            y_pred_exp = tf.expand_dims(predictions[:, :, 8], axis=-1)     # First 8 channels for horizon prediction
            if y_true.shape[-1]==10:
                sample_weight = y_true[:, :, 9]
            else:
                sample_weight = tf.ones_like(y_true_exp)

            # Compute losses
            total_loss = total_barlow_loss(y_true_exp, y_pred_exp, prediction_targets, prediction_out, embeddings1, embeddings2,
                                           loss_weight=1, horizon=0, params=params, model=None, sample_weight=sample_weight)

        # Backpropagation
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Calculate binary accuracy
        binary_accuracy = tf.reduce_mean(tf.keras.metrics.binary_accuracy(y_true_exp, y_pred_exp))
        return total_loss, binary_accuracy

    def train_model(train_dataset, val_dataset, params=None, save_best=True, patience=20, factor=0.5, min_lr=1e-5):

        # loss parameters
        hori_shift = 0  # Default value
        if params['TYPE_ARCH'].find('Hori')>-1:
            hori_shift = int(int(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Hori')+4:params['TYPE_ARCH'].find('Hori')+6])/1000*params['SRATE'])
            print('Using Horizon Timesteps:', hori_shift)

        if params['TYPE_ARCH'].find('Loss')>-1:
            loss_weight = (params['TYPE_ARCH'][params['TYPE_ARCH'].find('Loss')+4:params['TYPE_ARCH'].find('Loss')+7])
            weight = 1 if int(loss_weight[0])==1 else -1
            loss_weight = float(loss_weight[1])*10**(weight*float(loss_weight[2]))
            print('Using Loss Weight:', loss_weight)

        # Track the best validation F1 score
        best_val_f1 = 0.0
        epochs = params['NO_EPOCHS']

        # learning rate scheduler
        no_improve_epochs = 0

        # Instantiate F1 metric outside of `evaluate`
        val_f1_metric = MaxF1MetricHorizon()
        val_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
        val_loss_metric = tf.keras.metrics.Mean()

        print('Training model...')
        for epoch in range(epochs):

            current_lr = optimizer.learning_rate.numpy()
            # Initialize metrics for the current epoch
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy_avg = tf.keras.metrics.Mean()

            # Training loop
            for inputs, labels in train_dataset:
                # drop some batches
                if tf.random.uniform([]) < 0.2:
                    continue

                # Perform a training step
                loss, accuracy = train_step(inputs, labels, params=params, horizon=hori_shift, loss_weight=loss_weight)
                epoch_loss_avg.update_state(loss)
                epoch_accuracy_avg.update_state(accuracy)

            # Run validation and get F1 score, accuracy, precision, recall, and loss
            val_f1, val_accuracy, val_loss = evaluate(val_dataset, val_f1_metric, val_accuracy_metric, val_loss_metric)

            # Print epoch results
            print(f"Epoch {epoch+1}, Loss: {epoch_loss_avg.result().numpy()}, "
                  f"Accuracy: {epoch_accuracy_avg.result().numpy()}, "
                  f"Validation Loss: {val_loss:.4f}, "
                  f"Validation F1 Score: {val_f1:.4f}, "
                  f"Validation Accuracy: {val_accuracy:.4f}, "
                  )

            # Save model if the F1 score improved
            if save_best and val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improve_epochs = 0
                model.save(params['EXP_DIR'] + '/best_f1_model.h5')
                print(f"New best model saved with F1 Score: {best_val_f1:.4f}")
            else:
                no_improve_epochs += 1
            # print('No Improve Epochs:', no_improve_epochs)

            # Reset metrics for the next epoch
            epoch_loss_avg.reset_states()
            epoch_accuracy_avg.reset_states()
            val_f1_metric.reset_states()
            val_accuracy_metric.reset_states()
            val_loss_metric.reset_states()

            # Callbacks at the end of each epoch
            if no_improve_epochs >= patience:
                new_lr = max(current_lr * factor, min_lr)
                if new_lr < current_lr:
                    current_lr = new_lr
                    optimizer.learning_rate.assign(current_lr)
                    print(f"Reduced learning rate to {current_lr}")
                else:
                    print(f"Learning rate already at minimum value of {min_lr}, Early stopping")
                    break
                no_improve_epochs = 0  # Reset counter after reducing learning rate

    def evaluate(val_dataset, val_f1_metric, val_accuracy_metric, val_loss_metric):

        # List to store F1 scores at each threshold
        val_f1_scores = []

        # Iterate through validation dataset
        for inputs, labels in val_dataset:
            predictions = model(inputs, training=False)

            # Update metrics based on labels and predictions
            val_f1_score = val_f1_metric(labels, predictions)
            val_accuracy_metric.update_state(labels[:, :, 8], predictions[:, :, 8])
            val_loss_metric.update_state(labels[:, :, 8], predictions[:, :, 8])
            val_f1_scores.append(val_f1_score)

        # Ensure `f1_scores` contains values before calculating max F1
        if val_f1_scores:
            max_f1_score = tf.reduce_max(tf.stack(val_f1_scores))
        else:
            max_f1_score = 0.0  # or other default value

        val_accuracy = val_accuracy_metric.result().numpy()
        val_loss = val_loss_metric.result().numpy()

        return max_f1_score, val_accuracy, val_loss

    # logit oer sigmoid
    flag_sigmoid = 'SigmoidFoc' in params['TYPE_LOSS']
    print('Using Sigmoid:', flag_sigmoid)
    # params
    use_batch_norm = params['TYPE_REG'].find('BN')>-1
    use_weight_norm = params['TYPE_REG'].find('WReg')>-1
    use_layer_norm = params['TYPE_REG'].find('LN')>-1
    # this_activation = 'relu'
    this_activation = ELU(alpha=1)
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

    # load labels
    # inputs = Input(shape=(None, input_chans), name='inputs')
    inputs = Input(shape=(None, input_chans), name='inputs')

    if params['TYPE_ARCH'].find('ZNorm')>-1:
        print('Using ZNorm')
        inputs = Normalization()(inputs)

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
                        # dilations=(1, 2, 4, 8, 16), #, 32
                        padding='causal',
                        use_skip_connections=True,
                        dropout_rate=r_drop,
                        return_sequences=True,
                        activation=this_activation,
                        kernel_initializer=this_kernel_initializer,
                        # activation=this_activation,
                        # kernel_initializer=this_kernel_initializer,
                        use_batch_norm=use_batch_norm,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
        print(tcn_op.receptive_field)
        nets = tcn_op(inputs_nets)
        nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)

    if params['TYPE_ARCH'].find('L2N')>-1:
        tcn_output = Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(nets)
    else:
        tcn_output = nets

    # tcn_output = nets
    output_dim = 8
    prediction_output = Conv1D(output_dim, kernel_size=1, use_bias=True, activation='linear', name='prediction_output')(tcn_output)  # Output future values
    tmp_class = Conv1D(1, kernel_size=1, activation='sigmoid', use_bias=True, name='tmp_class')(tcn_output)

    # add confidence layer
    if params['TYPE_ARCH'].find('Confidence')>-1:
        print('Using Confidence Inputs')
        conf_inputs = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Confidence')(inputs)
        confidence = tf.reduce_mean(tf.square(conf_inputs-prediction_output), axis=-1, keepdims=True)
        tmp_class = Concatenate(axis=-1)([tmp_class, confidence])

    # compute probability
    classification_output = tmp_class
    concat_outputs = Concatenate(axis=-1)([prediction_output, classification_output])

    # Define model with both outputs
    if params['mode']!='train':
        concat_outputs = Lambda(lambda tt: tt[:, -1:, :], name='Last_Output')(concat_outputs)

    model = Model(inputs=inputs, outputs=concat_outputs)

    if params['WEIGHT_FILE']:
        print('load model')
        model.load_weights(params['WEIGHT_FILE'])

    current_lr = params['LEARNING_RATE']
    optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr)
    return model, train_model

