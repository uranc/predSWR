from tensorflow.keras.layers import Conv1D, Conv2D, ELU, Input, LSTM, Dense, Dropout, Layer, MultiHeadAttention, LayerNormalization, Normalization
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, Concatenate, Lambda
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import applications
# import tensorflow_addons as tfa
import tensorflow as tf
import pdb

class AttentivePooler(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, num_queries=1):
        super(AttentivePooler, self).__init__()
        self.query_tokens = self.add_weight(shape=(1, num_queries, embed_dim), initializer="random_normal", trainable=True)
        self.cross_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = tf.tile(self.query_tokens, [batch_size, 1, 1])  # Replicate query tokens across the batch
        attention_output, _ = self.cross_attention(query, inputs, return_attention_scores=True)
        return attention_output

def build_DBI_CNN(mode, params):

    # load labels
    nets_in = Input(shape=(84, 84, 3))
    nrows, ncols = nets_in.get_shape().as_list()[1:3]

    # get vgg16 layers
    vgg_model = get_vgg(nrows, ncols,
                        out_layer=['block1_conv2'],
                        is_trainable=False)
    vgg_outs = vgg_model(nets_in)

    tmp_1 = Conv2D(128, (3, 3),
            activation=None,
            padding='valid',
            use_bias=True,
            kernel_regularizer=l1(0.001),
            kernel_initializer='he_normal',
            name='conv_{0}'.format(1))(tmp_1)
    tmp_1 = ELU(alpha=0)(tmp_1)

    # get outputs
    outputs = tmp_1
    model = Model(inputs=[nets_in], outputs=[outputs])
    if params['WEIGHT_DIR']:
        print('load model')
        model.load_weights(params['WEIGHT_DIR'])

    model.compile(optimizer=tf.keras.optimizers.AdamW(lr=params['LEARNING_RATE']),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])
    return model


def build_multi_TCN(input_shape):
    inputs = Input(shape=input_shape)

    # Multiple TCNs with different kernel sizes and small dilations
    # if params['MASKING'].find('L1')>-1:
    #     from tensorflow.keras.layers import Masking
    #     masked = Masking(mask_value=0.0)(inputs)
    #     x = Dropout(rate=0.3)(masked)  # Drop 30% of the channels

    # Multiple TCNs with different kernel sizes and small dilations
    tcn1 = TCN(nb_filters=64, kernel_size=3, dilations=[1, 2, 4], return_sequences=True)(inputs)
    tcn2 = TCN(nb_filters=64, kernel_size=5, dilations=[1, 2, 4], return_sequences=True)(inputs)
    tcn3 = TCN(nb_filters=64, kernel_size=7, dilations=[1, 2], return_sequences=True)(inputs)

    # Concatenate outputs from all TCNs
    concatenated = Concatenate()([tcn1, tcn2, tcn3])

    if params['ATTENTION']:
        from tensorflow.keras.layers import Attention
        concatenated = Attention()([concatenated, concatenated])


    # Dense layer to combine the information from all scales
    combined_output = Dense(64, activation='relu')(concatenated)
    final_output = Dense(1, activation='sigmoid')(combined_output)

    model = Model(inputs, final_output)
    return model

def build_DBI_TCN(input_timepoints, input_chans=8, params=None):

    # params
    use_batch_norm = False
    use_weight_norm = True
    use_layer_norm = False
    use_l1_norm = False
    # this_activation = 'relu'
    this_activation = ELU(alpha=1)
    # this_activation = ELU(alpha=0.1)
    this_kernel_initializer = 'glorot_uniform'
    # this_kernel_initializer = 'he_normal'
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
    # inputs = Input(shape=(input_timepoints*2, input_chans), name='inputs')
    # inputs = Input(shape=(input_timepoints, input_chans), name='inputs')
    inputs = Input(shape=(None, input_chans), name='inputs')

    # import pdb
    # pdb.set_trace()
    # csd_inputs = CSDLayer()(inputs)

    #params['MASKING'].find('L1')>-1:
    flagMask = False
    if flagMask:
        from tensorflow.keras.layers import Masking
        inputs = Masking(mask_value=0.0)(inputs)
        # x = Dropout(rate=0.3)(masked)  # Drop 30% of the channels

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
        nets = tcn_op(inputs)

        # Slice & Out
        if params['mode'] == 'train':
            nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)
        else:
            nets = Lambda(lambda tt: tt[:, -1:, :], name='Slice_Output')(nets)

        # if use_weight_norm:
        #     wconv = WeightNormalization(wconv)
        # nets = wconv(nets)

        if params['TYPE_ARCH'].find('SelfAttention')>-1:
            print('Using SelfAttention')
            attention_layer = MultiHeadAttention(num_heads=4, key_dim=32, name='temporal_attention')
            nets = attention_layer(nets, nets)
        elif params['TYPE_ARCH'].find('LearnedAttention')>-1:
            print('Using LearnedAttention')
            nets = AttentivePooler(embed_dim=32, num_heads=4)(nets)  # Adjust embed_dim to match TCN output


        # sigmoid out
        nets = Dense(1, activation='sigmoid')(nets)
        # nets = Dense(1, activation='linear', kernel_initializer=this_kernel_initializer)(nets)
    elif model_type=='BaseAvg':
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
                        use_layer_norm=False,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
        print(tcn_op.receptive_field)
        dense_op = Dense(1, activation='sigmoid')

        nets = []
        for i_ch in range(8):
            nets_in = Lambda(lambda tt: tt[:, :, i_ch], name='Slice_Input_{}'.format(i_ch))(inputs)
            tcn_out = tcn_op(tf.expand_dims(nets_in, axis=-1))
            # nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)
            nets.append((tcn_out))

        nets = Concatenate(axis=-1)(nets)
        nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)
        nets = dense_op(nets)
        # nets = tf.reduce_max(nets, axis=-1, keepdims=True)

        # Slice & Out
        # nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)

    elif model_type=='Average':
        nets = []
        for i_ch in range(8):
            nets_in = Lambda(lambda tt: tt[:, :, i_ch], name='Slice_Input_{}'.format(i_ch))(inputs)
            nets.append(tf.expand_dims(nets_in, axis=-1))

        dropout_rate = 0
        all_nets = []
        for i_dilate in range(n_dilations):
            wconv = Conv1D(filters=n_filters,
                            kernel_size=n_kernels,
                            dilation_rate=2**(i_dilate),
                            # groups=input_chans,
                            padding='causal',
                            activation=this_activation,
                            use_bias = True,
                            bias_initializer='zeros',
                            # kernel_regularizer=l1(0.001),
                            # activation=ELU(alpha=0),
                            name='dconv1_{0}'.format(i_dilate),
                            kernel_initializer=this_kernel_initializer
                            # kernel_initializer='he_normal'
                            )
            if use_weight_norm:
                wconv = WeightNormalization(wconv)

            if i_dilate == 0:
                all_chs = []
                for i_ch in range(8):
                    if use_batch_norm:
                        tmp = wconv(nets[i_ch])
                        tmp = BatchNormalization()(tmp)
                        all_chs.append(tmp)
                    else:
                        all_chs.append(wconv(nets[i_ch]))
            else:
                tmp_chs = []
                for i_ch in range(8):
                    if use_batch_norm:
                        tmp = wconv(all_chs[i_ch])
                        tmp = BatchNormalization()(tmp)
                        tmp_chs.append(tmp)
                    else:
                        tmp_chs.append(wconv(all_chs[i_ch]))
                all_chs = tmp_chs

        # reduce dilation
        wconv = Conv1D(filters=n_filters/2,
                        kernel_size=n_kernels,
                        dilation_rate=1,
                        # groups=input_chans,
                        padding='causal',
                        use_bias = True,
                        bias_initializer='zeros',
                        activation=this_activation,
                        # kernel_regularizer=l1(0.001),
                        # activation=ELU(alpha=0),
                        name='de_dconv1_{0}'.format(0),
                        # kernel_initializer='he_normal'
                        kernel_initializer=this_kernel_initializer
                        )
        # pdb.set_trace()
        if use_weight_norm:
            wconv = WeightNormalization(wconv)
        if use_batch_norm:
            wconv = BatchNormalization()(wconv)

        tmp_chs = []
        for i_ch in range(8):
            if use_batch_norm:
                tmp = wconv(all_chs[i_ch])
                tmp = BatchNormalization()(tmp)
                tmp_chs.append(tmp)
            else:
                tmp_chs.append(wconv(all_chs[i_ch]))
        all_chs = tmp_chs

        # reduce dilation
        wconv = Conv1D(filters=n_filters/4,
                        kernel_size=n_kernels,
                        dilation_rate=1,
                        # groups=input_chans,
                        padding='causal',
                        use_bias = True,
                        bias_initializer='zeros',
                        activation=this_activation,
                        # kernel_regularizer=l1(0.001),
                        # activation=ELU(alpha=0),
                        name='de_dconv1_{0}'.format(1),
                        kernel_initializer=this_kernel_initializer
                        # kernel_initializer='he_normal'
                        )
        if use_weight_norm:
            wconv = WeightNormalization(wconv)
        if use_batch_norm:
            wconv = BatchNormalization()(wconv)

        dense_op = Dense(1,activation='sigmoid')
        tmp_chs = []
        for i_ch in range(8):
            if use_batch_norm:
                tmp = wconv(all_chs[i_ch])
                tmp = BatchNormalization()(tmp)
                tmp_chs.append(tmp)
            else:
                tmp_chs.append(dense_op(wconv(all_chs[i_ch])))
        all_chs = tmp_chs

        # concat
        nets = Concatenate(axis=-1)(all_chs)
        nets = tf.reduce_max(nets, axis=-1, keepdims=True)
        nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)

    # get outputs
    outputs = nets
    model = Model(inputs=[inputs], outputs=[outputs])
    # pdb.set_trace()
    if params['WEIGHT_FILE']:
        print('load model')
        model.load_weights(params['WEIGHT_FILE'])


    def custom_fbfce(weights=None):

        # pdb.set_trace()
        """Loss function"""
        def loss_fn(y_true, y_pred, weights=weights):
            if params['TYPE_LOSS'].find('FocalSmooth')>-1:
                print('FocalSmooth')
                total_loss = tf.keras.losses.binary_focal_crossentropy(y_true, y_pred, apply_class_balancing=True, label_smoothing=0.1)
            elif params['TYPE_LOSS'].find('Focal')>-1:
                print('Focal')
                aind = params['TYPE_LOSS'].find('Ax')+2
                alp = float(params['TYPE_LOSS'][aind:aind+3])/100
                gind = params['TYPE_LOSS'].find('Gx')+2
                gam = float(params['TYPE_LOSS'][gind:gind+3])/100
                print('Alpha: {0}, Gamma: {1}'.format(alp, gam))
                if alp == 0:
                    total_loss = tf.keras.losses.binary_focal_crossentropy(y_true, y_pred, gamma = gam)
                else:
                    total_loss = tf.keras.losses.binary_focal_crossentropy(y_true, y_pred, apply_class_balancing=True, alpha = alp, gamma = gam)
            elif params['TYPE_LOSS'].find('Anchor')>-1:
                print('Anchor')
                aind = params['TYPE_LOSS'].find('Ax')+2
                alp = float(params['TYPE_LOSS'][aind:aind+3])/100
                gind = params['TYPE_LOSS'].find('Gx')+2
                gam = float(params['TYPE_LOSS'][gind:gind+3])/100
                print('Alpha: {0}, Gamma: {1}'.format(alp, gam))
                if alp == 0:
                    total_loss = tf.keras.losses.binary_focal_crossentropy(y_true, y_pred, gamma = gam)
                else:
                    total_loss = tf.keras.losses.binary_focal_crossentropy(y_true, y_pred, apply_class_balancing=True, alpha = alp, gamma = gam)
            else:
                pdb.set_trace()
            if params['TYPE_LOSS'].find('TV')>-1:
                print('TV Loss')
                # total_loss += 1e-5*tf.image.total_variation(y_pred)
                total_loss += 1e-5*tf.reduce_sum(tf.image.total_variation(tf.expand_dims(y_pred, axis=-1)))
            if params['TYPE_LOSS'].find('L2')>-1:
                print('L2 smoothness Loss')
                total_loss += 1e-5*tf.reduce_mean((y_pred[1:]-y_pred[:-1])**2)
            if params['TYPE_LOSS'].find('Margin')>-1:
                print('Margin Loss')
                # pdb.set_trace()
                total_loss += 1e-4*tf.squeeze(y_pred * (1 - y_pred))

            return total_loss
        return loss_fn

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE']),
                    loss=custom_fbfce(),
                    metrics=[tf.keras.metrics.BinaryCrossentropy(),
                             tf.keras.metrics.BinaryAccuracy()])
    return model

def build_DBI_TCN_Horizon(input_timepoints, input_chans=8, params=None):

    # params
    use_batch_norm = params['TYPE_REG'].find('BN')>-1
    use_weight_norm = params['TYPE_REG'].find('WReg')>-1
    use_layer_norm = False
    use_l1_norm = False
    # this_activation = 'relu'
    # this_activation = tf.keras.activations.gelu
    this_activation = ELU(alpha=1)
    # this_activation = ELU(alpha=0.1)
    this_kernel_initializer = 'glorot_uniform'
    # this_kernel_initializer = 'he_normal'
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

    hori_shift = 0  # Default value
    if params['TYPE_ARCH'].find('Hori')>-1:
        print('Using Horizon Timesteps:')
        hori_shift = int(int(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Hori')+4:params['TYPE_ARCH'].find('Hori')+6])/1000*1250)
        print(hori_shift)

    if params['TYPE_ARCH'].find('Loss')>-1:
        print('Using Loss Weight:')
        loss_weight = (params['TYPE_ARCH'][params['TYPE_ARCH'].find('Loss')+4:params['TYPE_ARCH'].find('Loss')+7])
        weight = 1 if int(loss_weight[0])==1 else -1
        loss_weight = float(loss_weight[1])*10**(weight*float(loss_weight[2]))
        print(loss_weight)

    # load labels
    # inputs = Input(shape=(None, input_chans), name='inputs')
    inputs = Input(shape=(100, input_chans), name='inputs')

    if params['TYPE_ARCH'].find('CSD')>-1:
        csd_inputs = CSDLayer()(inputs)
        inputs_nets = Concatenate(axis=-1)([inputs, csd_inputs])
    else:
        inputs_nets = inputs

    if params['TYPE_ARCH'].find('ZNorm')>-1:
        print('Using ZNorm')
        inputs_nets = Normalization()(inputs_nets)

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
                        use_batch_norm=False,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
        print(tcn_op.receptive_field)
        nets = tcn_op(inputs_nets)
        nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)

    tcn_output = nets
    output_dim = 8
    tmp_pred = Dense(32, activation=this_activation, name='tmp_pred')(tcn_output)  # Output future values
    prediction_output = Dense(output_dim, activation='linear', name='prediction_output')(tmp_pred)  # Output future values

    if params['TYPE_ARCH'].find('SelfPosAtt')>-1:
        print('Using Self Positional Attention')
        # compute csd of the predicted values as well
        pred_CSD = CSDLayer()(prediction_output)
        att_in = Concatenate(axis=-1)([pred_CSD, prediction_output])
        embed_dim, num_heads, num_channels = 16, 4, 50  # Adjust based on model structure
        attention_layer = SelfAttentionPositional(embed_dim=embed_dim, num_heads=num_heads, num_channels=num_channels)
        attentive_output = attention_layer(att_in)
        tcn_output = Concatenate(axis=-1)([tcn_output, attentive_output])
        # pdb.set_trace()

    elif params['TYPE_ARCH'].find('LayerAtt')>-1:
        print('Using LearnedAttention')
        pred_CSD = CSDLayer()(prediction_output)

        lfp_output = TwoStageAttentivePooling(embed_dim=8, num_heads=4, num_queries=50)(prediction_output)
        csd_output = TwoStageAttentivePooling(embed_dim=8, num_heads=4, num_queries=50)(pred_CSD)
        attentive_output = Concatenate(axis=-1)([lfp_output, csd_output])
        tcn_output = Concatenate(axis=-1)([tcn_output, attentive_output])

    # sigmoid out
    tmp_class = Dense(32, activation=this_activation, name='tmp_class')(tcn_output)

    # add confidence layer
    if params['TYPE_ARCH'].find('Confidence')>-1:
        print('Using Confidence Inputs')
        conf_inputs = Lambda(lambda tt: tt[:, -50:, :], name='Slice_Confidence')(inputs)
        confidence = tf.reduce_mean(tf.square(conf_inputs-prediction_output), axis=-1, keepdims=True)
        tmp_class = Concatenate(axis=-1)([tmp_class, confidence])

    # compute probability
    classification_output = Dense(1, activation='sigmoid', name='classification_output')(tmp_class)
    concat_outputs = Concatenate(axis=-1)([prediction_output, classification_output])
    # concat_outputs = Lambda(lambda tt: tt[:, -50:, :], name='Slice_Output')(concat_outputs)
    # Define model with both outputs
    model = Model(inputs=inputs, outputs=concat_outputs)


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE']),
                    loss=custom_fbfce(horizon=hori_shift, loss_weight=loss_weight, params=params),
                    metrics=[custom_mse_metric, custom_binary_accuracy]
                  )
                    # metrics=[tf.keras.metrics.BinaryCrossentropy(),
                    #          tf.keras.metrics.BinaryAccuracy()])

    if params['WEIGHT_FILE']:
        print('load model')
        model.load_weights(params['WEIGHT_FILE'])

    return model

def build_DBI_multi_TCN_Horizon(input_timepoints, input_chans=8, params=None):

    # params
    use_batch_norm = params['TYPE_REG'].find('BN')>-1
    use_weight_norm = params['TYPE_REG'].find('WReg')>-1
    use_layer_norm = False
    use_l1_norm = False
    # this_activation = 'relu'
    # this_activation = tf.keras.activations.gelu
    this_activation = ELU(alpha=1)
    # this_activation = ELU(alpha=0.1)
    this_kernel_initializer = 'glorot_uniform'
    # this_kernel_initializer = 'he_normal'
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

    hori_shift = 0  # Default value
    if params['TYPE_ARCH'].find('Hori')>-1:
        print('Using Horizon Timesteps:')
        hori_shift = int(int(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Hori')+4:params['TYPE_ARCH'].find('Hori')+6])/1000*1250)
        print(hori_shift)

    if params['TYPE_ARCH'].find('Loss')>-1:
        print('Using Loss Weight:')
        loss_weight = (params['TYPE_ARCH'][params['TYPE_ARCH'].find('Loss')+4:params['TYPE_ARCH'].find('Loss')+7])
        weight = 1 if int(loss_weight[0])==1 else -1
        loss_weight = float(loss_weight[1])*10**(weight*float(loss_weight[2]))
        print(loss_weight)

    # load labels
    # inputs = Input(shape=(None, input_chans), name='inputs')
    inputs = Input(shape=(100, input_chans), name='inputs')

    if params['TYPE_ARCH'].find('CSD')>-1:
        csd_inputs = CSDLayer()(inputs)
        inputs_nets = Concatenate(axis=-1)([inputs, csd_inputs])
    else:
        inputs_nets = inputs

    if params['TYPE_ARCH'].find('ZNorm')>-1:
        print('Using ZNorm')
        inputs_nets = Normalization()(inputs_nets)

    # get TCN
    if model_type=='Base':
        from tcn import TCN
        tcn_op = TCN(nb_filters=n_filters,
                        kernel_size=n_kernels,
                        nb_stacks=1,
                        dilations=[n_dilations+2, n_dilations, n_dilations+4, 2 ], #[2 ** i for i in range(n_dilations)]
                        # dilations=(1, 2, 4, 8, 16), #, 32
                        padding='causal',
                        use_skip_connections=True,
                        dropout_rate=r_drop,
                        return_sequences=True,
                        activation=this_activation,
                        kernel_initializer=this_kernel_initializer,
                        # activation=this_activation,
                        # kernel_initializer=this_kernel_initializer,
                        use_batch_norm=False,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
        print(tcn_op.receptive_field)
        nets = tcn_op(inputs_nets)
        nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)

    tcn_output = nets
    output_dim = 8
    tmp_pred = Dense(32, activation=this_activation, name='tmp_pred')(tcn_output)  # Output future values
    prediction_output = Dense(output_dim, activation='linear', name='prediction_output')(tmp_pred)  # Output future values

    if params['TYPE_ARCH'].find('SelfPosAtt')>-1:
        print('Using Self Positional Attention')
        # compute csd of the predicted values as well
        pred_CSD = CSDLayer()(prediction_output)
        att_in = Concatenate(axis=-1)([pred_CSD, prediction_output])
        embed_dim, num_heads, num_channels = 16, 4, 50  # Adjust based on model structure
        attention_layer = SelfAttentionPositional(embed_dim=embed_dim, num_heads=num_heads, num_channels=num_channels)
        attentive_output = attention_layer(att_in)
        tcn_output = Concatenate(axis=-1)([tcn_output, attentive_output])
        # pdb.set_trace()

    elif params['TYPE_ARCH'].find('LayerAtt')>-1:
        print('Using LearnedAttention')
        pred_CSD = CSDLayer()(prediction_output)

        lfp_output = TwoStageAttentivePooling(embed_dim=8, num_heads=4, num_queries=50)(prediction_output)
        csd_output = TwoStageAttentivePooling(embed_dim=8, num_heads=4, num_queries=50)(pred_CSD)
        attentive_output = Concatenate(axis=-1)([lfp_output, csd_output])
        tcn_output = Concatenate(axis=-1)([tcn_output, attentive_output])

    # sigmoid out
    tmp_class = Dense(32, activation=this_activation, name='tmp_class')(tcn_output)

    # add confidence layer
    if params['TYPE_ARCH'].find('Confidence')>-1:
        print('Using Confidence Inputs')
        conf_inputs = Lambda(lambda tt: tt[:, -50:, :], name='Slice_Confidence')(inputs)
        confidence = tf.reduce_mean(tf.square(conf_inputs-prediction_output), axis=-1, keepdims=True)
        tmp_class = Concatenate(axis=-1)([tmp_class, confidence])

    # compute probability
    classification_output = Dense(1, activation='sigmoid', name='classification_output')(tmp_class)
    concat_outputs = Concatenate(axis=-1)([prediction_output, classification_output])
    # concat_outputs = Lambda(lambda tt: tt[:, -50:, :], name='Slice_Output')(concat_outputs)
    # Define model with both outputs
    model = Model(inputs=inputs, outputs=concat_outputs)


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE']),
                    loss=custom_fbfce(horizon=hori_shift, loss_weight=loss_weight, params=params),
                    metrics=[custom_mse_metric, custom_binary_accuracy]
                  )
                    # metrics=[tf.keras.metrics.BinaryCrossentropy(),
                    #          tf.keras.metrics.BinaryAccuracy()])

    if params['WEIGHT_FILE']:
        print('load model')
        model.load_weights(params['WEIGHT_FILE'])

    return model


def build_DBI_TCN_Dorizon(input_timepoints, input_chans=8, params=None):

    # params
    use_batch_norm = params['TYPE_REG'].find('BN')>-1
    use_weight_norm = params['TYPE_REG'].find('WReg')>-1
    use_layer_norm = False
    use_l1_norm = False
    # this_activation = 'relu'
    this_activation = ELU(alpha=1)
    # this_activation = tf.keras.activations.gelu
    # this_activation = ELU(alpha=0.1)
    this_kernel_initializer = 'glorot_uniform'
    # this_kernel_initializer = 'he_normal'
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

    if params['TYPE_ARCH'].find('Dori')>-1:
        print('Using Horizon Timesteps:')
        hori_shift = int(int(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Dori')+4:params['TYPE_ARCH'].find('Dori')+6])/1000*1250)
        print(hori_shift)

    if params['TYPE_ARCH'].find('Loss')>-1:
        print('Using Loss Weight:')
        loss_weight = (params['TYPE_ARCH'][params['TYPE_ARCH'].find('Loss')+4:params['TYPE_ARCH'].find('Loss')+7])
        weight = 1 if int(loss_weight[0])==1 else -1
        loss_weight = float(loss_weight[1])*10**(weight*float(loss_weight[2]))
        print(loss_weight)

    # load labels
    # inputs = Input(shape=(None, input_chans), name='inputs')
    inputs = Input(shape=(100, input_chans), name='inputs')

    if params['TYPE_ARCH'].find('CSD')>-1:
        csd_inputs = CSDLayer()(inputs)
        inputs_nets = Concatenate(axis=-1)([inputs, csd_inputs])
    else:
        inputs_nets = inputs
    # pdb.set_trace()
    if params['TYPE_ARCH'].find('ZNorm')>-1:
        print('Using ZNorm')
        inputs_nets = Normalization()(inputs_nets)

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
                        use_batch_norm=False,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
        print(tcn_op.receptive_field)
        # nets = tcn_op(inputs)
        tcn_clas = TCN(nb_filters=n_filters,
                        kernel_size=n_kernels,
                        nb_stacks=1,
                        dilations=[2 ** i for i in range(n_dilations)],
                        # dilations=(1, 2, 4, 8, 16), #, 32
                        padding='causal',
                        use_skip_connections=use_batch_norm,
                        dropout_rate=r_drop,
                        return_sequences=True,
                        activation=this_activation,
                        kernel_initializer=this_kernel_initializer,
                        # activation=this_activation,
                        # kernel_initializer=this_kernel_initializer,
                        use_batch_norm=False,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
        # horizon = 13
        # horizon_targets = Lambda(lambda tt: tt[:, -50:, :], name='Slice_Inputs')(inputs)
        nets = tcn_op(inputs_nets)
        output_dim = 8
        # tmp_pred = Dense(32, activation=this_activation, name='tmp_pred')(nets)  # Output future values
        prediction_output = Dense(output_dim, activation='linear', name='prediction_output')(nets)  # Output future values

        # pdb.set_trace()
        prediction_out_class = tcn_clas(prediction_output)
        if params['TYPE_ARCH'].find('Dual')>-1:
            print('Using DualLoss')
            out_class = tcn_clas(inputs)
            prediction_out_class = Concatenate(axis=-1)([prediction_out_class, out_class])

        # horizon_outputs = Lambda(lambda tt: tt[:, -input_timepoints+horizon:, :], name='Slice_Horizon')(nets)
        pred_out = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Pred_Output')(prediction_output)
        tcn_out = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Class_Output')(prediction_out_class)

        # tmp_class = Dense(32, activation=this_activation, name='tmp_class')(tcn_output)
        classification_output = Dense(1, activation='sigmoid', name='classification_output')(tcn_out)
        concat_outputs = Concatenate(axis=-1)([pred_out, classification_output])
        # concat_outputs = Concatenate(axis=-1)([prediction_output, classification_output])

    # Define model with both outputs
    model = Model(inputs=inputs, outputs=concat_outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE']),
                    loss=custom_fbfce(horizon=hori_shift, loss_weight=loss_weight, params=params),
                    metrics=[custom_mse_metric, custom_binary_accuracy]
                  )
                    # metrics=[tf.keras.metrics.BinaryCrossentropy(),
                    #          tf.keras.metrics.BinaryAccuracy()])

    if params['WEIGHT_FILE']:
        print('load model')
        model.load_weights(params['WEIGHT_FILE'])

    return model


def build_DBI_TCN_Corizon(input_timepoints, input_chans=8, params=None):

    # params
    use_batch_norm = params['TYPE_REG'].find('BN')>-1
    use_weight_norm = params['TYPE_REG'].find('WReg')>-1
    use_layer_norm = False
    use_l1_norm = False
    # this_activation = 'relu'
    this_activation = ELU(alpha=1)
    # this_activation = tf.keras.activations.gelu
    # this_activation = ELU(alpha=0.1)
    this_kernel_initializer = 'glorot_uniform'
    # this_kernel_initializer = 'he_normal'
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

    if params['TYPE_ARCH'].find('Cori')>-1:
        print('Using Horizon Timesteps:')
        hori_shift = int(int(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Cori')+4:params['TYPE_ARCH'].find('Cori')+6])/1000*1250)
        print(hori_shift)

    if params['TYPE_ARCH'].find('Loss')>-1:
        print('Using Loss Weight:')
        loss_weight = (params['TYPE_ARCH'][params['TYPE_ARCH'].find('Loss')+4:params['TYPE_ARCH'].find('Loss')+7])
        weight = 1 if int(loss_weight[0])==1 else -1
        loss_weight = float(loss_weight[1])*10**(weight*float(loss_weight[2]))
        print(loss_weight)

    # load labels
    # inputs = Input(shape=(None, input_chans), name='inputs')
    inputs = Input(shape=(100, input_chans), name='inputs')

    # if params['TYPE_ARCH'].find('CSD')>-1:
    csd_inputs = CSDLayer()(inputs)
    inputs_nets = Concatenate(axis=-1)([inputs, csd_inputs])
    # else:
    #     inputs_nets = inputs
    # pdb.set_trace()
    if params['TYPE_ARCH'].find('ZNorm')>-1:
        print('Using ZNorm')
        inputs_nets = Normalization()(inputs_nets)

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
                        use_batch_norm=False,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
        print(tcn_op.receptive_field)
        # nets = tcn_op(inputs)
        tcn_clas = TCN(nb_filters=n_filters,
                        kernel_size=n_kernels,
                        nb_stacks=1,
                        dilations=[2 ** i for i in range(n_dilations)],
                        # dilations=(1, 2, 4, 8, 16), #, 32
                        padding='causal',
                        use_skip_connections=use_batch_norm,
                        dropout_rate=r_drop,
                        return_sequences=True,
                        activation=this_activation,
                        kernel_initializer=this_kernel_initializer,
                        # activation=this_activation,
                        # kernel_initializer=this_kernel_initializer,
                        use_batch_norm=False,
                        use_layer_norm=use_layer_norm,
                        use_weight_norm=use_weight_norm,
                        go_backwards=False,
                        return_state=False)
        # horizon = 13
        # horizon_targets = Lambda(lambda tt: tt[:, -50:, :], name='Slice_Inputs')(inputs)
        nets = tcn_op(inputs_nets)
        output_dim = 8
        # tmp_pred = Dense(32, activation=this_activation, name='tmp_pred')(nets)  # Output future values
        prediction_output = Dense(output_dim, activation='linear', name='prediction_output')(nets)  # Output future values


        prediction_out_class = tcn_clas(prediction_output)
        in_out_class = tcn_clas(inputs)
        csd_out_class = tcn_clas(csd_inputs)
        class_concat = Concatenate(axis=-1)([in_out_class, csd_out_class, prediction_out_class])
        # horizon_outputs = Lambda(lambda tt: tt[:, -input_timepoints+horizon:, :], name='Slice_Horizon')(nets)
        pred_out = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Pred_Output')(prediction_output)
        tcn_out = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Class_Output')(class_concat)

        # tmp_class = Dense(32, activation=this_activation, name='tmp_class')(tcn_output)
        classification_output = Dense(1, activation='sigmoid', name='classification_output')(tcn_out)
        concat_outputs = Concatenate(axis=-1)([pred_out, classification_output])

    model = Model(inputs=inputs, outputs=concat_outputs)

            # mse_loss = tf.reduce_mean(tf.square(prediction_targets-prediction_out)) # multiply by labels

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE']),
                    loss=custom_fbfce(horizon=hori_shift, loss_weight=loss_weight, params=params),
                    metrics=[custom_mse_metric, custom_binary_accuracy]
                  )
                    # metrics=[tf.keras.metrics.BinaryCrossentropy(),
                    #          tf.keras.metrics.BinaryAccuracy()])

    if params['WEIGHT_FILE']:
        print('load model')
        model.load_weights(params['WEIGHT_FILE'])

    return model

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

    # import pdb
    # pdb.set_trace()
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

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.get('LEARNING_RATE', 1e-3))

    # Training Step with Binary Accuracy Monitoring
    @tf.function
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

    def train_model(train_dataset, val_dataset, params=None, save_best=True):
        epochs=params['NO_EPOCHS']
        # Track the best validation F1 score
        best_val_f1 = 0.0

        # Instantiate F1 metric outside of `evaluate`
        f1_metric = MaxF1Metric()
        val_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
        val_precision_metric = tf.keras.metrics.Precision()
        val_recall_metric = tf.keras.metrics.Recall()

        for epoch in range(epochs):
            # Initialize metrics for the current epoch
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy_avg = tf.keras.metrics.Mean()

            # Training loop
            for inputs, labels, weights in train_dataset:
                # Perform a training step
                loss, accuracy = train_step(inputs, labels, sample_weight=weights, params=params)
                epoch_loss_avg.update_state(loss)
                epoch_accuracy_avg.update_state(accuracy)

            # Run validation and get F1 score, accuracy, precision, and recall
            val_f1, val_accuracy, val_precision, val_recall = evaluate(val_dataset, f1_metric, val_accuracy_metric, val_precision_metric, val_recall_metric)

            # Print epoch results
            print(f"Epoch {epoch+1}, Loss: {epoch_loss_avg.result().numpy()}, "
            f"Accuracy: {epoch_accuracy_avg.result().numpy()}, "
            f"Validation F1 Score: {val_f1:.4f}, "
            f"Validation Accuracy: {val_accuracy:.4f}, "
            f"Validation Precision: {val_precision:.4f}, "
            f"Validation Recall: {val_recall:.4f}")

            # Save model if the F1 score improved
            if save_best and val_f1 > best_val_f1:
                best_val_f1 = val_f1
                model.save('best_f1_model.h5')
                print(f"New best model saved with F1 Score: {best_val_f1:.4f}")

            # Reset metrics for the next epoch
            epoch_loss_avg.reset_states()
            epoch_accuracy_avg.reset_states()
            f1_metric.reset_states()
            val_accuracy_metric.reset_states()
            val_precision_metric.reset_states()
            val_recall_metric.reset_states()

    def evaluate(val_dataset, f1_metric, val_accuracy_metric, val_precision_metric, val_recall_metric):
        # Reset metric state
        f1_metric.reset_states()
        val_accuracy_metric.reset_states()
        val_precision_metric.reset_states()
        val_recall_metric.reset_states()

        # List to store F1 scores at each threshold
        f1_scores = []

        # Iterate through validation dataset
        for inputs, labels in val_dataset:
            predictions = model(inputs, training=False)

            # Update metrics based on labels and predictions
            f1_score = f1_metric(labels, predictions)
            val_accuracy_metric.update_state(labels, predictions)
            val_precision_metric.update_state(labels, predictions)
            val_recall_metric.update_state(labels, predictions)
            f1_scores.append(f1_score)

        # Ensure `f1_scores` contains values before calculating max F1
        if f1_scores:
            max_f1_score = tf.reduce_max(tf.stack(f1_scores))
        else:
            max_f1_score = 0.0  # or other default value

        val_accuracy = val_accuracy_metric.result().numpy()
        val_precision = val_precision_metric.result().numpy()
        val_recall = val_recall_metric.result().numpy()

        return max_f1_score, val_accuracy, val_precision, val_recall


    return model, train_model


# Prototype Update Function
def update_prototypes(embeddings, labels, p_event, p_non_event):
    event_embeddings = tf.boolean_mask(embeddings, labels)
    non_event_embeddings = tf.boolean_mask(embeddings, 1 - labels)

    if tf.reduce_sum(tf.cast(labels, tf.float32)) > 0:
        p_event.assign(tf.reduce_mean(event_embeddings, axis=0))
    if tf.reduce_sum(tf.cast(1 - labels, tf.float32)) > 0:
        p_non_event.assign(tf.reduce_mean(non_event_embeddings, axis=0))


# def build_DBI_TCN_Horizon_Updated(input_timepoints, input_chans=8, embedding_dim=32, params=None):
#     from tcn import TCN
#     inputs = Input(shape=(input_timepoints*2, input_chans), name='inputs')

#     # TCN Backbone
#     tcn_layer = TCN(
#         nb_filters=params['NO_FILTERS'],
#         kernel_size=params['NO_KERNELS'],
#         dilations=[2 ** i for i in range(params['NO_DILATIONS'])],
#         return_sequences=True,
#         use_skip_connections=True,
#         dropout_rate=params.get('DROPOUT', 0.0),
#         activation='relu',
#         # use_batch_norm=True,
#         use_weight_norm=True,
#         padding='causal'
#     )

#     tcn_output = tcn_layer(inputs)
#     tcn_output = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Pred_Output')(tcn_output)

#     # Embedding Layer for Prototype and Barlow Twins Losses
#     embedding_layer = Dense(embedding_dim, activation='linear', name='embedding')
#     embeddings = embedding_layer(tcn_output)

#     # Learnable Prototypes for events and non-events
#     p_event = tf.Variable(tf.random.normal([embedding_dim]), trainable=True, name="p_event")
#     p_non_event = tf.Variable(tf.random.normal([embedding_dim]), trainable=True, name="p_non_event")

#     # Prediction Head for Classification
#     final_output = Dense(1, activation='sigmoid', name='classification_output')(embeddings)

#     # Build Model
#     model = Model(inputs=inputs, outputs=final_output)

#     # Prototype-Based Loss
#     def prototype_loss(embeddings, labels):
#         d_event = tf.norm(embeddings - p_event, axis=-1)
#         d_non_event = tf.norm(embeddings - p_non_event, axis=-1)
#         return tf.reduce_mean(labels * d_event + (1 - labels) * d_non_event)

#     # Barlow Twins Loss
#     def barlow_twins_loss(embeddings_A, embeddings_B, lambda_param=0.005):
#         embeddings_A = (embeddings_A - tf.reduce_mean(embeddings_A, axis=0)) / tf.math.reduce_std(embeddings_A, axis=0)
#         embeddings_B = (embeddings_B - tf.reduce_mean(embeddings_B, axis=0)) / tf.math.reduce_std(embeddings_B, axis=0)
#         N = tf.shape(embeddings_A)[0]
#         cross_corr = tf.matmul(tf.transpose(embeddings_A), embeddings_B) / tf.cast(N, tf.float32)
#         on_diag = tf.reduce_sum(tf.square(tf.linalg.diag_part(cross_corr) - 1))
#         off_diag = tf.reduce_sum(tf.square(cross_corr) - tf.square(tf.linalg.diag_part(cross_corr)))
#         return on_diag + lambda_param * off_diag

#     def augment_data(inputs, jitter_strength=0.05, noise_strength=0.02, max_shift=5):
#         """
#         Apply data augmentation to generate slightly different views of the input data.

#         Args:
#         - inputs: Input tensor of shape [batch, time_steps, channels].
#         - jitter_strength: Strength of jitter (fraction of values).
#         - noise_strength: Standard deviation of Gaussian noise added.
#         - max_shift: Maximum shift in time steps for time shifting.

#         Returns:
#         - Augmented tensor of the same shape as inputs.
#         """

#         # 1. Jitter (slight random variation in values)
#         jitter = tf.random.uniform(tf.shape(inputs), -jitter_strength, jitter_strength)
#         inputs_jittered = inputs + jitter

#         # 2. Gaussian Noise
#         noise = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=noise_strength)
#         inputs_noisy = inputs_jittered + noise

#         # 3. Time Shifting (shifting along the time dimension)
#         time_steps = tf.shape(inputs)[1]
#         shift = tf.random.uniform([], -max_shift, max_shift, dtype=tf.int32)
#         inputs_shifted = tf.roll(inputs_noisy, shift=shift, axis=1)

#         return inputs_shifted

#     # Training Step with Dynamic Loss Weighting
#     optimizer = tf.keras.optimizers.Adam(learning_rate=params.get('LEARNING_RATE', 1e-3))
#     alpha = params.get('BARLOW_TWIN_WEIGHT', 0.1)

#     @tf.function
#     def train_step(x, y):
#         # import pdb
#         y = tf.squeeze(y)
#         # pdb.set_trace()
#         with tf.GradientTape() as tape:
#             # Forward pass
#             embeddings = model(x, training=True)

#             # Prototype-based loss
#             proto_loss = prototype_loss(embeddings, y)


#             # Generate augmented views and compute Barlow Twins loss
#             # x_aug1 = augment_data(x)
#             # x_aug2 = augment_data(x)
#             # embeddings_A = model(x_aug1, training=True)
#             # embeddings_B = model(x_aug2, training=True)
#             # barlow_loss = barlow_twins_loss(embeddings_A, embeddings_B)
#             barlow_loss = 0
#             # Combine losses
#             total_loss = proto_loss + alpha * barlow_loss

#         gradients = tape.gradient(total_loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#         return total_loss

#     # Prototype Update Function
#     def update_prototypes(embeddings, labels):
#         """Update prototypes based on mean embeddings of event and non-event classes."""
#         event_embeddings = tf.boolean_mask(embeddings, labels > 0.5)
#         non_event_embeddings = tf.boolean_mask(embeddings, labels <= 0.5)

#         # Update prototypes to the mean of the respective classes
#         if tf.size(event_embeddings) > 0:
#             p_event.assign(tf.reduce_mean(event_embeddings, axis=0))
#         else:
#             p_event.assign(tf.zeros_like(p_event))

#         if tf.size(non_event_embeddings) > 0:
#             p_non_event.assign(tf.reduce_mean(non_event_embeddings, axis=0))
#         else:
#             p_non_event.assign(tf.zeros_like(p_non_event))

#     # Training Loop with Prototype Update
#     def train_model(train_dataset, epochs=50):
#         for epoch in range(epochs):
#             epoch_loss_avg = tf.keras.metrics.Mean()
#             all_embeddings = []
#             all_labels = []
#             n = 0
#             for inputs, labels, weights in train_dataset:
#                 n += 1
#                 print(f"Batch {n}")
#                 # import pdb
#                 # pdb.set_trace()
#                 loss = train_step(inputs, labels)
#                 epoch_loss_avg.update_state(loss)

#                 # # Collect embeddings and labels for prototype update
#                 # all_embeddings.append(model(inputs, training=False))
#                 # all_labels.append(labels)
#                 if n == 10:
#                     break

#             # # Update prototypes after each epoch
#             # all_embeddings = tf.concat(all_embeddings, axis=0)
#             # all_labels = tf.concat(all_labels, axis=0)
#             # update_prototypes(all_embeddings, all_labels)

#             print(f"Epoch {epoch+1}, Loss: {epoch_loss_avg.result().numpy()}")

#     return model, train_model

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

def custom_mse_metric(y_true, y_pred):
    # Calculate MSE for the first 8 channels
    prediction_out = y_pred[:, :, :8]
    y_true_exp = y_true[:, :, :8]
    mse_metric = tf.reduce_mean(tf.square(prediction_out - y_true_exp))
    return mse_metric

def custom_binary_accuracy(y_true, y_pred):
    # Calculate Binary Accuracy for the last channel
    y_true_exp = tf.expand_dims(y_true[:, :, -1], axis=-1)
    y_pred_exp = tf.expand_dims(y_pred[:, :, -1], axis=-1)
    return tf.keras.metrics.binary_accuracy(y_true_exp, y_pred_exp)

def custom_l1_regularization_loss(model, l1_lambda=1e-5):
    # Calculate the L1 loss on all trainable weights
    l1_loss = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in model.trainable_weights])
    return l1_lambda * l1_loss

def custom_l2_regularization_loss(model, l2_lambda=1e-5):
    # Calculate the L2 loss on all trainable weights
    l2_loss = tf.add_n([tf.reduce_sum(tf.square(w)) for w in model.trainable_weights])
    return l2_lambda * l2_loss

def early_onset_preference_loss(y_true, y_pred, threshold=0.5, early_onset_threshold=5, penalty_factor=0.1, reward_factor=0.05):

    # Reshape for compatibility if needed
    y_true_sq = tf.squeeze(y_true, axis=-1)
    y_pred_sq = tf.squeeze(y_pred, axis=-1)

    # Calculate delay in detection
    detected_times = tf.argmax(tf.cast(y_pred_sq >= threshold, tf.int32), axis=1)
    true_event_times = tf.argmax(tf.cast(y_true_sq, tf.int32), axis=1)
    delay = detected_times - true_event_times

    # Penalty for late detections
    late_penalty = tf.where(delay > early_onset_threshold, penalty_factor * tf.cast(delay - early_onset_threshold, tf.float32), 0.0)

    # Reward for slightly early detections
    early_reward = tf.where((delay < 0) & (delay >= -early_onset_threshold), reward_factor * tf.cast(-delay, tf.float32), 0.0)

    # Combine BCE loss with penalty and reward terms
    onset_loss = late_penalty - early_reward

    return tf.reduce_mean(onset_loss)


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

def threshold_consistent_margin_loss(y_true, y_pred):
    margin = 0.2  # adjust the margin value as needed
    positive_loss = K.maximum(0.0, y_pred - y_true + margin)
    negative_loss = K.maximum(0.0, y_true - y_pred + margin)
    total_loss = K.mean(positive_loss + negative_loss)
    return total_loss

def tcml_loss(y_true, y_pred, threshold=0.5, margin=0.2):
    positive_mask = tf.equal(y_true, 1.0)
    negative_mask = tf.equal(y_true, 0.0)

    pos_loss = tf.maximum(0.0, threshold + margin - y_pred)
    neg_loss = tf.maximum(0.0, y_pred - (threshold - margin))

    pos_loss = tf.where(positive_mask, pos_loss, 0.0)
    neg_loss = tf.where(negative_mask, neg_loss, 0.0)

    loss = tf.reduce_mean(pos_loss + neg_loss)
    return loss

def cosine_similarity(a, b):
    dot_product = tf.reduce_sum(a * b, axis=-1)
    norm_a = tf.norm(a, axis=-1)
    norm_b = tf.norm(b, axis=-1)
    return dot_product / (norm_a * norm_b)

def tcm_loss(y_true, y_pred, m_pos=0.5, m_neg=0.5, lambda_pos=1.0, lambda_neg=1.0):
    y_true = tf.cast(y_true, tf.float32)

    # Calculate cosine similarity scores
    cos_sim = cosine_similarity(y_true, y_pred)

    # Identify positive and negative samples
    pos_mask = tf.equal(y_true, 1.0)
    neg_mask = tf.equal(y_true, 0.0)

    # Compute positive and negative losses
    pos_loss = lambda_pos * tf.reduce_sum((m_pos - cos_sim) * tf.cast(cos_sim <= m_pos, tf.float32)) / tf.reduce_sum(tf.cast(pos_mask, tf.float32))
    neg_loss = lambda_neg * tf.reduce_sum((cos_sim - m_neg) * tf.cast(cos_sim >= m_neg, tf.float32)) / tf.reduce_sum(tf.cast(neg_mask, tf.float32))

    # Sum the positive and negative losses
    tcm_loss_value = pos_loss + neg_loss

    return tcm_loss_value

def adaptive_early_onset_loss(y_true, y_pred, threshold=0.5, onset_confidence=3):
    early_loss = early_onset_preference_loss(y_true, y_pred, threshold=threshold)
    dynamic_threshold = tf.Variable(threshold)

    if onset_confidence >= 3:
        dynamic_threshold.assign(dynamic_threshold * 1.05)
    elif onset_confidence < 3:
        dynamic_threshold.assign(tf.maximum(dynamic_threshold * 0.95, threshold))

    combined_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + early_loss
    return combined_loss
import tensorflow as tf

class AdaptiveEventDetector(tf.keras.layers.Layer):
    def __init__(self, initial_confidence_threshold=0.5, mse_threshold=0.1, moving_avg_window=10, past_window=20):
        super(AdaptiveEventDetector, self).__init__()
        self.initial_confidence_threshold = initial_confidence_threshold
        self.mse_threshold = mse_threshold
        self.moving_avg_window = moving_avg_window
        self.past_window = past_window

    def call(self, inputs, labels):
        """
        inputs: Tensor of shape [batch, time_steps] representing probability predictions.
        labels: Binary ground truth labels [batch, time_steps] for tuning thresholds.
        """
        # Step 1: Moving average and dynamic baseline
        moving_avg = tf.keras.layers.AveragePooling1D(pool_size=self.moving_avg_window, strides=1, padding='same')(inputs)
        past_avg = tf.keras.layers.AveragePooling1D(pool_size=self.past_window, strides=1, padding='same')(inputs)

        # Step 2: Adaptive Threshold Based on Confidence
        detection = tf.where(inputs > (past_avg + self.initial_confidence_threshold), 1.0, 0.0)

        # Calculate slope as trend-based confidence score
        slope = tf.math.reduce_mean(tf.math.abs(inputs[:, 1:] - inputs[:, :-1]), axis=1, keepdims=True)

        # Step 3: Confidence Verification Phase
        confidence = tf.where(slope > self.mse_threshold, 1.0, 0.0)
        combined_detection = tf.where(detection * confidence > 0, 1.0, 0.0)

        # Step 4: Dynamic Adjustment based on ground truth labels
        def update_thresholds(detections, labels):
            tp = tf.reduce_sum(detections * labels)
            fn = tf.reduce_sum((1 - detections) * labels)
            fp = tf.reduce_sum(detections * (1 - labels))

            # Adjust threshold based on missed detections
            adj_factor = tf.cond(tp / (tp + fn + 1e-5) < 0.8, lambda: 0.05, lambda: -0.02)  # Dynamic adj. factor
            self.initial_confidence_threshold += adj_factor
            self.mse_threshold += adj_factor * 0.5  # Adjust MSE if too sensitive to noise

        # Apply adjustments to thresholds
        tf.py_function(update_thresholds, [combined_detection, labels], Tout=[])

        # Return detected events
        return combined_detection

class MaxF1Metric(tf.keras.metrics.Metric):
    def __init__(self, thresholds=tf.linspace(0.0, 1.0, 11), **kwargs):
        super().__init__(**kwargs)
        self.thresholds = thresholds
        self.max_f1 = self.add_weight(name="max_f1", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Initialize F1 scores list for different thresholds
        f1_scores = []

        for threshold in self.thresholds:
            # Apply threshold to predictions
            y_pred_thresh = tf.cast(y_pred >= threshold, tf.float32)

            # Calculate TP, FP, FN
            tp = tf.reduce_sum(y_pred_thresh * y_true)
            fp = tf.reduce_sum(y_pred_thresh * (1 - y_true))
            fn = tf.reduce_sum((1 - y_pred_thresh) * y_true)

            # Calculate precision and recall
            precision = tp / (tp + fp + tf.keras.backend.epsilon())
            recall = tp / (tp + fn + tf.keras.backend.epsilon())

            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
            f1_scores.append(f1)

        # Update max F1 score
        self.max_f1.assign(tf.reduce_mean(f1_scores))

    def result(self):
        # Return the maximum F1 score
        return self.max_f1

    def reset_states(self):
        # Reset the max F1 score to zero at the beginning of each epoch
        self.max_f1.assign(0.0)

class SelfAttentionPositional(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, num_channels):
        super(SelfAttentionPositional, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.positional_bias = self.add_weight(shape=(1, num_channels, embed_dim), initializer="zeros", trainable=True)
        self.norm = LayerNormalization()

    def call(self, inputs):
        inputs += self.positional_bias  # Add positional bias for each channel
        attention_output = self.attention(inputs, inputs)
        return self.norm(attention_output + inputs)

class TwoStageAttentivePooling(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, num_queries=1):
        super(TwoStageAttentivePooling, self).__init__()
        self.deep_query = self.add_weight(shape=(1, num_queries, embed_dim), initializer="random_normal", trainable=True)
        self.superficial_query = self.add_weight(shape=(1, num_queries, embed_dim), initializer="random_normal", trainable=True)
        self.deep_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.superficial_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.norm = LayerNormalization()

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Replicate queries for each batch
        deep_query = tf.tile(self.deep_query, [batch_size, 1, 1])
        superficial_query = tf.tile(self.superficial_query, [batch_size, 1, 1])

        # Separate attention heads for deep and superficial layers
        deep_attention_output = self.deep_attention(deep_query, inputs[:, :, :4])  # Assuming first 4 channels are deep
        superficial_attention_output = self.superficial_attention(superficial_query, inputs[:, :, 4:])  # Last 4 channels superficial

        # Concatenate and normalize
        combined_attention = self.concat([deep_attention_output, superficial_attention_output])
        return self.norm(combined_attention)


def custom_fbfce(loss_weight=1, horizon=0, params=None):
    def loss_fn(y_true, y_pred, loss_weight=loss_weight, horizon=horizon):
        # print(sample_weight.shape)
        prediction_targets = y_true[:, horizon:, :8]  # Targets starting from horizon
        prediction_out = y_pred[:, :-horizon, :8]  # Predicted outputs before the horizon
        y_true_exp = tf.expand_dims(y_true[:, :, -1], axis=-1)
        y_pred_exp = tf.expand_dims(y_pred[:, :, -1], axis=-1)  # First 8 channels for horizon prediction

        # y_pred = classification_output
        if params['TYPE_LOSS'].find('FocalSmooth')>-1:
            print('FocalSmooth')
            total_loss = tf.keras.losses.binary_focal_crossentropy(y_true_exp, y_pred_exp, apply_class_balancing=True, label_smoothing=0.1)
        elif params['TYPE_LOSS'].find('FocalSmoothless')>-1:
            print('FocalRegular')
            total_loss = tf.keras.losses.binary_focal_crossentropy(y_true_exp, y_pred_exp, apply_class_balancing=True)
        elif params['TYPE_LOSS'].find('Focal')>-1:
            print('Focal')
            aind = params['TYPE_LOSS'].find('Ax')+2
            alp = float(params['TYPE_LOSS'][aind:aind+3])/100
            gind = params['TYPE_LOSS'].find('Gx')+2
            gam = float(params['TYPE_LOSS'][gind:gind+3])/100
            print('Alpha: {0}, Gamma: {1}'.format(alp, gam))
            if alp == 0:
                total_loss = tf.keras.losses.binary_focal_crossentropy(y_true_exp, y_pred_exp, gamma = gam)
            else:
                total_loss = tf.keras.losses.binary_focal_crossentropy(y_true_exp, y_pred_exp, apply_class_balancing=True, alpha = alp, gamma = gam)
        else:
            pdb.set_trace()

        # extra losses that can also be combined
        if params['TYPE_LOSS'].find('TV')>-1:
            print('TV Loss')
            # total_loss += 1e-5*tf.image.total_variation(y_pred)
            total_loss += 1e-5*tf.reduce_sum(tf.image.total_variation(tf.expand_dims(y_pred_exp, axis=-1)))
        if params['TYPE_LOSS'].find('L2')>-1:
            print('L2 smoothness Loss')
            total_loss += 1e-5*tf.reduce_mean((y_pred_exp[1:]-y_pred_exp[:-1])**2)
        if params['TYPE_LOSS'].find('Margin')>-1:
            print('Margin Loss')
            # pdb.set_trace()
            total_loss += 1e-3*tf.squeeze(y_pred_exp * (1 - y_pred_exp))
        if params['TYPE_LOSS'].find('TMSE')>-1:
            print('Truncated MSE Loss')
            total_loss += 1e-1*truncated_mse_loss(y_true_exp, y_pred_exp, tau=4.0)
        if params['TYPE_LOSS'].find('EarlyOnset')>-1:
            print('Early Onset Preference Loss')
            total_loss += 1e-3*early_onset_preference_loss(y_true_exp, y_pred_exp, threshold=0.5, early_onset_threshold=5, penalty_factor=0.1, reward_factor=0.05)
        if params['TYPE_LOSS'].find('L1Reg')>-1:
            print('L1 Regularization Loss')
            total_loss += custom_l1_regularization_loss(model, l1_lambda=1e-3)
        if params['TYPE_LOSS'].find('L2Reg')>-1:
            print('L2 Regularization Loss')
            total_loss += custom_l2_regularization_loss(model, l1_lambda=1e-3)
        # mse_loss = tf.reduce_mean(tf.square(prediction_targets-prediction_out)) # multiply by labels
        # mse_loss = tf.reduce_mean(y_true_exp[:,horizon:,:]*tf.square(prediction_targets-prediction_out)) # multiply by labels

        print('rec_weights')
        rec_weights = y_true_exp[:,horizon:,:] + tf.keras.backend.epsilon()
        mse_loss = tf.reduce_mean(rec_weights*tf.square(prediction_targets-prediction_out)) # multiply by labels
        # mse_loss = combined_prediction_loss(prediction_targets, prediction_out, y_true_exp, event_weight=loss_weight)
        total_loss += loss_weight*mse_loss
        return total_loss
    return loss_fn



# def build_DBI_TCN_CSD(input_timepoints, input_chans=8, params=None):

#     # params
#     use_batch_norm = False
#     use_weight_norm = True
#     use_layer_norm = False
#     use_l1_norm = False
#     # this_activation = 'relu'
#     this_activation = ELU(alpha=1)
#     # this_activation = ELU(alpha=0.1)
#     # this_kernel_initializer = 'glorot_uniform'
#     this_kernel_initializer = 'he_normal'
#     model_type = params['TYPE_MODEL']
#     n_filters = params['NO_FILTERS']
#     n_kernels = params['NO_KERNELS']
#     n_dilations = params['NO_DILATIONS']
#     if params['TYPE_ARCH'].find('Drop')>-1:
#         r_drop = float(params['TYPE_ARCH'][params['TYPE_ARCH'].find('Drop')+4:params['TYPE_ARCH'].find('Drop')+6])/100
#         print('Using Dropout')
#         print(r_drop)
#     else:
#         r_drop = 0.0

#     # load labels
#     inputs = Input(shape=(None, input_chans), name='inputs')

#     csd_inputs = CSDLayer()(inputs)

#     # get TCN
#     if model_type=='Base':
#         from tcn import TCN
#         tcn_op = TCN(nb_filters=n_filters,
#                         kernel_size=n_kernels,
#                         nb_stacks=1,
#                         dilations=[2 ** i for i in range(n_dilations)],
#                         # dilations=(1, 2, 4, 8, 16), #, 32
#                         padding='causal',
#                         use_skip_connections=True,
#                         dropout_rate=r_drop,
#                         return_sequences=True,
#                         activation=this_activation,
#                         kernel_initializer=this_kernel_initializer,
#                         # activation=this_activation,
#                         # kernel_initializer=this_kernel_initializer,
#                         use_batch_norm=False,
#                         use_layer_norm=use_layer_norm,
#                         use_weight_norm=use_weight_norm,
#                         go_backwards=False,
#                         return_state=False)
#         print(tcn_op.receptive_field)
#         tcn_csd = TCN(nb_filters=n_filters,
#                         kernel_size=n_kernels,
#                         nb_stacks=1,
#                         dilations=[2 ** i for i in range(n_dilations)],
#                         # dilations=(1, 2, 4, 8, 16), #, 32
#                         padding='causal',
#                         use_skip_connections=True,
#                         dropout_rate=r_drop,
#                         return_sequences=True,
#                         activation=this_activation,
#                         kernel_initializer=this_kernel_initializer,
#                         # activation=this_activation,
#                         # kernel_initializer=this_kernel_initializer,
#                         use_batch_norm=False,
#                         use_layer_norm=use_layer_norm,
#                         use_weight_norm=use_weight_norm,
#                         go_backwards=False,
#                         return_state=False)
#         nets = tcn_op(inputs)
#         nets_csd = tcn_csd(csd_inputs)

#         nets = Concatenate(axis=-1)([nets, nets_csd])

#         # Smooth Conv
#         wconv = Conv1D(filters=n_filters,
#                 kernel_size=3,
#                 dilation_rate=1,
#                 padding='causal',
#                 activation=this_activation,
#                 use_bias = True,
#                 bias_initializer='zeros',
#                 name='deconv1_{0}'.format(1),
#                 kernel_initializer=this_kernel_initializer
#                 )

#         # Smooth Conv
#         wconv2 = Conv1D(filters=n_filters/2,
#                 kernel_size=3,
#                 dilation_rate=1,
#                 padding='causal',
#                 activation=this_activation,
#                 use_bias = True,
#                 bias_initializer='zeros',
#                 name='deconv2_{0}'.format(1),
#                 kernel_initializer=this_kernel_initializer
#                 )

#         # Slice & Out
#         if params['mode'] == 'train':
#             nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)
#         else:
#             nets = Lambda(lambda tt: tt[:, -1:, :], name='Slice_Output')(nets)

#         if use_weight_norm:
#             wconv = WeightNormalization(wconv)
#             wconv2 = WeightNormalization(wconv2)
#         nets = wconv(nets)
#         nets = wconv2(nets)

#         # sigmoid out
#         nets = Dense(1, activation='sigmoid')(nets)

#     # get outputs
#     outputs = nets
#     model = Model(inputs=[inputs], outputs=[outputs])

#     def custom_fbfce(weights=None):
#         """Loss function"""
#         def loss_fn(y_true, y_pred, weights=weights):
#             if params['TYPE_LOSS'].find('FocalSmooth')>-1:
#                 print('FocalSmooth')
#                 total_loss = tf.keras.losses.binary_focal_crossentropy(y_true, y_pred, apply_class_balancing=True, label_smoothing=0.1)
#             elif params['TYPE_LOSS'].find('Focal')>-1:
#                 print('Focal')
#                 aind = params['TYPE_LOSS'].find('Ax')+2
#                 alp = float(params['TYPE_LOSS'][aind:aind+3])/100
#                 gind = params['TYPE_LOSS'].find('Gx')+2
#                 gam = float(params['TYPE_LOSS'][gind:gind+3])/100
#                 print('Alpha: {0}, Gamma: {1}'.format(alp, gam))
#                 if alp == 0:
#                     total_loss = tf.keras.losses.binary_focal_crossentropy(y_true, y_pred, gamma = gam)
#                 else:
#                     total_loss = tf.keras.losses.binary_focal_crossentropy(y_true, y_pred, apply_class_balancing=True, alpha = alp, gamma = gam)
#             elif params['TYPE_LOSS'].find('Anchor')>-1:
#                 print('Anchor')
#                 aind = params['TYPE_LOSS'].find('Ax')+2
#                 alp = float(params['TYPE_LOSS'][aind:aind+3])/100
#                 gind = params['TYPE_LOSS'].find('Gx')+2
#                 gam = float(params['TYPE_LOSS'][gind:gind+3])/100
#                 print('Alpha: {0}, Gamma: {1}'.format(alp, gam))
#                 if alp == 0:
#                     total_loss = tf.keras.losses.binary_focal_crossentropy(y_true, y_pred, gamma = gam)
#                 else:
#                     total_loss = tf.keras.losses.binary_focal_crossentropy(y_true, y_pred, apply_class_balancing=True, alpha = alp, gamma = gam)
#             else:
#                 pdb.set_trace()
#             if params['TYPE_LOSS'].find('TV')>-1:
#                 print('TV Loss')
#                 # total_loss += 1e-5*tf.image.total_variation(y_pred)
#                 total_loss += 1e-5*tf.reduce_sum(tf.image.total_variation(tf.expand_dims(y_pred, axis=-1)))
#             if params['TYPE_LOSS'].find('L2')>-1:
#                 print('L2 smoothness Loss')
#                 total_loss += 1e-5*tf.reduce_mean((y_pred[1:]-y_pred[:-1])**2)
#             if params['TYPE_LOSS'].find('Margin')>-1:
#                 print('Margin Loss')
#                 # pdb.set_trace()
#                 total_loss += 1e-4*tf.squeeze(y_pred * (1 - y_pred))
#             return total_loss
#         return loss_fn

#     # model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=params['LEARNING_RATE']),
#     # model.trainable = True
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE']),
#                     loss=custom_fbfce(),
#                     metrics=[tf.keras.metrics.BinaryCrossentropy(),
#                              tf.keras.metrics.BinaryAccuracy()])

#     if params['WEIGHT_FILE']:
#         print('load model')
#         model.load_weights(params['WEIGHT_FILE'])

#     return model
