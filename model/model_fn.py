from tensorflow.keras.layers import Conv1D, Conv2D, ELU, Input, LSTM, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, Concatenate, Lambda
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import applications
import tensorflow_addons as tfa
import tensorflow as tf
import pdb

def get_vgg(nrows, ncols, out_layer=None, is_trainable=False, inc_top=False):
    vgg_model = applications.VGG16(
        weights="imagenet",
        include_top=inc_top,
        input_shape=(nrows, ncols, 3))

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
    # Getting output tensor of the last VGG layer that we want to include
    #vgg_model.outputs = [layer_dict[out].output for out in out_layer]
    for out in layer_dict.keys():
        # print('hello')
        layer_dict[out].trainable = is_trainable

    if not out_layer:
        outputs = [layer_dict[out].output for out in layer_dict.keys()]
        outputs = outputs[1:]
    else:
        outputs = [layer_dict[out].output for out in out_layer]
    # Create model and compile
    model = Model([vgg_model.input], outputs)
    model.trainable = is_trainable
    model.compile(loss='mse', optimizer='adam')
    return model

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

def build_DBI_TCN(input_timepoints, input_chans=8, params=None):
    from tcn import TCN

    use_batch_norm = False
    use_weight_norm = True
    use_l1_norm = False
    n_filters = 64
    n_kernels = 7
    n_dilations = 4
    # load labels
    # inputs = Input(shape=(None, input_chans))    
    # inputs = Input(shape=(None, input_chans), name='inputs')
    # inputs = Input(shape=(input_timepoints*2, input_chans), name='inputs')
    inputs = Input(shape=(None, input_chans), name='inputs')
    # weights = Input(shape=(), name='weights')
    # nets = inputs

    # get TCN
    baseTCN = False
    if baseTCN:
        nets = TCN(nb_filters=n_filters,
                    kernel_size=n_kernels,
                    nb_stacks=1,
                    dilations=[2 ** i for i in range(n_dilations)],
                    # dilations=(1, 2, 4, 8, 16), #, 32
                    padding='causal',
                    use_skip_connections=True,
                    dropout_rate=0.0,
                    return_sequences=True,
                    activation='relu',
                    kernel_initializer='glorot_uniform',
                    use_batch_norm=False,
                    use_layer_norm=False,
                    use_weight_norm=True,
                    go_backwards=False,
                    return_state=False)(inputs)
        # pdb.set_trace()
        # nets = Dense(1, kernel_initializer=GlorotUniform(), use_bias=True)(nets)
        nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)
        nets = Dense(1, activation='sigmoid', kernel_initializer=GlorotUniform())(nets)
        # nets = Activation('sigmoid')(nets)
        # nets = Flatten()(nets)
        # pdb.set_trace()
    else:
        nets = []
        for i_ch in range(8):
            nets_in = Lambda(lambda tt: tt[:, :, i_ch], name='Slice_Input_{}'.format(i_ch))(inputs)
            nets.append(tf.expand_dims(nets_in, axis=-1))

        # pdb.set_trace()
        dropout_rate = 0
        all_nets = []
        for i_dilate in range(n_dilations):
            wconv = Conv1D(filters=n_filters,
                            kernel_size=n_kernels,
                            dilation_rate=2**(i_dilate),
                            # groups=input_chans,
                            padding='causal',
                            activation='relu',
                            use_bias = True,
                            bias_initializer='zeros',
                            # kernel_regularizer=l1(0.001),
                            # activation=ELU(alpha=0),
                            name='dconv1_{0}'.format(i_dilate),
                            kernel_initializer='glorot_uniform'
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
                        activation='relu',
                        # kernel_regularizer=l1(0.001),
                        # activation=ELU(alpha=0),
                        name='de_dconv1_{0}'.format(0),
                        # kernel_initializer='he_normal'
                        kernel_initializer='glorot_uniform'
                        )
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
                        activation='relu',
                        # kernel_regularizer=l1(0.001),
                        # activation=ELU(alpha=0),
                        name='de_dconv1_{0}'.format(1),
                        kernel_initializer='glorot_uniform'
                        # kernel_initializer='he_normal'
                        )
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
        
        # concat
        nets = Concatenate(axis=-1)(all_chs)
        nets = Lambda(lambda tt: tt[:, -input_timepoints:, :], name='Slice_Output')(nets)
        # pdb.set_trace()
        # nets = tf.expand_dims(nets, axis=1)

        # reduce mean and get output
        # nets = tf.reduce_mean(nets, axis=-1, keepdims=True)
        
        nets = Dense(1, use_bias = True,
                        # bias_initializer=Constant(value=-5.0),
                        bias_initializer='zeros',
                        activation='sigmoid', 
                        kernel_initializer=GlorotUniform())(nets)
    # pdb.set_trace()
    # get outputs
    outputs = nets
    model = Model(inputs=[inputs], outputs=[outputs])
    if params['WEIGHT_FILE']:
        print('load model')
        model.load_weights(params['WEIGHT_FILE'])

    
    def custom_fbfce(weights=None):#y_true, y_pred
        """Loss function"""
        # return tf.keras.losses.BinaryFocalCrossentropy(y_true, y_pred)#, apply_class_balancing=True
        def loss_fn(y_true, y_pred, weights=weights):
            # tf.keras.losses.binary_focal_crossentropy
            # tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True)
            y_true = tf.expand_dims(y_true, axis=-1)
            y_pred = tf.expand_dims(y_pred, axis=-1)
            loss_function = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, reduction='none')

            # gamma = 2.0
            # alpha = 0.25
            
            # Compute the focal loss
            # pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            # focal_loss = -alpha * (1 - pt) ** gamma * tf.math.log(pt + tf.keras.backend.epsilon())            
            total_loss = loss_function(y_true, y_pred)
            # pdb.set_trace()
            # total_loss = loss_function(y_true, y_pred, sample_weight=weights)
            # total_loss = tf.keras.losses.binary_focal_crossentropy(y_true, y_pred, apply_class_balancing=True)
            # pdb.set_trace()
            # weighted_loss = tf.reduce_mean(focal_loss, axis=-1)
            weighted_loss = loss_function
            # weighted_loss = tf.multiply(focal_loss, weights)
            return weighted_loss
            # return tf.keras.losses.binary_focal_crossentropy(y_true, y_pred)#, apply_class_balancing=True
        return loss_fn
    
    # pdb.set_trace()
    # model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=params['LEARNING_RATE']),
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['LEARNING_RATE']),
                    # loss=custom_fbfce(weights=weights),
                    loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, label_smoothing=0.1), 
                    # loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True), 
                    metrics=[tf.keras.metrics.BinaryCrossentropy(), 
                             tf.keras.metrics.BinaryAccuracy()])
    model.trainable = True
                    # tfa.metrics.F1Score(num_classes=2, threshold=0.5)
                    # , sample_weight=[1-params['RIPPLE_RATIO'], params['RIPPLE_RATIO']])
                    #   metrics='sparse_categorical_crossentropy') 'binary_focal_crossentropy'
                    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    # metrics=['sparse_categorical_accuracy'])
    return model

def build_Prida_LSTM(input_shape,n_layers=3,layer_size=20):
    '''
    model = build_LSTM(input_shape,lr,dropout_rate,n_layers,layer_size,seed) 
    Builds the specified LSTM model \n
    Inputs:        
        input_shape:        
        x:                [timesteps x n_channels] input dimensionality of the data,
        n_layers:        int, # of LSTM layers
        layer_size:     int, # number of LSTM units per layer
        
    Output: 
        model: LSTM keras model
    '''
    K.clear_session()
    dropout_rate=0.2            # Hard fix to a standard value
    # input layer
    inputs = Input(shape=input_shape)
    
    #LSTM layers
    for i in range(n_layers):
        if i==0:
            x = LSTM(layer_size, return_sequences=True,
                                kernel_initializer=GlorotUniform(),
                                recurrent_initializer=Orthogonal(),
                                )(inputs)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        else:
            x = LSTM(layer_size, return_sequences=True,
                        kernel_initializer=GlorotUniform(),
                        recurrent_initializer=Orthogonal(),
                        )(x)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
    predictions = Dense(1, activation='sigmoid',kernel_initializer=GlorotUniform())(x)
    # Define model
    model = Model(inputs=inputs,
                               outputs=predictions,
                               name='BCG_LSTM')

    opt = tf.keras.optimizers.Adam(learning_rate=0.005)  # Hard fixed to 0.005
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mse'])

    return model
