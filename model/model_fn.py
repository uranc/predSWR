from tensorflow.keras import applications
from tensorflow.keras.layers import Conv1D, Conv2D, ELU, Input, LSTM, Dense, Dropout, BatchNormalization, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras import backend as K
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

    # load labels
    inputs = Input(shape=(input_timepoints, input_chans))
    # nets = inputs

    # get TCN
    baseTCN = False
    if baseTCN:
        nets = TCN(nb_filters=64,
                    kernel_size=5,
                    nb_stacks=1,
                    # dilations=[2 ** i for i in range(9)],
                    dilations=(1, 2, 4, 8), #, 32
                    padding='causal',
                    use_skip_connections=True,
                    dropout_rate=0.0,
                    return_sequences=True,
                    activation='relu',
                    kernel_initializer='glorot_uniform',
                    # kernel_initializer='he_normal',
                    use_depthwise=False,
                    use_batch_norm=False,
                    use_layer_norm=False,
                    use_weight_norm=True,
                    go_backwards=False,
                    return_state=False)(inputs)
        # pdb.set_trace()
        # nets = Dense(1, kernel_initializer=GlorotUniform(), use_bias=True)(nets)
        nets = Dense(1, activation='sigmoid', kernel_initializer=GlorotUniform())(nets)
        # nets = Activation('sigmoid')(nets)
        nets = Flatten()(nets)
    else:
        dropout_rate = 0.0
        nets = inputs
        all_nets = []
        for i_dilate in range(5):
            nets = WeightNormalization(Conv1D(filters=128,
                        kernel_size=3,
                        dilation_rate=2**(i_dilate),
                        groups=input_chans,
                        padding='causal',
                        # activation='relu',
                        activation=ELU(alpha=0),
                        name='dconv1_{0}'.format(i_dilate),
                        kernel_initializer='glorot_uniform'))(nets)
            all_nets.append(nets)
            if dropout_rate:
                nets = Dropout(dropout_rate)(nets)
            # wrap it. WeightNormalization API is different than BatchNormalization or LayerNormalization.
            # nets = WeightNormalization(nets)(nets)
            # with K.name_scope('norm_{}'.format(i_dilate)):
                # nets = WeightNormalization(nets)
            # nets = Conv1D(filters=64,
            #             kernel_size=2,
            #             dilation_rate=2**(i_dilate),
            #             groups=input_chans,
            #             padding='causal',
            #             activation='relu',
            #             # activation=ELU(alpha=0),
            #             name='dconv2_{0}'.format(i_dilate),
            #             kernel_initializer='glorot_uniform')(nets)
        
        tt = tf.reduce_sum(tf.stack(all_nets, axis=-1))
        pdb.set_trace()
        # reduce dilation
        nets = WeightNormalization(Conv1D(filters=128,
                    kernel_size=3,
                    dilation_rate=1,
                    groups=input_chans,
                    padding='causal',
                    # activation='relu',
                    activation=ELU(alpha=0),
                    name='de_dconv1_{0}'.format(0),
                    kernel_initializer='glorot_uniform'))(nets)
        if dropout_rate:
            nets = Dropout(dropout_rate)(nets)
            
        nets = WeightNormalization(Conv1D(filters=128,
                    kernel_size=3,
                    dilation_rate=1,
                    groups=input_chans,
                    padding='causal',
                    # activation='relu',
                    activation=ELU(alpha=0),
                    name='de_dconv1_{0}'.format(1),
                    kernel_initializer='glorot_uniform'))(nets)
        if dropout_rate:
            nets = Dropout(dropout_rate)(nets)            
        # with K.name_scope('norm_{}'.format(10)):
        # nets = WeightNormalization()(nets)
        # nets = Conv1D(filters=64,
        #             kernel_size=2,
        #             dilation_rate=1,
        #             groups=input_chans,
        #             padding='causal',
        #             activation='relu',
        #             # activation=ELU(alpha=0),
        #             name='de_dconv2_{0}'.format(0),
        #             kernel_initializer='glorot_uniform')(nets)
        
        # reduce mean and get output
        nets = tf.reduce_mean(nets, axis=-1, keepdims=True)
        nets = Dense(1, activation='sigmoid', kernel_initializer=GlorotUniform())(nets)
        nets = Flatten()(nets)
        # pdb.set_trace()
        

    # get outputs
    outputs = nets
    model = Model(inputs=[inputs], outputs=[outputs])
    if params['WEIGHT_FILE']:
        print('load model')
        model.load_weights(params['WEIGHT_FILE'])

    # def loss_fn(y_true, y_pred):
    #     # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
    #     if K.ndim(y_true) == K.ndim(y_pred):
    #         y_true = K.squeeze(y_true, -1)
    #     # convert dense predictions to labels
    #     y_pred_labels = K.argmax(y_pred, axis=-1)
    #     y_pred_labels = K.cast(y_pred_labels, K.floatx())
    #     return K.cast(K.equal(y_true, y_pred_labels), K.floatx())
    # pdb.set_trace()
    
    # def frame_wise_binary_focal_crossentropy(y_true, y_pred):
    #     gamma = 2.0
    #     alpha = 0.25

    #     # Compute the binary cross entropy loss
    #     bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)

    #     # Compute the focal loss
    #     pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    #     focal_loss = -alpha * (1 - pt) ** gamma * tf.math.log(pt + tf.keras.backend.epsilon())

    #     # Compute the frame-wise loss
    #     # frame_wise_loss = tf.reduce_mean(focal_loss, axis=-1)
    #     frame_wise_loss = focal_loss
        
    #     # Compute the weighted frame-wise loss
    #     if anchor_point is not None:
    #         anchor_point = tf.argmax(tf.where(tf.equal(y_true[:-1], 0) & tf.equal(y_true[1:], 1)))
    #         decay_weights = tf.linspace(1.0, 0.1, tf.shape(frame_wise_loss)[1] - anchor_point, dtype=tf.float32)
    #         decay_weights = tf.concat([tf.ones([tf.shape(frame_wise_loss)[0], anchor_point]), decay_weights], axis=1)
    #         decay_weights = tf.concat([decay_weights, tf.reverse(decay_weights[:, :-1], axis=[1])], axis=1)
    #         weighted_frame_wise_loss = frame_wise_loss * decay_weights
    #     else:
    #         weighted_frame_wise_loss = frame_wise_loss
    #     total_loss = weighted_frame_wise_loss
        
        # pdb.set_trace()
        # return total_loss
    
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=params['LEARNING_RATE']),
                    # loss = frame_wise_binary_focal_crossentropy,
                    loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True), 
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
