from tensorflow.keras import applications
from tensorflow.keras.layers import Conv2D, ELU, Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
import tensorflow.keras.backend as keras_backend
import tensorflow as tf

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

    model.compile(optimizer=optimizers.AdamW(lr=params['LEARNING_RATE']),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])
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
    keras_backend.clear_session()
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
