from keras import losses, Input, Model, optimizers
from keras.activations import softmax
from keras.layers import Conv1D, Multiply, Add, Activation, Flatten, Dense
from keras.regularizers import l2


def wavenet_block(n_filters, filter_size, dilation_rate, regularization_coef):
    def f(input_):
        residual = input_
        tanh_out = Conv1D(n_filters, filter_size,
                          dilation_rate=dilation_rate,
                          padding='causal',
                          activation='tanh',
                          kernel_regularizer=l2(regularization_coef))(input_)
        sigmoid_out = Conv1D(n_filters, filter_size,
                             dilation_rate=dilation_rate,
                             padding='causal',
                             activation='sigmoid',
                             kernel_regularizer=l2(regularization_coef))(input_)

        merged = Multiply()([tanh_out, sigmoid_out])
        skip_out = Conv1D(n_filters * 2, 1, padding='same',
                          kernel_regularizer=l2(regularization_coef))(merged)

        out = Conv1D(n_filters, 1, padding='same',
                     kernel_regularizer=l2(regularization_coef))(merged)
        full_out = Add(name="Block_{}_Out".format(dilation_rate))([out, residual])
        return full_out, skip_out

    return f


def get_basic_generative_model(nr_filters, input_size, nr_layers, lr, loss, clipping, skip_conn_filters,
                               regularization_coef, output_size=256):
    if loss is "MSE":
        model_loss = losses.MSE
    elif loss is "MAE":
        model_loss = losses.MAE
    elif loss is "CAT":
        model_loss = losses.sparse_categorical_crossentropy
    else:
        raise ValueError('Use one of the following loss functions: MSE, MAE, CAT (categorical crossentropy)')

    if clipping:
        clipvalue = .5
    else:
        clipvalue = 20

    input_ = Input(shape=(input_size, 1))
    A, B = wavenet_block(nr_filters, 2, 1, regularization_coef=regularization_coef)(input_)
    skip_connections = [B]
    for i in range(1, nr_layers):
        dilation_rate = 2 ** i
        A, B = wavenet_block(nr_filters, 2, dilation_rate, regularization_coef=regularization_coef)(A)
        skip_connections.append(B)
    net = Add()(skip_connections)
    net = Activation('relu')(net)
    net = Conv1D(skip_conn_filters, 1, activation='relu', kernel_regularizer=l2(regularization_coef))(net)
    net = Conv1D(skip_conn_filters, 1, kernel_regularizer=l2(regularization_coef))(net)
    net = Flatten()(net)

    if model_loss is losses.sparse_categorical_crossentropy:
        net = Dense(output_size, activation=softmax, name="Model_Output")(net)
    else:
        net = Dense(1, name="Model_Output", kernel_regularizer=l2(regularization_coef))(net)

    model = Model(inputs=input_, outputs=net)
    optimizer = optimizers.adam(lr=lr, clipvalue=clipvalue)
    model.compile(loss=model_loss, optimizer=optimizer)
    return model
