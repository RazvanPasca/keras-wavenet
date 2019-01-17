import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from keras import losses, optimizers, callbacks
from keras.callbacks import TensorBoard
from keras.layers import Flatten, Dense, \
    Input, Activation, Conv1D, Add, Multiply
from keras.models import Model, load_model


class PlotCallback(callbacks.Callback):
    def __init__(self, model_name, save_path):
        super().__init__()
        self.model_name = model_name
        self.epoch = 0
        self.save_path = save_path

    def on_train_begin(self, logs={}):
        return

    """Callback to save plots at epoch end"""

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        if self.epoch % 10 == 0 or self.epoch == 1:
            starting_point = 0
            nr_predictions = min(3000, train_sequence_length - starting_point - frame_size - 1)
            predictions = np.zeros(nr_predictions)
            position = 0

            for step in range(starting_point, starting_point + nr_predictions):
                input = np.reshape(x[step:step + frame_size], (-1, frame_size, 1))
                predicted = self.model.predict(input)
                predictions[position] = predicted
                position += 1
            plt.figure()
            plt.title(model_name)
            plt.plot(train_sequence, label="Train")
            plt.plot(range(starting_point + frame_size, starting_point + nr_predictions + frame_size), predictions,
                     label="Predicted")
            plt.legend()
            plt.savefig(self.save_path + '/' + str(epoch) + ".png")
        return


def plot_predictions(model, save=False):
    starting_point = 0
    nr_predictions = min(3000, train_sequence_length - starting_point - frame_size - 1)
    predictions = np.zeros(nr_predictions)
    position = 0
    for step in range(starting_point, starting_point + nr_predictions):
        input = np.reshape(x[step:step + frame_size], (-1, frame_size, 1))
        predicted = model.predict(input)
        predictions[position] = predicted
        position += 1
    plt.figure()
    plt.title(model_name)
    plt.plot(train_sequence, label="Train")
    plt.plot(range(starting_point + frame_size, starting_point + nr_predictions + frame_size), predictions,
             label="Predicted")
    plt.legend()
    plt.show()


def wavenetBlock(n_filters, filter_size, dilation_rate):
    def f(input_):
        residual = input_
        tanh_out = Conv1D(n_filters, filter_size,
                          dilation_rate=dilation_rate,
                          padding='same',
                          activation='tanh')(input_)
        sigmoid_out = Conv1D(n_filters, filter_size,
                             dilation_rate=dilation_rate,
                             padding='same',
                             activation='sigmoid')(input_)

        merged = Multiply()([tanh_out, sigmoid_out])
        skip_out = Conv1D(n_filters, 1, padding='same')(merged)

        out = Conv1D(n_filters, 1, padding='same')(merged)
        full_out = Add(name="Block_{}_Out".format(dilation_rate))([out, residual])
        return full_out, skip_out

    return f


def get_basic_generative_model(nr_filters, input_size, nr_layers, lr, loss):
    if loss is "MSE":
        model_loss = losses.MSE
    else:
        model_loss = losses.MAE

    input_ = Input(shape=(input_size, 1))
    A, B = wavenetBlock(nr_filters, 2, 1)(input_)
    # A = Conv1D(nr_filters, 2, dilation_rate=1, padding='same')(input_)
    skip_connections = [B]
    for i in range(1, nr_layers):
        dilation_rate = 2 ** i
        A, B = wavenetBlock(nr_filters, 2, dilation_rate)(A)
        skip_connections.append(B)
    net = Add()(skip_connections)
    net = Activation('relu')(net)
    net = Conv1D(1, 1, activation='relu')(net)
    net = Conv1D(1, 1)(net)
    net = Flatten()(net)
    net = Dense(1, name="Model_Output")(net)
    model = Model(input=input_, output=net)
    optimizer = optimizers.adam(lr=lr, clipvalue=1.)
    model.compile(loss=model_loss, optimizer=optimizer)
    model.summary()
    return model


def frame_generator(target_series, frame_size, frame_shift, batch_size):
    # TODO pick batches randomly such that we don't overfit growing phase
    series_len = len(target_series)
    X = []
    y = []
    while 1:
        batch_start = np.random.choice(range(0, series_len - frame_shift * batch_size - frame_size - 1))
        for i in range(batch_start, batch_start + frame_shift * batch_size, frame_shift):
            frame = target_series[i:i + frame_size]
            temp = target_series[i + frame_size]
            X.append(frame.reshape(frame_size, 1))
            y.append(temp)
            if len(X) == batch_size:
                yield np.array(X), np.array(y)
                X = []
                y = []


def train_model(nr_train_steps, nr_val_steps, save_path):
    valid_sequence = train_sequence
    model = get_basic_generative_model(nr_filters, frame_size, nr_layers, lr=lr, loss=loss)

    print('Total training steps:', nr_train_steps)
    print('Total validation steps:', nr_val_steps)

    training_data_gen = frame_generator(train_sequence, frame_size, frame_shift,
                                        batch_size)
    validation_data_gen = frame_generator(valid_sequence, frame_size, frame_shift, batch_size)

    tensor_board_callback = TensorBoard(log_dir=save_path, write_graph=True)
    plot_figure_callback = PlotCallback(model_name, save_path)

    model.fit_generator(training_data_gen, steps_per_epoch=nr_train_steps, epochs=n_epochs,
                        validation_data=validation_data_gen, validation_steps=nr_val_steps, verbose=2,
                        callbacks=[tensor_board_callback, plot_figure_callback])

    print('Saving model...')
    model.save(save_path + '.h5')
    print('\nDone!')


def test_model():
    model = load_model(
        path + '.h5')
    plot_predictions(model)


n_epochs = 10
batch_size = 1
nr_layers = 8
frame_size = 2 ** nr_layers
nr_filters = 32
frame_shift = 1
lr = 0.0001
loss = 'MSE'

print("Frame size is {}".format(frame_size))
model_name = "Wavenet_L:{}_Ep:{}_Lr:{}_BS:{}_FS:{}_{}_clip".format(nr_layers,
                                                                   n_epochs,
                                                                   lr,
                                                                   batch_size,
                                                                   frame_shift, loss)

valid_sequence_length = 512
train_sequence_length = 2048

x = np.linspace(0 - np.pi / 4, 2 * np.pi, train_sequence_length)
train_sequence = np.sin(x)
now = datetime.datetime.now()
path = 'models/' + model_name + '/' + now.strftime("%Y-%m-%d %H:%M")

if not os.path.exists(path):
    os.makedirs(path)

nr_train_steps = 1000
nr_val_steps = 100

if __name__ == '__main__':
    train_model(nr_train_steps, nr_val_steps, path)
    test_model()
