import os

from keras.regularizers import l2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras import losses, optimizers, callbacks
from keras.activations import softmax
from keras.callbacks import TensorBoard, CSVLogger
from keras.layers import Flatten, Dense, \
    Input, Activation, Conv1D, Add, Multiply
from keras.models import Model, load_model
from datasets.CatLFP import CatLFP
from keras.backend.tensorflow_backend import set_session
from textwrap import wrap

import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


class PlotCallback(callbacks.Callback):
    def __init__(self, model_name, plot_period, classifying, frame_size, nr_predictions_steps, save_path):
        super().__init__()
        self.model_name = model_name
        self.epoch = 0
        self.save_path = save_path
        self.nr_prediction_steps = nr_predictions_steps
        self.frame_size = frame_size
        self.classifying = classifying
        self.plot_period = plot_period

    def on_train_begin(self, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1

        if self.epoch % self.plot_period == 0 or self.epoch == 1:
            get_predictions(self.model, self.epoch, self.save_path, self.classifying, nr_steps=self.nr_prediction_steps,
                            starting_point=0, teacher_forcing=True)
            get_predictions(self.model, self.epoch, self.save_path, self.classifying, nr_steps=self.nr_prediction_steps,
                            starting_point=0, teacher_forcing=False)
        return

    def on_train_end(self, logs=None):
        get_predictions(self.model, self.epoch, self.save_path, self.classifying, nr_steps=self.nr_prediction_steps,
                        starting_point=0, teacher_forcing=True)
        get_predictions(self.model, self.epoch, self.save_path, self.classifying, nr_steps=self.nr_prediction_steps,
                        starting_point=0, teacher_forcing=False)
        return


def plot_predictions(train_seq, title, nr_predictions, predictions, save_path, starting_point, teacher_forcing):
    plt.figure(figsize=(16, 12))
    plt.title("\n".join(wrap(model_name + '_TF:' + str(teacher_forcing), 33)))
    plt.plot(train_seq[:nr_predictions + frame_size], label="Original sequence")
    plt.plot(range(starting_point + frame_size, starting_point + nr_predictions + frame_size), predictions,
             label="Predicted sequence")
    plt.legend()
    plt.savefig(save_path + '/' + str(title) + "TF:" + str(teacher_forcing) + ".png")
    plt.show()
    plt.close()


def get_predictions(model, epoch, save_path, classifying, nr_steps=10000, starting_point=0, teacher_forcing=True):
    nr_predictions = min(nr_steps, dataset.val_length - starting_point - frame_size - 1)
    movies_concat = np.tile(movies, 2)
    seqs_to_predict = zip(movies_concat, trial, channels)

    predictions = np.zeros(nr_predictions)
    position = 0

    for m_index, t_index, c_index in seqs_to_predict:
        sequence = dataset.validation[m_index][t_index, c_index, :]
        if teacher_forcing:
            for step in range(starting_point, starting_point + nr_predictions):
                input_sequence = np.reshape(sequence[step:step + frame_size], (-1, frame_size, 1))
                predicted = decode_model_output(model.predict(input_sequence), classifying)
                predictions[position] = predicted
                position += 1
        else:
            input_sequence = np.reshape(sequence[:frame_size], (-1, frame_size, 1))
            for step in range(starting_point, starting_point + nr_predictions):
                predicted = decode_model_output(model.predict(input_sequence), classifying)
                predictions[position] = predicted
                input_sequence = np.append(input_sequence[:, 1:, :], np.reshape(predicted, (-1, 1, 1)), axis=1)
                position += 1

        plot_predictions(sequence, str(epoch) + "M:{}_T:{}_C:{}".format(m_index, t_index, c_index), nr_predictions,
                         predictions, save_path, starting_point, teacher_forcing)
        position = 0


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
                               output_size=256, regularization_coef=0.001):
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
        net = Dense(output_size, activation=softmax, name="Model_Output", kernel_regularizer=l2(regularization_coef))(
            net)
    else:
        net = Dense(1, name="Model_Output", kernel_regularizer=l2(regularization_coef))(net)

    model = Model(inputs=input_, outputs=net)
    optimizer = optimizers.adam(lr=lr, clipvalue=clipvalue)
    model.compile(loss=model_loss, optimizer=optimizer)
    # model.summary()
    return model


def decode_model_output(model_logits, classifying):
    if classifying:
        bin_index = np.argmax(model_logits)
        a = (dataset.bins[bin_index - 1] + dataset.bins[bin_index]) / 2
        return a
    return model_logits


def train_model(nr_train_steps, nr_val_steps, clip, random, save_path, skip_conn_filters, regularization_coef):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = get_basic_generative_model(nr_filters, frame_size, nr_layers, lr=lr, loss=loss, clipping=clip,
                                       skip_conn_filters=skip_conn_filters, regularization_coef=regularization_coef)

    print('Total training steps:', nr_train_steps)
    print('Total validation steps:', nr_val_steps)
    classifying = True if loss == "CAT" else False

    tensor_board_callback = TensorBoard(log_dir=save_path, write_graph=True)
    log_callback = CSVLogger(save_path + "/session_log.csv")
    plot_figure_callback = PlotCallback(model_name, 1, classifying, frame_size=frame_size,
                                        nr_predictions_steps=500,
                                        save_path=save_path)

    model.fit_generator(dataset.train_frame_generator(frame_size, batch_size, classifying),
                        steps_per_epoch=nr_train_steps, epochs=n_epochs,
                        validation_data=dataset.validation_frame_generator(frame_size, batch_size, classifying),
                        validation_steps=nr_val_steps,
                        verbose=1,
                        callbacks=[tensor_board_callback, plot_figure_callback, log_callback])

    print('Saving model and results...')
    model.save(save_path + '.h5')
    print('\nDone!')


def test_model(save_path):
    model = load_model(
        save_path + '.h5')
    model.summary()
    classifying = True if loss == "CAT" else False
    get_predictions(model, "After_training", save_path, classifying, nr_steps=3000, teacher_forcing=False)
    get_predictions(model, "After_training", save_path, classifying, nr_steps=3000, teacher_forcing=True)


n_epochs = 50
batch_size = 32
nr_layers = 8
frame_size = 2 ** nr_layers
nr_filters = 16
frame_shift = 8
lr = 0.00001
loss = 'CAT'
clip = True
random = True
nr_bins = 256
skip_conn_filters = 32
regularization_coef = 0.0001
nr_train_steps = 3600  # dataset.get_total_length("TRAIN") // batch_size // 400
nr_val_steps = 2000  # np.ceil(0.1*dataset.get_total_length("VAL"))
np.random.seed(42)

print("Frame size is {}".format(frame_size))

dataset = CatLFP(nr_bins=nr_bins)
channels = np.random.choice(dataset.nr_channels, 6)
trial = np.random.choice(dataset.trials_per_condition, 6)
movies = np.arange(1, 4)

now = datetime.datetime.now()
model_name = "Wavenet_L:{}_Ep:{}_StpEp:{}_Lr:{}_BS:{}_Fltrs:{}_SkipFltrs:{}_L2:{}_FS:{}_{}_Clip:{}_Rnd:{}".format(
    nr_layers,
    n_epochs,
    nr_train_steps,
    lr,
    batch_size,
    nr_filters,
    skip_conn_filters,
    regularization_coef,
    frame_shift,
    loss, clip,
    random)
save_path = '/data2/razpa/LFP_models/' + model_name + '/' + now.strftime("%Y-%m-%d %H:%M")
print(model_name)

if __name__ == '__main__':
    train_model(nr_train_steps, nr_val_steps, clip, random, save_path, skip_conn_filters=skip_conn_filters,
                regularization_coef=regularization_coef)
    test_model(save_path)

