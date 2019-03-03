import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

from textwrap import wrap
from keras import losses, optimizers, callbacks
from keras.activations import softmax
from keras.callbacks import TensorBoard
from keras.layers import Flatten, Dense, \
    Input, Activation, Conv1D, Add, Multiply, Lambda
from keras.models import Model, load_model
from LFP_Dataset import LFPDataset


class PlotCallback(callbacks.Callback):
    def __init__(self, model_name, nr_epochs, classifying, frame_size, nr_predictions_steps, save_path):
        super().__init__()
        self.model_name = model_name
        self.epoch = 0
        self.save_path = save_path
        self.nr_epochs = nr_epochs
        self.nr_prediction_steps = nr_predictions_steps
        self.frame_size = frame_size
        self.classifying = classifying

    def on_train_begin(self, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        if self.epoch % 10 == 0 or self.epoch == 1 or self.epoch == self.nr_epochs:
            get_predictions(self.model, self.epoch, self.save_path, self.classifying, nr_steps=self.nr_prediction_steps,
                            starting_point=0, teacher_forcing=True)
            get_predictions(self.model, self.epoch, self.save_path, self.classifying, nr_steps=self.nr_prediction_steps,
                            starting_point=0, teacher_forcing=False)
        return


def plot_predictions(train_seq, epoch, nr_predictions, predictions, save_path, starting_point, teacher_forcing):
    plt.figure(figsize=(16, 12))
    plt.title("\n".join(wrap(model_name + '_TF:' + str(teacher_forcing), 33)))
    plt.plot(train_seq[:nr_predictions + frame_size], label="Original sequence")
    plt.plot(range(starting_point + frame_size, starting_point + nr_predictions + frame_size), predictions,
             label="Predicted sequence")
    plt.legend()
    plt.savefig(save_path + '/' + str(epoch) + "TF:" + str(teacher_forcing) + ".png")
    plt.show()


def get_predictions(model, epoch, save_path, classifying, nr_steps=10000, starting_point=0, teacher_forcing=True):
    nr_predictions = min(nr_steps, dataset.train_length - starting_point - frame_size - 1)
    train_seq = dataset.get_validation_set(1, 1, 0)
    predictions = np.zeros(nr_predictions)
    position = 0

    if teacher_forcing:
        for step in range(starting_point, starting_point + nr_predictions):
            input_sequence = np.reshape(train_seq[step:step + frame_size], (-1, frame_size, 1))
            predicted = decode_model_output(model.predict(input_sequence), classifying)
            predictions[position] = predicted
            position += 1
    else:
        input_sequence = np.reshape(train_seq[:frame_size], (-1, frame_size, 1))
        for step in range(starting_point, starting_point + nr_predictions):
            predicted = decode_model_output(model.predict(input_sequence), classifying)
            predictions[position] = predicted
            input_sequence = np.append(input_sequence[:, 1:, :], np.reshape(predicted, (-1, 1, 1)), axis=1)
            position += 1

    plot_predictions(train_seq, epoch, nr_predictions, predictions, save_path, starting_point, teacher_forcing)


def wavenet_block(n_filters, filter_size, dilation_rate):
    def f(input_):
        residual = input_
        tanh_out = Conv1D(n_filters, filter_size,
                          dilation_rate=dilation_rate,
                          padding='causal',
                          activation='tanh')(input_)
        sigmoid_out = Conv1D(n_filters, filter_size,
                             dilation_rate=dilation_rate,
                             padding='causal',
                             activation='sigmoid')(input_)

        merged = Multiply()([tanh_out, sigmoid_out])
        skip_out = Conv1D(n_filters * 4, 1, padding='same')(merged)

        out = Conv1D(n_filters, 1, padding='same')(merged)
        full_out = Add(name="Block_{}_Out".format(dilation_rate))([out, residual])
        return full_out, skip_out

    return f


def get_basic_generative_model(nr_filters, input_size, nr_layers, lr, loss, clipping, output_size=256):
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
    A, B = wavenet_block(nr_filters, 2, 1)(input_)
    skip_connections = [B]
    for i in range(1, nr_layers):
        dilation_rate = 2 ** i
        A, B = wavenet_block(nr_filters, 2, dilation_rate)(A)
        skip_connections.append(B)
    net = Add()(skip_connections)
    net = Activation('relu')(net)
    net = Conv1D(nr_bins, 1, activation='relu')(net)
    net = Conv1D(nr_bins, 1)(net)
    net = Flatten()(net)

    if model_loss is losses.sparse_categorical_crossentropy:
        net = Lambda(lambda x: x + 1e-9)(net)
        net = Dense(output_size, activation=softmax, name="Model_Output")(net)
        # epsilons = np.array([1e-10 for _ in range(output_size)])
        # k_epsilons = variable(epsilons)
        # epsilon_inputs = Input(tensor=k_epsilons)
        # net = Add()([net, epsilon_inputs])
    else:
        net = Dense(1, name="Model_Output")(net)

    model = Model(input=input_, output=net)
    optimizer = optimizers.adam(lr=lr, clipvalue=clipvalue)
    model.compile(loss=model_loss, optimizer=optimizer)
    model.summary()
    return model


def encode_input_to_bin(target_val):
    bin = np.searchsorted(bins, target_val, side='left')
    return bin


def decode_model_output(model_logits, classifying):
    if classifying:
        bin_index = np.argmax(model_logits)
        a = (bins[bin_index - 1] + bins[bin_index]) / 2
        return a
    return model_logits


def train_model(nr_train_steps, nr_val_steps, clip, random, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = get_basic_generative_model(nr_filters, frame_size, nr_layers, lr=lr, loss=loss, clipping=clip)

    print('Total training steps:', nr_train_steps)
    print('Total validation steps:', nr_val_steps)
    classifying = True if loss == "CAT" else False
    outputTransform = encode_input_to_bin if classifying else lambda x: x

    tensor_board_callback = TensorBoard(log_dir=save_path, write_graph=True)
    plot_figure_callback = PlotCallback(model_name, n_epochs, classifying, frame_size=frame_size,
                                        nr_predictions_steps=3000,
                                        save_path=save_path)

    model.fit_generator(dataset.train_frame_generator(frame_size, batch_size, outputTransform),
                        steps_per_epoch=nr_train_steps, epochs=n_epochs,
                        validation_data=dataset.validation_frame_generator(frame_size, batch_size, outputTransform),
                        validation_steps=nr_val_steps, verbose=2,
                        callbacks=[tensor_board_callback, plot_figure_callback])

    print('Saving model...')
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
nr_layers = 6
frame_size = 2 ** nr_layers
nr_filters = 32
frame_shift = 8
lr = 0.0001
loss = 'CAT'
clip = True
random = True

print("Frame size is {}".format(frame_size))
model_name = "Wavenet_L:{}_Ep:{}_Lr:{}_BS:{}_Filters:{}_FS:{}_{}_Clip:{}_Rnd:{}".format(nr_layers,
                                                                                        n_epochs,
                                                                                        lr,
                                                                                        batch_size, nr_filters,
                                                                                        frame_shift, loss, clip,
                                                                                        random)
dataset = LFPDataset("/home/gabir/DATASETS/CER01A50/Bin_cer01a50-LFP.json", )

min_train_seq = np.floor(dataset.values_range[0])
max_train_seq = np.ceil(dataset.values_range[1])
nr_bins = 256
bins = np.linspace(min_train_seq, max_train_seq, nr_bins)
bin_size = bins[1] - bins[0]
nr_train_steps = dataset.train_length // batch_size
nr_val_steps = dataset.validation_length // batch_size

now = datetime.datetime.now()
save_path = 'LFP_models/' + model_name + '/' + now.strftime("%Y-%m-%d %H:%M")

if __name__ == '__main__':
    train_model(nr_train_steps, nr_val_steps, clip, random, save_path)
    test_model(save_path)
