import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from keras import losses
from keras.callbacks import Callback, TensorBoard
from keras.layers import Flatten, Dense, \
    Input, Activation, Conv1D, Add, Multiply
from keras.models import Model, load_model
from scipy.io.wavfile import read, write


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
        skip_out = Conv1D(1, 1, activation='relu', padding='same')(merged)
        out = Add(name="Block_{}_out".format(dilation_rate))([skip_out, residual])
        return out, skip_out

    return f


def get_basic_generative_model(input_size, nr_layers):
    input_ = Input(shape=(input_size, 1))
    A, B = wavenetBlock(64, 2, 1)(input_)
    skip_connections = [B]
    for i in range(1, nr_layers + 1):
        dilation_rate = 2 ** i
        print(dilation_rate)
        A, B = wavenetBlock(64, 2, dilation_rate)(A)
        skip_connections.append(B)
    net = Add()(skip_connections)
    net = Activation('relu')(net)
    net = Conv1D(1, 1, activation='relu')(net)
    net = Conv1D(1, 1)(net)
    net = Flatten()(net)
    net = Dense(256, activation='relu', name="Model_Output")(net)
    net = Dense(1)(net)
    model = Model(input=input_, output=net)
    model.compile(loss=losses.mean_squared_error, optimizer='adam')
    model.summary()
    return model


def get_audio(filename):
    sr, audio = read(filename)
    audio = audio.astype(float)
    audio = audio - audio.min()
    audio = audio / (audio.max() - audio.min())
    audio = (audio - 0.5) * 2
    return sr, audio


def frame_generator(audio, frame_size, batch_size):
    # TODO pick batches randomly such that we don't overfitt growing
    audio_len = len(audio)
    X = []
    y = []
    while 1:
        for i in range(0, audio_len - frame_size - 1, batch_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                break
            if i + frame_size >= audio_len:
                break
            temp = audio[i + frame_size]
            X.append(frame.reshape(frame_size, 1))
            y.append(temp)
            if len(X) == batch_size:
                yield np.array(X), np.array(y)
                X = []
                y = []


def get_audio_from_model(model, sr, duration, seed_audio):
    print('Generating audio...')
    new_audio = np.zeros((sr * duration))
    curr_sample_idx = 0
    while curr_sample_idx < new_audio.shape[0]:
        distribution = np.array(model.predict(seed_audio.reshape(1,
                                                                 frame_size, 1)
                                              ), dtype=float).reshape(256)
        distribution /= distribution.sum().astype(float)
        predicted_val = np.random.choice(range(256), p=distribution)
        ampl_val_8 = ((((predicted_val) / 255.0) - 0.5) * 2.0)
        ampl_val_16 = (np.sign(ampl_val_8) * (1 / 256.0) * ((1 + 256.0) ** abs(
            ampl_val_8) - 1)) * 2 ** 15
        new_audio[curr_sample_idx] = ampl_val_16
        seed_audio[:-1] = seed_audio[1:]
        seed_audio[-1] = ampl_val_16
        pc_str = str(round(100 * curr_sample_idx / float(new_audio.shape[0]), 2))
        sys.stdout.write('Percent complete: ' + pc_str + '\r')
        sys.stdout.flush()
        curr_sample_idx += 1
    print('Audio generated.')
    return new_audio.astype(np.int16)


class SaveAudioCallback(Callback):
    def __init__(self, ckpt_freq, sr, seed_audio):
        super(SaveAudioCallback, self).__init__()
        self.ckpt_freq = ckpt_freq
        self.sr = sr
        self.seed_audio = seed_audio

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.ckpt_freq == 0:
            ts = str(int(time.time()))
            filepath = os.path.join('output/', 'ckpt_' + ts + '.wav')
            audio = get_audio_from_model(self.model, self.sr, 0.5, self.seed_audio)
            write(filepath, self.sr, audio)


n_epochs = 50
batch_size = 64
nr_layers = 4
frame_size = 2 ** (nr_layers + 1)
print("Frame size is {}".format(frame_size))

valid_sequence_length = 1024
train_sequence_length = 4096

x = np.linspace(0, 4 * np.pi, train_sequence_length)


def audio_main():
    train_sequence = np.sin(5 * x)
    # train_sequence = train_sequence + np.random.normal(0, 0.1, train_sequence.shape)
    valid_sequence = train_sequence[5 * frame_size:]

    nr_train_steps = (train_sequence_length - frame_size - 1) // batch_size
    nr_val_steps = (valid_sequence.shape[0] - frame_size - 1) // batch_size

    plt.figure()
    plt.title("Train sequence and valid sequence")
    plt.plot(train_sequence, label="Train")
    plt.plot(valid_sequence, label="Valid")
    plt.legend()
    plt.show()

    model = get_basic_generative_model(frame_size, nr_layers)
    print('Total training steps:', nr_train_steps)
    print('Total validation steps:', nr_val_steps)

    name = "Wavenet_NrLayers:{}_Epochs:{}_BatchSize:{}_4pi".format(nr_layers, n_epochs, batch_size)

    validation_data_gen = frame_generator(train_sequence, frame_size, batch_size)
    training_data_gen = frame_generator(valid_sequence, frame_size, batch_size)
    tensor_board_callback = TensorBoard(log_dir='tmp/' + name, write_graph=True)

    model.fit_generator(training_data_gen, steps_per_epoch=nr_train_steps, epochs=n_epochs,
                        validation_data=validation_data_gen, validation_steps=nr_val_steps, verbose=2,
                        callbacks=[tensor_board_callback])

    print('Saving model...')
    model.save('models/' + name + '.h5')
    print('\nDone!')


def test_model():
    model = load_model('models/Wavenet_NrLayers:4_Epochs:50_BatchSize:64_4pi.h5')
    train_sequence = np.sin(5 * x)

    nr_predictions = 64
    # starting_point = np.random.choice(range(train_sequence_length - frame_size))
    predictions = np.zeros(nr_predictions)
    starting_point = 0
    position = 0
    for step in range(starting_point, starting_point + nr_predictions):
        input = np.reshape(x[step:step + frame_size], (-1, frame_size, 1))
        predicted = model.predict(input)
        predictions[position] = predicted
        position += 1

    plt.figure()
    plt.title("Original sequence and predicted sequence")
    plt.plot(train_sequence, label="Train")
    plt.plot(range(starting_point, starting_point + nr_predictions), predictions, label="Predicted")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    audio_main()
    # test_model()
