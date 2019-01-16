import matplotlib.pyplot as plt
import numpy as np
from keras import losses, metrics, optimizers
from keras.callbacks import TensorBoard
from keras.layers import Flatten, Dense, \
    Input, Activation, Conv1D, Add, Multiply
from keras.models import Model, load_model
from scipy.io.wavfile import read


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
        skip_out = Conv1D(n_filters, 1, activation='relu', padding='same')(merged)
        out = Conv1D(n_filters, 1, activation='relu', padding='same')(merged)
        out = Add(name="Block_{}_Out".format(dilation_rate))([out, residual])
        return out, skip_out

    return f


def get_basic_generative_model(nr_filters, input_size, nr_layers, lr):
    input_ = Input(shape=(input_size, 1))
    A, B = wavenetBlock(nr_filters, 2, 1)(input_)
    # A = Conv1D(nr_filters, 2, dilation_rate=1, padding='same')(input_)
    skip_connections = [B]
    for i in range(1, nr_layers + 1):
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
    optimizer = optimizers.adam(lr=lr, clipvalue=5.)
    model.compile(loss=losses.mean_squared_error, optimizer=optimizer, metrics=[metrics.mean_absolute_error])
    model.summary()
    return model


def get_audio(filename):
    sr, audio = read(filename)
    audio = audio.astype(float)
    audio = audio - audio.min()
    audio = audio / (audio.max() - audio.min())
    audio = (audio - 0.5) * 2
    return sr, audio


def frame_generator(audio, frame_size, frame_shift, batch_size):
    # TODO pick batches randomly such that we don't overfitt growing
    audio_len = len(audio)
    X = []
    y = []
    while 1:
        for i in range(0, audio_len - frame_size - 1, frame_shift):
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


n_epochs = 50
batch_size = 32
nr_layers = 8
frame_size = 2 ** (nr_layers + 1)
nr_filters = 32
frame_shift = 4
lr = 0.0003
print("Frame size is {}".format(frame_size))
name = "Wavenet_Layers:{}_Epochs:{}_Lr:{}_BS:{}_FShift:{}_TA_clip".format(nr_layers,
                                                                          n_epochs,
                                                                          lr,
                                                                          batch_size,
                                                                          frame_shift)

valid_sequence_length = 1024
train_sequence_length = 4096

x = np.linspace(0, 4 * np.pi, train_sequence_length)


def train_model():
    train_sequence = np.sin(x)
    # train_sequence = train_sequence + np.random.normal(0, 0.1, train_sequence.shape)
    valid_sequence = train_sequence[frame_size // 2:]

    nr_train_steps = (train_sequence_length - frame_size - 1) // batch_size
    nr_val_steps = (valid_sequence.shape[0] - frame_size - 1) // batch_size // 2

    model = get_basic_generative_model(nr_filters, frame_size, nr_layers, lr=lr)
    print('Total training steps:', nr_train_steps)
    print('Total validation steps:', nr_val_steps)

    training_data_gen = frame_generator(train_sequence, frame_size, frame_shift,
                                        batch_size)
    validation_data_gen = frame_generator(valid_sequence, frame_size, frame_shift, batch_size)
    tensor_board_callback = TensorBoard(log_dir='tmp/' + name, write_graph=True)

    model.fit_generator(training_data_gen, steps_per_epoch=nr_train_steps, epochs=n_epochs,
                        validation_data=validation_data_gen, validation_steps=nr_val_steps, verbose=2,
                        callbacks=[tensor_board_callback])

    print('Saving model...')
    model.save('models/' + name + '.h5')
    print('\nDone!')


def test_model():
    path = 'models/' + name + '.h5'
    # path = 'models/' + 'Wavenet_Layers:7_Epochs:50_Lr:0.0003_BS:32_FShift:4_TA_clip' + '.h5'
    model = load_model(
        path)
    train_sequence = np.sin(x)

    nr_predictions = 3000
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
    plt.title(path[7:-3])
    plt.plot(train_sequence, label="Train")
    plt.plot(range(starting_point, starting_point + nr_predictions), predictions, label="Predicted")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_model()
    test_model()
