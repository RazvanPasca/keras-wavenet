import matplotlib.pyplot as plt
import numpy as np
from keras import callbacks


def decode_model_output(model_logits, classifying, bins):
    if classifying:
        bin_index = np.argmax(model_logits)
        a = (bins[bin_index - 1] + bins[bin_index]) / 2
        return a
    return model_logits


def plot_predictions(original_sequence, image_title, nr_predictions, frame_size, predicted_sequence, save_path,
                     starting_point,
                     teacher_forcing):
    plt.figure(figsize=(16, 12))
    title = image_title + "TF:{}".format(teacher_forcing)
    plt.title(title)
    plt.plot(range(starting_point + frame_size, starting_point + nr_predictions + frame_size),
             original_sequence[starting_point + frame_size:starting_point + nr_predictions + frame_size],
             label="Original sequence")
    plt.plot(range(starting_point + frame_size, starting_point + nr_predictions + frame_size), predicted_sequence,
             label="Predicted sequence")
    plt.legend()
    plt.savefig(save_path + '/' + title + ".png")
    plt.show()
    plt.close()


def get_predictions_on_sequence(model,
                                model_params,
                                original_sequence,
                                nr_predictions,
                                image_name,
                                starting_point=0,
                                teacher_forcing=True):
    nr_actual_predictions = min(nr_predictions, original_sequence.size - starting_point - model_params.frame_size - 1)
    predicted_sequence = np.zeros(nr_actual_predictions)
    position = 0

    if teacher_forcing:
        for step in range(starting_point, starting_point + nr_actual_predictions):
            input_sequence = np.reshape(original_sequence[step:step + model_params.frame_size],
                                        (-1, model_params.frame_size, 1))
            predicted = decode_model_output(model.predict(input_sequence), model_params.get_classifying(),
                                            model_params.dataset.bins)
            predicted_sequence[position] = predicted
            position += 1
    else:
        input_sequence = np.reshape(original_sequence[:model_params.frame_size], (-1, model_params.frame_size, 1))
        for step in range(starting_point, starting_point + nr_actual_predictions):
            predicted = decode_model_output(model.predict(input_sequence), model_params.get_classifying(),
                                            model_params.dataset.bins)
            predicted_sequence[position] = predicted
            input_sequence = np.append(input_sequence[:, 1:, :], np.reshape(predicted, (-1, 1, 1)), axis=1)
            position += 1

    plot_predictions(original_sequence, image_name, nr_actual_predictions,
                     model_params.frame_size, predicted_sequence, model_params.get_save_path(), starting_point,
                     teacher_forcing)


def generate_prediction_name(seq_addr):
    name = ''
    for key in seq_addr:
        name += '{}:{}_'.format(key, str(seq_addr[key]))
    return name


def get_predictions(model, model_params, nr_steps=1000):
    pred_seqs = model_params.dataset.prediction_sequences
    for source in pred_seqs:
        for sequence, addr in pred_seqs[source]:
            image_name = generate_prediction_name(addr)
            get_predictions_on_sequence(model, model_params, sequence, nr_steps, image_name, 500, True)
            get_predictions_on_sequence(model, model_params, sequence, nr_steps, image_name, 500, False)


class PlotCallback(callbacks.Callback):
    def __init__(self, model_params, plot_period, nr_predictions_steps):
        super().__init__()
        self.model_params = model_params
        self.epoch = 0
        self.nr_prediction_steps = nr_predictions_steps
        self.plot_period = plot_period

    def on_train_begin(self, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1

        if self.epoch % self.plot_period == 0 or self.epoch == 1:
            get_predictions(self.model, self.model_params, nr_steps=self.nr_prediction_steps)

    def on_train_end(self, logs=None):
        get_predictions(self.model, self.model_params, nr_steps=self.nr_prediction_steps)
