from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, CSVLogger
from keras.models import load_model
from plot_utils import PlotCallback, get_predictions
from training_parameters import ModelTrainingParameters
from wavenet_model import get_basic_generative_model
import os
import tensorflow as tf


def configure_gpu(gpu):
    global config
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)


def train_model(model_params):
    log_training_session(model_parameters)

    if not os.path.exists(model_params.get_save_path()):
        os.makedirs(model_params.get_save_path())

    model = get_basic_generative_model(model_params.nr_filters,
                                       model_params.frame_size,
                                       model_params.nr_layers,
                                       lr=model_params.lr,
                                       loss=model_params.loss,
                                       clipping=model_params.clip,
                                       skip_conn_filters=model_params.skip_conn_filters,
                                       regularization_coef=model_params.regularization_coef)

    tensor_board_callback = TensorBoard(log_dir=model_params.get_save_path(),
                                        write_graph=True)
    log_callback = CSVLogger(model_params.get_save_path() + "/session_log.csv")
    plot_figure_callback = PlotCallback(model_params, 1, nr_predictions_steps=100,
                                        starting_point=1200 - model_params.frame_size)

    model.fit_generator(
        model_params.dataset.train_frame_generator(model_params.frame_size,
                                                   model_params.batch_size,
                                                   model_params.get_classifying()),
        steps_per_epoch=model_params.nr_train_steps, epochs=model_params.n_epochs,
        validation_data=model_params.dataset.validation_frame_generator(model_params.frame_size,
                                                                        model_params.batch_size,
                                                                        model_params.get_classifying()),
        validation_steps=model_params.nr_val_steps,
        verbose=1,
        callbacks=[tensor_board_callback, plot_figure_callback, log_callback])

    print('Saving model and results...')
    model.save(model_params.save_path + '.h5')
    print('\nDone!')


def test_model(model_params):
    save_path = model_params.get_save_path()
    model = load_model(save_path + '.h5')
    model.summary()
    get_predictions(model, model_params, "TrainEnd", starting_point=1200 - model_params.frame_size,
                    nr_prediction_steps=100)
    print("Finished testing model")


def log_training_session(model_params):
    print("Frame size is {}".format(model_params.frame_size))
    print(model_params.get_model_name())
    print('Total training steps:', model_params.nr_train_steps)
    print('Total validation steps:', model_params.nr_val_steps)


if __name__ == '__main__':
    configure_gpu(0)

    model_parameters = ModelTrainingParameters()

    train_model(model_parameters)
    test_model(model_parameters)
