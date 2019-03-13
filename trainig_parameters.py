import datetime

from datasets.CatLFP import CatLFP


class ModelTrainingParameters():
    def __init__(self):
        self.n_epochs = 50
        self.batch_size = 32
        self.nr_layers = 8
        self.frame_size = 2 ** self.nr_layers
        self.nr_filters = 32
        self.frame_shift = 8
        self.lr = 0.00001
        self.loss = 'CAT'
        self.clip = True
        self.random = True
        self.nr_bins = 256
        self.skip_conn_filters = 32
        self.regularization_coef = 0.0001
        self.nr_train_steps = 1850  # dataset.get_total_length("TRAIN") // batch_size // 400
        self.nr_val_steps = 1000  # np.ceil(0.1*dataset.get_total_length("VAL"))
        self.save_path = None
        self.dataset = CatLFP(nr_bins=self.nr_bins)

    def get_model_name(self):
        return "Wavenet_L:{}_Ep:{}_StpEp:{}_Lr:{}_BS:{}_Fltrs:{}_SkipFltrs:{}_L2:{}_FS:{}_{}_Clip:{}_Rnd:{}".format(
            self.nr_layers,
            self.n_epochs,
            self.nr_train_steps,
            self.lr,
            self.batch_size,
            self.nr_filters,
            self.skip_conn_filters,
            self.regularization_coef,
            self.frame_shift,
            self.loss,
            self.clip,
            self.random)

    def get_save_path(self):
        if self.save_path is None:
            self.save_path = './LFP_models/' + self.get_model_name() + '/' + datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M")
        return self.save_path

    def get_classifying(self):
        return self.loss == "CAT"
