from imblearn.over_sampling import RandomOverSampler
from imblearn.tensorflow import balanced_batch_generator
from tensorflow.python.keras.utils.data_utils import Sequence


class BalancedDataGenerator(Sequence):
    """ImageDataGenerator + RandomOversampling"""

    def __init__(self, x, y, datagen, batch_size=32, classes=None):
        self.datagen = datagen
        self.batch_size = min(batch_size, x.shape[0])
        if classes is not None:
            self.class_indices = dict(zip(classes, range(len(classes))))
        datagen.fit(x)
        self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(
            x.shape[0], -1), y, sampler=RandomOverSampler(), batch_size=self.batch_size, keep_sparse=True)
        self._shape = (self.steps_per_epoch * batch_size, *x.shape[1:])

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1, *self._shape[1:])
        return self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()
