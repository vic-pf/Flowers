import sys
from os import path
import numpy as np
import pandas as pd
import seaborn as sn
from time import time
import tensorflow as tf
from os import path, makedirs
import DatasetGeneratorFlowers
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

# Import the Desired Version of EfficientNet
from tensorflow.keras.applications import EfficientNetB0


# Variables
NUM_CLASSES = 2
IMG_SIZE = 224


def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    model = EfficientNetB0(
        include_top=False, input_tensor=inputs, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


def plot_hist(hist, name):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig('results/' + name + '.png')


def plot_loss(hist, name):
    plt.clf()
    plt.plot(hist.history['loss'], label="train_loss")
    plt.plot(hist.history['val_loss'], label="val_loss")
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig('results/' + name + '.png')


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          _type='',
                          cmap=plt.cm.Blues):

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        title = 'normalized_confusion_matrix'
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        title = 'confusion_matrix'

    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    cfm_plot.figure.savefig('results/' + title + '_' + _type + '.png')


def print_classification_report(report, _type):
    report_data = []
    lines = report.split('\n')
    with open('results/classification_report_' + _type + '.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')


if __name__ == '__main__':
    train_data_dir = sys.argv[1]
    train_per = float(sys.argv[2])
    epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    augmentation = False
    balanced_batch = False
    for a in range(5, len(sys.argv)):
        if(sys.argv[a] == '-a'):
            augmentation = True
        elif(sys.argv[a] == '-b'):
            balanced_batch = True

    if(train_per > 1):
        train_per = float(train_per/100)

    dataset = DatasetGeneratorFlowers.DatasetGeneratorFlowers(train_data_dir, augmentation=augmentation,
                                                              balanced_batch=balanced_batch, percentage=train_per, IMG_SIZE=IMG_SIZE)

    train_generator, validation_generator, train_steps, validation_steps = dataset.create_generators(
        batch_size)

    filepath = 'recovered_weights.{epoch:02d}-{val_acc:.2f}.hdf5'
    saveModels = ModelCheckpoint(filepath, monitor='val_acc', verbose=0,
                                 save_best_only=False, save_weights_only=False, mode='auto', period=1)
    csv_logger = CSVLogger('training.log')
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    model = build_model(num_classes=NUM_CLASSES)
    if path.exists('first_try1.h5'):
        model.load_weights('first_try1.h5')
    model.summary()

    earlystop_callback = EarlyStopping(
        monitor='val_accuracy', min_delta=0.0001, patience=10)

    history = model.fit(
        train_generator, epochs=epochs, steps_per_epoch=train_steps, validation_data=validation_generator, verbose=2, callbacks=[earlystop_callback],)

    unfreeze_model(model)
    history = model.fit(
        train_generator, epochs=epochs, steps_per_epoch=train_steps, validation_data=validation_generator, verbose=2, callbacks=[earlystop_callback],)

    if not path.exists('results'):
        makedirs('results')

    plot_hist(history,  'accuracy')
    plot_loss(history,  'loss')

    y_true = np.argmax(dataset.Y_Train, axis=-1)
    # get proba predictions
    y_aux = model.predict(x=dataset.X_Train, verbose=2)
    y = []

    for pos in y_aux:  # convert probability to binary classes
        if(pos[0] > pos[1]):
            y.append(0)
        else:
            y.append(1)
    y_pred = np.array(y)

    try:
        print('Train Classification Report')
        cr = classification_report(
            y_true=y_true, y_pred=y_pred, target_names=dataset.class_indices)
        print(cr)
        print_classification_report(cr, 'train')
    except Exception as e:
        print(e)
        print('Error on print Train Classification Report')

    try:
        plot_confusion_matrix(
            y_true, y_pred, dataset.class_indices, _type='train')
    except:
        print('Error on plot Train Confusion Matrix')

    y_true = np.argmax(dataset.Y_Validation, axis=-1)
    # get proba predictions
    y_aux = model.predict(x=dataset.X_Validation, verbose=2)
    y = []

    for pos in y_aux:  # convert probability to binary classes
        if(pos[0] > pos[1]):
            y.append(0)
        else:
            y.append(1)
    y_pred = np.array(y)

    try:
        print('Validation Classification Report')
        cr = classification_report(
            y_true=y_true, y_pred=y_pred, target_names=dataset.class_indices)
        print(cr)
        print_classification_report(cr, 'validation')
    except:
        print('Error on print Classification Report')

    try:
        plot_confusion_matrix(
            y_true, y_pred, dataset.class_indices, _type='validation')
    except:
        print('Error on plot Validation Confusion Matrix')

    model.save_weights('results/first_try1.h5')
