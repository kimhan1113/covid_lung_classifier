import argparse

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Permute
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

# thirdparty libraries that requires extra installations
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow.keras as keras
from imutils import paths

import os
import collections
import time
import glob


# keras.backend.set_image_data_format('channels_first')
def load_data_(data_path='dataset', dim=128):
    images = []
    # print(data_path)
    for i in range(len(NUM_CLASSES)):
        # print("class : {} # of images : {}".format(NUM_CLASSES[i],
        #                                            len(glob.glob(os.path.join(data_path, '{}/*.*'.format(NUM_CLASSES[i]))))))
        images += glob.glob(os.path.join(data_path, '{}/*.*'.format(NUM_CLASSES[i])))
        images += list(paths.list_images(os.path.join(data_path, '{}'.format(NUM_CLASSES[i]))))
    num_images = len(images)

    if num_images == 0:
        raise RuntimeError('no images found in {}'.format(data_path))

    x = np.zeros((num_images, dim, dim, 3))
    y = np.zeros((num_images, len(NUM_CLASSES)))
    labels = []
    for i, image in enumerate(images):
        image_arr = cv2.imread(image)
        image_resized = cv2.resize(image_arr, (dim, dim))
        x[i, :, :, :] = image_resized
        label = image.split(os.path.sep)[-2]
        labels.append(label)
        label = np.array(labels)
        # cls = int(os.path.dirname(image).split('/')[-1]) - 1
        lb = LabelBinarizer()
        labelss = lb.fit_transform(label)
        y = to_categorical(labelss)

        # y[i, labelss] = 1

    return x, y


def load_data(data_path=['dataset'], dim=128):

    if not isinstance(data_path, list):
        raise RuntimeError('you should feed path as list type')
    if len(data_path) == 0:
        raise RuntimeError('Please feed at least one directory')
    xs = []
    ys = []

    for path in data_path:
        try:
            x, y = load_data_(data_path=path, dim=dim)
            xs.append(x)
            ys.append(y)
        except Exception as e:
            print(e)
    return np.concatenate(xs, axis=0).astype(np.float64) / 255.0, np.concatenate(ys, axis=0).astype(np.float64)


def classifier(dim):


    transfer_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    transfer_model.trainable = False
    model = Sequential()
    model.add(transfer_model)
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    for layer in model.layers:
        print(layer.output_shape)
    # initiate RMSprop optimizer
    model_json = model.to_json()
    with open(os.path.join(CKPT_DIR, "model.json"), "w") as json_file:
        json_file.write(model_json)
    return model


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='train solder joint classifier')
    parser.add_argument('--name', dest='name',
                        help='name of directory to save model',
                        required=False,
                        default='weight', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    dim = 128
    NUM_CLASSES = ['covid', 'normal']
    CKPT_DIR = os.path.join('output',
                            'ckpts',
                            str(dim),
                            args.name)
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)

    batch_size = 2
    epochs = 10

    # train_dirs = ['data/train/train',
    #               'data/train/2019.11.20_BoschQFP_Raw',
    #               'data/train/2019.11.21_EControl1.5CBI_Raw',
    #               'data/train/2019.12.27_EControl2ndCBI_Singan',
    #               'data/test/test'  # NOTE : test data is included in train db
    #               ]

    train_dirs = ['dataset']
    test_dirs = ['dataset_test']

    x_train, y_train = load_data(data_path=train_dirs, dim=dim)
    x_test, y_test = load_data(data_path=test_dirs, dim=dim)

    # Let's train the model using RMSprop
    model = classifier(dim)
    opt = keras.optimizers.RMSprop(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.1,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        fill_mode='nearest',
        cval=0.)

    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        # callbacks=[keras.callbacks.ModelCheckpoint(CKPT_DIR + '/weights.{epoch:02d}-{val_accuracy:.2f}.hdf5',
                                                                   # monitor='val_accuracy',
                                                                   # save_best_only=False, verbose=1,
                                                                   # save_weights_only=True, period=1)],
                        validation_data=(x_test, y_test),
                        verbose=1)

    model.save(args["model"], save_format="h5")


    # --------------------------------------------------------------------------------

    '''
    predIdxs = model.predict(testX, batch_size=BS)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)

    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs,
                                target_names=lb.classes_))

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

    # plot the training loss and accuracy
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on COVID-19 Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])

    # serialize the model to disk
    print("[INFO] saving COVID-19 detector model...")
    
    model.save('covid19_3.model', save_format="h5")
    '''