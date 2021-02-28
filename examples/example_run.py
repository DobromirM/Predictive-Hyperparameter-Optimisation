from keras.utils import np_utils

from boarml.architectures import CnnBaseArchitecture
from boarml.generators import ModelGenerator
import tensorflow as tf
import numpy as np

from examples.helper import define_hp_searcher_normal, perform_random_search
import time

# Preparing the data
from predictiveopt import PredictiveHyperOpt


def get_image_data(data_set, num_classes):
    if data_set == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif data_set == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        raise Exception('Invalid Dataset!')

    if len(x_train.shape) > 3:
        train_channels = 3
    else:
        train_channels = 1

    if len(x_test.shape) > 3:
        test_channels = 3
    else:
        test_channels = 1

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], train_channels)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], test_channels)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    num_classes = 10
    epochs = 100
    batch_size = 512

    x_train, y_train, x_test, y_test = get_image_data('mnist', num_classes)

    shape = x_train.shape[1:]

    # Creating the architecture from file
    arch = CnnBaseArchitecture(shape, 10)
    arch = arch.build_from_file('architecture.txt')

    # Creating the generator
    generator = ModelGenerator(arch, 'keras', removal_rate=0.2, duplication_rate=0.2, amendment_rate=0.5)

    # Performing Predictive Hyper Opt
    hp_searcher = PredictiveHyperOpt(total_models_count=100, full_models_count=10, final_epoch=epochs,
                                     partial_epochs=[2, 1], final_models_count=10,
                                     batch_size=batch_size)

    start_time = time.time()
    accuracy, architecture = hp_searcher.run(generator, x_train, y_train, x_test, y_test)
    pred_opt_time = time.time() - start_time

    print(f'Predictive opt finished in {pred_opt_time}')

    # Performing Random Search
    generator_for_random = ModelGenerator(arch, 'keras', removal_rate=0.2, duplication_rate=0.2, amendment_rate=0.5)
    random_search_accuracy = perform_random_search(generator, x_train, y_train, x_test, y_test,
                                                   total_time=pred_opt_time,
                                                   epochs=epochs,
                                                   batch_size=batch_size)

    print('\n\n\n\n\n')
    print(f'Accuracy for Random Search: {random_search_accuracy}%')

    print(f'Accuracy for Predictive Hyper Opt: {accuracy}%')
    print('\n')
    print(architecture)
