from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags

import constituent_models.mnist_cnn1 as mnist_cnn1
import logging
import os
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
from keras import backend as K
FLAGS = flags.FLAGS

import sys
from mnist_helper import *

model_type = 'EAT_'

def mnist_tutorial_cw(train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=True, nb_epochs=6,
                      batch_size=128, nb_classes=10, source_samples=10,
                      learning_rate=0.001, attack_iterations=100,
                      model_path=os.path.join("models", "mnist"),
                      targeted=True):
    """
    MNIST tutorial for Carlini and Wagner's attack
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param source_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :param model_path: path to the model file
    :param targeted: should we run a targeted attack? or untargeted?
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # MNIST-specific dimensions
    img_rows = 28
    img_cols = 28
    channels = 1

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    K.set_session(sess)

    set_log_level(logging.DEBUG)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)
    K.set_learning_phase(1)
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    models = {}
    preds = {}
    for model_name in ['mlp', 'cnn', 'hrnn']:
        try:
            print('[DEBUG] Loading model.')
            models[model_name] = load_model('{}{}'.format(model_type, model_name))
        except:
            print('[ERROR] Adversarially Trained models not found! Train and save strengthened models first. Then, run this.')
            exit(1)
        
        preds[model_name] = models[model_name](x)
    
    rng = np.random.RandomState([2017, 8, 30])

    # Evaluate the accuracy of the Adv trained MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy_test = ''
    attacks = {}

    # Make computations graphs for the attacks
    for model_name in models.keys():
        accuracy = model_eval(sess, x, y, preds[model_name], X_test, Y_test, args=eval_params)
        accuracy_test += '{} {}\n'.format(model_name, accuracy)

        # Instantiate a CW attack object
        wrap = KerasModelWrapper(models[model_name])
        attacks['$PGD_{}$'.format(model_name[0])] = ProjectedGradientDescent(wrap, sess=sess)

    # Make the output tensor for specification in the attacks parameters
    idxs = [np.where(np.argmax(Y_test, axis=1) == i)[0][0] for i in range(10)]
    if targeted:
        one_hot = np.zeros((10, 10))
        one_hot[np.arange(10), np.arange(10)] = 1

        adv_inputs = np.array([[instance] * 10 for instance in X_test[idxs]],
                              dtype=np.float32)
        adv_inputs = adv_inputs.reshape((100, 28, 28, 1))
        adv_ys = np.array([one_hot] * 10, dtype=np.float32).reshape((100, 10))
        yname = "y_target"
    else:
        adv_inputs = X_test[idxs]
        adv_ys = None
        yname = "y"

    attack_params = { 'eps': 0.3, yname: adv_ys, 'eps_iter': 0.05 }
    
    table_header = '{}model '.format(model_type)
    accuracy_attack = ''
    
    for model_name in models.keys():
        
        accuracy_attack += '{} '.format(model_name)

        # For each model, apply all attacks
        for attack_name in attacks.keys():
            print('[DEBUG] Attacking {} using {}.'.format(model_name, attack_name))

            # Code brach entered only once for creating the table header with attack names
            if attack_name not in table_header:
                table_header += '{} '.format(attack_name)

            adv = attacks[attack_name].generate_np(adv_inputs, **attack_params)
            if targeted:
                adv_accuracy = model_eval(sess, x, y, preds[model_name], adv, adv_ys, args={'batch_size': 10})
            else:
                adv_accuracy = model_eval(sess, x, y, preds[model_name], adv, Y_test[idxs], args={'batch_size': 10})

            accuracy_attack += '{} '.format(adv_accuracy * 100)

        # Move on to attack the next model    
        accuracy_attack += '\n'

    print(table_header)
    print(accuracy_attack)
    print(accuracy_test)
    
    # Close TF session
    sess.close()

    # Finally, block & display a grid of all the adversarial examples
    return report


def main(argv=None):
    mnist_tutorial_cw(viz_enabled=FLAGS.viz_enabled,
                      nb_epochs=FLAGS.nb_epochs,
                      batch_size=FLAGS.batch_size,
                      nb_classes=FLAGS.nb_classes,
                      source_samples=FLAGS.source_samples,
                      learning_rate=FLAGS.learning_rate,
                      attack_iterations=FLAGS.attack_iterations,
                      model_path=FLAGS.model_path,
                      targeted=FLAGS.targeted)


if __name__ == '__main__':
    flags.DEFINE_boolean('viz_enabled', False, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 100, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_string('model_path', os.path.join("models", "mnist"),
                        'Path to save or load the model file')
    flags.DEFINE_integer('attack_iterations', 1000,
                         'Number of iterations to run attack; 1000 is good')
    flags.DEFINE_boolean('targeted', False,
                         'Run the tutorial in targeted mode?')

    tf.app.run()

    '''
    --------------------------------------
    Avg. rate of successful adv. examples 0.2000
    Avg. L_2 norm of perturbations 0.3456
    '''
