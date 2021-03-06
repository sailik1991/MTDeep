# Project      : MTDeep-- Boosting the Security of Deep Neural Nets Against Adversarial Attacks
#                with Moving Target Defense
# laboratory   : Yochan
# Last update  : 19th May, 2019
# username     : sailiks1991
# name         : Sailik Sengupta
# description  : Adversarially trains a set of models and saves them.

import sys
from mnist_helper import *

import keras
import tensorflow as tf
import cleverhans

import constituent_models.mnist_cnn1 as mnist_cnn1
import constituent_models.mnist_hierarchical_rnn as mnist_hierarchical_rnn
import constituent_models.mnist_irnn as mnist_irnn
import constituent_models.mnist_mlp as mnist_mlp

# --- Fast Gradient Method ----
def get_FGM_cg(sess, wrap, x):
    attack = cleverhans.attacks.FastGradientMethod(wrap, sess=sess)
    attack_params = {'eps': 0.3}
    adv_x = attack.generate(x, **attack_params)
    # adv_x = tf.stop_gradient(adv_x)
    return adv_x

def main(sess):
    # ----- Get train and test data -----
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # ----- Get models for the ensemble -----
    models = {}
    for model_name in ['mlp', 'cnn', 'hrnn']:
        try:
            print('[DEBUG] Loading model.')
            models[model_name] = load_model(model_name)
        except:
            print('[DEBUG] Loading failed. Trying to train the constituent model.')
            models = get_trained_models(x_train, y_train, x_test, y_test)

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # --- Ensemble Adversarial training of constituent networks ---
    x_train, y_train, x_test, y_test = process_data(x_train, y_train, x_test, y_test)
    print('[DEBUG] Adversarial Training of the specified model against an attack.')
    adv_train_params = {
            'nb_epochs': 5,
            'batch_size': 128,
            'learning_rate': 0.001
    }

    adv_xs = {}
    for model_name in models.keys():
        wrap = cleverhans.utils_keras.KerasModelWrapper(models[model_name])
        adv_xs[model_name] = get_FGM_cg(sess, wrap, x)

    for model_name in models.keys():
        for attack_name in adv_xs.keys():
            cleverhans.utils_tf.model_train(sess, x, y, models[model_name](x), x_train, y_train, predictions_adv=models[model_name](adv_xs[attack_name]), evaluate=None, args=adv_train_params, save=False)
        save_model('EAT_{}'.format(model_name), models[model_name])

    print('[DEBUG] Adversarially trained models saved.')
    return

if __name__ == '__main__':
    sess = tf.Session()
    keras.backend.set_session(sess)
    main(sess)
