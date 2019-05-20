'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.optimizers import RMSprop
from mnist_helper import process_data
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
import tensorflow as tf

batch_size = 128
num_classes = 10
epochs = 20

x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, 10))
train_params = {
    'nb_epochs': epochs,
    'batch_size': batch_size,
    'learning_rate': 0.001,
}

img_rows=28
img_cols=28
input_shape = (img_rows, img_cols, 1)

def get_vanilla_model():
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    return model

def get_model(x_train, y_train, x_test, y_test):
    x_train, y_train, x_test, y_test = process_data(x_train, y_train, x_test, y_test)
    model = get_vanilla_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', score[1])
    return model

def get_at_model(sess, x_train, y_train, x_test, y_test):
    x_train, y_train, x_test, y_test = process_data(x_train, y_train, x_test, y_test)
    model = get_vanilla_model()
    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.3}
    adv_x = fgsm.generate(x, **fgsm_params)
    adv_x = tf.stop_gradient(adv_x)
    """
    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, model_at(x), x_test, y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, model_at(adv_x_at), x_test,
                              y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
    """
    model_train(sess, x, y, model(x), x_train, y_train,
                 predictions_adv=model(adv_x), evaluate=None,#evaluate_2,
                 args=train_params, save=False)
    eval_par = {'batch_size': batch_size}
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = keras.utils.to_categorical(y_test, 10)
    acc = model_eval(sess, x, y, model(x), x_test, y_test, args=eval_par)
    print('Test accuracy on test examples: %0.4f\n' % acc)
    acc = model_eval(sess, x, y, model(adv_x), x_test, y_test, args=eval_par)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)

    return model_at
