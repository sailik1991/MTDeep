'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
import os
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod 
from cleverhans.utils_keras import KerasModelWrapper
import tensorflow as tf
from mnist_helper import process_data

batch_size = 128
num_classes = 10
epochs = 12

x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, 10))
train_params = {
    'nb_epochs': epochs,
    'batch_size': batch_size,
    'learning_rate': 0.001,
}

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

def get_vanilla_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

model = get_vanilla_model()
wrap = KerasModelWrapper(model)
fgsm_params = {'eps': 0.3}

def get_model(x_train, y_train, x_test, y_test):
    x_train, y_train, x_test, y_test = process_data(x_train, y_train, x_test, y_test)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', score[1])
    print(x_test.shape)
    y = model.predict(x_test)
    print(y)
    return model
   
def get_at_model(sess, x_train, y_train, x_test, y_test):
    model = get_vanilla_model
    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, model(x), x_test, y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, model(adv_x), x_test,
                              y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)

    model_train(sess, x, y, model(x), x_train, y_train,
                 predictions_adv=model(adv_x), evaluate=None,
                 args=train_params, save=False)

    print( "==== Model evaluation after training ====")
    evaluate_2()
    return model

def attack_model(sess, model, x_test, y_test):
    fgsm = FastGradientMethod(wrap, sess=sess)
    adv_x = fgsm.generate(x, **fgsm_params)
    adv_x = tf.stop_gradient(adv_x)

    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = keras.utils.to_categorical(y_test, 10)
    K.set_learning_phase(0)
    # Accuracy of the model on legitimate test inputs
    eval_params = {'batch_size': batch_size}
    accuracy_t = model_eval(sess, x, y, model(x), x_test, y_test,
                          args=eval_params)
    print('Test accuracy on legitimate examples: %0.4f' % accuracy_t)

    # Accuracy of the model on adversarial examples
    accuracy_a = model_eval(sess, x, y, model(adv_x), x_test,
                          y_test, args=eval_params)
    print('Test accuracy on adversarial examples: %0.4f' % accuracy_a)
    return (accuracy_t, accuracy_a) 
