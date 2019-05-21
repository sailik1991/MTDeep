# Project      : MTDeep-- Boosting the Security of Deep Neural Nets Against Adversarial Attacks
#                with Moving Target Defense
# laboratory   : Yochan
# Last update  : 19th May, 2019
# username     : sailiks1991
# name         : Sailik Sengupta
# description  : Provides helper code for data processsing, model loading etc.

import keras

img_rows, img_cols = 28, 28
num_classes = 10

def process_data(x_train, y_train, x_test, y_test, toCategorical=True):
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    if toCategorical:
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

def reshape_input_image(x_train):
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_train /= 255
    return x_train

def save_model(model_name, model):
    model_json = model.to_json()
    with open("saved_models/{}.json".format(model_name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("saved_models/{}.h5".format(model_name))

def load_model(model_name):
    print("Trying to laod model from disk...")
    json_file = open('saved_models/{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("saved_models/{}.h5".format(model_name))
    print("Loaded model from disk...")
    return loaded_model

