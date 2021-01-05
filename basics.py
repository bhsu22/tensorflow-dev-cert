import time
import os

import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_run_logdir():
    datetime = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    print(os.path.join("logs", datetime))
    return "logs/"


def get_sequential_classifier_model(shape):
    """
    Get a classification network using the sequential api.
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=list(shape)))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    return model


def train_image_classifier():
    """
    Train an sequential image classifier net.
    """
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.1
    )
    n_samples, sample_x, sample_y = X_train.shape
    print(X_train.shape)
    print(y_train)
    X_train, X_valid = X_train / 255.0, X_valid / 255.0

    model = get_sequential_classifier_model((sample_x, sample_y))

    # gets the model summary
    print(model.summary())

    # getting a specific layer and its weights.
    dense_layer_0 = model.get_layer("dense_1")
    print(dense_layer_0.get_weights())

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=get_run_logdir(), histogram_freq=1, profile_batch=0
    )
    # sparse categorical cross entroy: y = [1,0,2,3,0,...]
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"],
        callbacks=[tensorboard_callback],
    )

    # train the model
    model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))


def get_sequential_regression_model(shape):
    """
    Gets a regression network using sequential api.
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(100, input_shape=shape, activation="relu"))
    model.add(keras.layers.Dense(1, activation="linear"))
    return model


def get_functional_regression_model1(X):
    """
    Get a regression network using the functional API.
    """
    input_layer = keras.layers.Input(shape=X.shape[1])
    layer1 = keras.layers.Dense(30, activation="relu")(input_layer)
    layer2 = keras.layers.Dense(30, activation="relu")(layer1)
    concat = keras.layers.Concatenate()([input_layer, layer2])
    output = keras.layers.Dense(1, activation="softplus")(concat)
    model = keras.models.Model(inputs=[input_layer], outputs=[output])
    return model


def get_functional_regression_model2(X):
    """
    Get a regression network using the functional API taking two inputs.
    """
    X1, X2 = X
    input_layer1 = keras.layers.Input(shape=X1.shape[1])
    input_layer2 = keras.layers.Input(shape=X2.shape[1])
    layer1 = keras.layers.Dense(30, activation="relu")(input_layer1)
    layer2 = keras.layers.Dense(30, activation="relu")(layer1)
    concat = keras.layers.Concatenate()([layer2, input_layer2])
    output = keras.layers.Dense(1, activation="softplus")(concat)

    model = keras.models.Model(inputs=[input_layer1, input_layer2], outputs=[output])
    return model


def get_functional_regression_model3(X):
    """
    Get a regression network using the functional API taking two inputs
    and giving two outputs.
    """
    X1, X2 = X
    input_layer1 = keras.layers.Input(shape=X1.shape[1])
    input_layer2 = keras.layers.Input(shape=X2.shape[1])
    layer1 = keras.layers.Dense(30, activation="relu")(input_layer1)
    layer2 = keras.layers.Dense(30, activation="relu")(layer1)
    concat = keras.layers.Concatenate()([layer2, input_layer2])

    output = keras.layers.Dense(1, activation="softplus")(concat)
    aux_output = keras.layers.Dense(1, activation="softplus")(layer2)
    model = keras.models.Model(
        inputs=[input_layer1, input_layer2], outputs=[output, aux_output]
    )
    return model


def train_regression_model(
    sequential=False, functional1=False, functional2=False, functional3=False
):
    """
    Train a regresion model on the california housing dataset.
    """
    housing = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data, housing.target, test_size=0.1
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.1
    )
    n_samples, n_features = X_train.shape

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    if sequential:
        model = get_sequential_regression_model([n_features])
    elif functional1:
        model = get_functional_regression_model1(X_train)
    elif functional2:
        X_trainA, X_trainB = X_train[:, :6], X_train[:, 6:]
        X_validA, X_validB = X_valid[:, :6], X_valid[:, 6:]
        X_testA, X_testB = X_test[:, :6], X_test[:, 6:]

        X_train = (X_trainA, X_trainB)
        X_valid = (X_validA, X_validB)
        X_test = (X_testA, X_testB)

        model = get_functional_regression_model2(X_train)
    elif functional3:
        X_train = (X_train[:, :6], X_train[:, 6:])
        X_valid = (X_valid[:, :6], X_valid[:, 6:])
        X_test = (X_test[:, :6], X_test[:, 6:])

        y_train = [y_train, y_train]
        y_valid = [y_valid, y_valid]
        y_test = [y_test, y_test]

        model = get_functional_regression_model3(X_train)

    print(model.summary())
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=get_run_logdir())

    model.compile(
        loss=["mse", "mse"],
        optimizer="sgd",
        metrics=["mse"],
        callbacks=[tensorboard_callback],
    )
    model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid))


if __name__ == "__main__":
    train_image_classifier()
