import sys
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback


class ConfusionMatrixCallback(Callback):
    def __init__(self, x_val, y_val):
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.x_val)
        pred_labels = predictions.argmax(axis=1)
        true_labels = self.y_val.argmax(axis=1)
        conf_matrix = confusion_matrix(true_labels, pred_labels)

        n_correct = np.sum(true_labels == pred_labels)
        accuracy = round(n_correct / len(pred_labels), 4)

        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title(f"Confusion Matrix - Epoch {epoch+1}, Accuracy: {accuracy}")
        plt.savefig(f"model-{epoch+1}.png")
        plt.close()


# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

# convert the labels to a 1-hot encoding
encoder = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_train = encoder.fit_transform(y_train)
y_test = y_test.reshape(-1, 1)
y_test = encoder.fit_transform(y_test)

# Create an instance of the custom callback
conf_matrix_callback = ConfusionMatrixCallback(x_test, y_test)


# TODO: evaluate different activation functions (relu, tanh, sigmoid, selu, elu)
activation = "tanh"

# TODO: evaluate different numbers of filters
n_filters = 8

# TODO: evaluate different number of neurons in dense layer
n_neurons_in_dense_layer = 16

# TODO: evaluate optimzers (sgd, rmsprop, adagrad, adam)
optimzer = "sgd"

model = Sequential([
    Conv2D(n_filters, (5, 5), activation=activation, input_shape=(28, 28, 1)),
    MaxPooling2D(),
    Conv2D(n_filters, (5, 5), activation=activation),
    MaxPooling2D(),
    Flatten(),
    Dense(n_neurons_in_dense_layer, activation=activation),
    Dense(10, activation="softmax")
])

# output the model summary
with open("model.txt", "w") as f:
    # Temporarily set stdout to the file object
    original_stdout = sys.stdout
    sys.stdout = f
    model.summary()
    # Reset stdout back to original
    sys.stdout = original_stdout

# don't change anything below here.
model.compile(loss="categorical_crossentropy", optimizer=optimzer, metrics=["accuracy"])
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=200,
    callbacks=[conf_matrix_callback]
)
