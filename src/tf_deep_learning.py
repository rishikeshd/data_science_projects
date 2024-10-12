from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def tf_model(input_shape, hidden_layers, output_units, dropout_rate, learning_rate):
    """
        Create a TensorFlow sequential model with specified architecture.

        Args:
            input_shape (int): The shape of the input data (number of features).
            hidden_layers (list of int): A list containing the number of units for each hidden layer.
            output_units (int): The number of units in the output layer (classes).
            dropout_rate (float): The dropout rate to apply after each hidden layer to prevent overfitting.
            learning_rate (float): The learning rate for the Adam optimizer.

        Returns:
            model: A compiled TensorFlow Keras Sequential model ready for training.
        """

    model = Sequential()
    # add input layer
    model.add(Dense(units=hidden_layers[0], activation='relu', input_shape=(input_shape,)))

    # hidden layers in a loop
    for layer, num_units in enumerate(hidden_layers[1:]):
        model.add(Dense(units=num_units, activation='relu'))
        model.add(Dropout(dropout_rate))

    # Add the output layer
    model.add(tf.keras.layers.Dense(output_units, activation='softmax'))

    # Compile the model
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # Tune the learning rate
    model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def fit_and_evaluate(model, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate):
    """
        Fit the model on the training data and evaluate its performance on training and validation sets.

        Args:
            model: A TensorFlow Keras model to be trained and evaluated.
            X_train (array-like): Training feature data.
            y_train (array-like): Training target data (one-hot encoded).
            X_val (array-like): Validation feature data.
            y_val (array-like): Validation target data (one-hot encoded).
            epochs (int): The number of epochs to train the model.
            batch_size (int): The batch size for training.
            learning_rate (float): The learning rate used for training the model.

        Returns:
            tuple: A tuple containing the following:
                - model: The trained TensorFlow Keras model.
                - history: The history object containing training metrics.
                - train_accuracy (float): The accuracy of the model on the training data.
                - val_accuracy (float): The accuracy of the model on the validation data.
        """
    # early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)  # Stop if no improvement for 3 epochs

    # Fit the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stopping])
    # Evaluate the model
    train_loss, train_accuracy = model.evaluate(X_train, y_train)
    val_loss, val_accuracy = model.evaluate(X_val, y_val)

    return model, history, train_accuracy, val_accuracy