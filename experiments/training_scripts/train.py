import os
import argparse
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

max_features = 20000
maxlen = 400
embedding_dims = 300
filters = 250
kernel_size = 3
hidden_dims = 250


def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    # input data and model directory
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args, _ = parser.parse_known_args()
    return args


def get_train_data(train_dir):
    x_train = pd.read_csv(os.path.join(train_dir, 'x_train.csv'), header=None, index_col=None).values
    y_train = pd.read_csv(os.path.join(train_dir, 'y_train.csv'), header=None, index_col=None).values
    print('x train', x_train.shape, 'y train', y_train.shape)
    return x_train, y_train


def get_test_data(test_dir):
    x_test = pd.read_csv(os.path.join(test_dir, 'x_test.csv'), header=None, index_col=None).values
    y_test = pd.read_csv(os.path.join(test_dir, 'y_test.csv'), header=None, index_col=None).values
    print('x test', x_test.shape, 'y test', y_test.shape)
    return x_test, y_test


def get_model():
    embedding_layer = tf.keras.layers.Embedding(max_features,
                                                embedding_dims,
                                                input_length=maxlen)

    sequence_input = tf.keras.Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = tf.keras.layers.Dropout(0.2)(embedded_sequences)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(hidden_dims, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    preds = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(sequence_input, preds)


if __name__ == "__main__":
    args = parse_args()

    x_train, y_train = get_train_data(args.train)
    x_test, y_test = get_test_data(args.test)

    model = get_model()

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x_train,
                        y_train,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        validation_data=(x_test, y_test),
                        verbose=2)

    final_val_acc = history.history['val_acc'][-1]
    print('final validation accuracy:', final_val_acc)

    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    tf.contrib.saved_model.save_keras_model(model, args.model_dir)
