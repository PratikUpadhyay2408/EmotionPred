import tensorflow as tf
import keras
import pandas as pd
import pathlib as p
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, TensorBoard
import os
from sklearn.model_selection import  train_test_split
from sklearn.metrics import f1_score


def plot_model_loss(model_history):
    # summarize history for loss
    f, ax = plt.subplots(figsize=(6, 6))
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join('Results', 'model_loss.png'))


def plot_model_accuracy(model_history):
    # summarize history for accuracy
    f, ax = plt.subplots(figsize=(6, 6))

    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join('Results', 'model_accuracy.png'))


def make_df_dataset( df ):

    X = df.drop(["label"], axis=1)
    Y = df["label"]
    x_train, x_test, y_train, y_test =  train_test_split(X, Y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def tf_make_csv_dataset(csv_file_path, cols):
    dataset = tf.data.experimental.make_csv_dataset(csv_file_path,
                                                    batch_size=250,
                                                    column_names=cols,
                                                    label_name="label" )
    dataset = dataset.map(pack_features_vector)
    return dataset


def pack_features_vector( features, labels ):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


def fcn_model(input_dim):
    # Define the neural network model
    model = Sequential()
    # Add input layer
    model.add(Dense(units=64, activation='relu', input_dim=input_dim))
    # Add hidden layers
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    # Add output layer
    model.add(Dense(units=5, activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def write_combined_df(data_dir, csv_name=None, num_instances=-1):
    # Create a list of delayed objects for reading Parquet files
    full_df = pd.DataFrame()
    for parquet_file in data_dir.glob('*.parquet'):
        print(f"reading {parquet_file}")
        temp_df = pd.read_parquet(parquet_file)
        if num_instances > 0:
            # stratified sampling of n examples from each subject.
            temp_df = temp_df.groupby("label").sample(n=num_instances, random_state=42)

        full_df = pd.concat([full_df, temp_df])

    feature_label_df = full_df.drop(["sid"], axis=1)  # Specify axis=1 to drop columns

    if csv_name:
        feature_label_df.to_csv(csv_name, index=False)  # Set index=False to avoid saving the DataFrame index

    return feature_label_df, len(feature_label_df)


if __name__ == "__main__":

    data_dir = p.Path("MergedData")
    csv_file_path = 'complete_combined_dataset.csv'

    sensordata_df, n_instances = write_combined_df(data_dir, num_instances=25000)
    train_size = int(0.7 * n_instances)
    val_size = int(0.15 * n_instances)
    test_size = int(0.15 * n_instances)

    columns = ["c_acc_x", "c_acc_y", "c_acc_z", "ecg", "emg", "c_eda", "c_temp", "resp",
                "w_acc_x", "w_acc_y", "w_acc_z", "w_eda", "bvp", "w_temp", "label"]

    # "label is not counted as input col"
    model = fcn_model( len(columns)-1 )

    # stop if validation loss isn't changing much, avoid overfitting.
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    tensorboard_callback = TensorBoard(log_dir='Logs/logs', histogram_freq=1, write_graph=True, write_images=True)

    print( "starting to learn")

    x_train, x_test, y_train, y_test = make_df_dataset(sensordata_df)

    training = model.fit(x=x_train,
                         y=y_train,
                         batch_size=64,
                         callbacks=[earlystopper,tensorboard_callback],
                         validation_split=0.3,
                         epochs=10 )
    plot_model_loss(training)
    plot_model_accuracy(training)

    # Predict the labels on the test set
    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred.argmax(axis=1), average='weighted')
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    model.save( os.path.join("Results", "FinalModel.h5") )