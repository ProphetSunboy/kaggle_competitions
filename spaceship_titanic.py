import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import SGD

# Load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

def drop_not_concerned_columns(data, columns):
    return data.drop(columns, axis=1)


not_concerned_columns = ["PassengerId", "Name", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "RoomService", "Cabin"]
submit_id = test_data["PassengerId"]
train_data = drop_not_concerned_columns(train_data, not_concerned_columns)
test_data = drop_not_concerned_columns(test_data, not_concerned_columns)


# Fixing Nan values
def fix_na(*args):
    for data in args:
        data["HomePlanet"] = data["HomePlanet"].fillna("Earth")
        data["CryoSleep"] = data["CryoSleep"].fillna(False)
        data["Destination"] = data["Destination"].fillna("TRAPPIST-1e")
        data["Age"] = data["Age"].fillna(0)
        data["VIP"] = data["VIP"].fillna(False)

    return args

train_data, test_data = fix_na(train_data, test_data)


# Normalize
def destination_enc(data):
    le = LabelEncoder()
    le.fit(["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
    data["Destination"] = le.transform(data["Destination"])
    return data


train_data = destination_enc(train_data)
test_data = destination_enc(test_data)


def planet_to_int(data):
    le = LabelEncoder()
    le.fit(["Earth", "Europa", "Mars"])
    data["HomePlanet"] = le.transform(data["HomePlanet"])
    return data


train_data = planet_to_int(train_data)
test_data = planet_to_int(test_data)


def bool_to_int(data, columns):
    le = LabelEncoder()
    le.fit([False, True])
    for column in columns:
        data[column] = le.transform(data[column])
    return data


bool_columns = ['CryoSleep', 'VIP', 'Transported']
test_bool = ['CryoSleep', 'VIP']
train_data = bool_to_int(train_data, bool_columns)
test_data = bool_to_int(test_data, test_bool)


def normalize_age(data):
    ss = StandardScaler()
    data["Age"] = ss.fit_transform(data["Age"].values.reshape(-1, 1))
    return data


train_data = normalize_age(train_data)
test_data = normalize_age(test_data)


def split_valid_test_data(data, fraction=0.8):
    data_y = data["Transported"]
    data_x = data.drop(["Transported"], axis=1)

    train_valid_split_idx = int(len(data_x) * fraction)
    train_x = data_x[:train_valid_split_idx]
    train_y = data_y[:train_valid_split_idx]

    valid_test_split_idx = (len(data_x) - train_valid_split_idx) // 2
    test_x = data_x[train_valid_split_idx + valid_test_split_idx:]
    test_y = data_y[train_valid_split_idx + valid_test_split_idx:]

    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = split_valid_test_data(train_data)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, input_dim=train_x.shape[1], activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

sgd = SGD(learning_rate=0.001)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=100,
                    batch_size=32, validation_data=(test_x, test_y))


predictions = model.predict(test_data)

for i, prediction in enumerate(predictions):
    predictions[i] = bool(round(prediction[0]))

predictions = predictions.reshape(-1).astype(bool)
predictions = pd.DataFrame({'PassengerId': submit_id,
                           'Transported': predictions})

predictions.to_csv('predictions.csv', index=False)


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.plot(history.history['val_'+string])
  plt.ylabel('val_'+string)
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")