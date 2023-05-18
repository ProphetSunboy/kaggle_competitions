import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt

# load data
data = pd.read_csv("Iris.csv")

class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
def type_to_int(data):
    le = LabelEncoder()
    le.fit(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    data["Species"] = le.transform(data["Species"])
    return data

data = type_to_int(data).sample(frac=1, random_state=42)
train_data = data[:131].drop('Id', axis=1)
train_y = train_data.pop('Species')
test_data = data[131:]
test_y = test_data.pop('Species')
test_id = test_data.pop('Id')

print(train_data.head())
print(train_y.head())

inputs = tf.keras.Input(shape=train_data.shape[1])
x = tf.keras.layers.Dense(16, activation='relu')(inputs)
x = tf.keras.layers.Dense(8, activation='relu')(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-2,
        patience=3,
        verbose=2,
    )
]

history = model.fit(train_data, train_y, epochs=100, batch_size=16, callbacks=callbacks)

predictions = model.predict(test_data)
predictions_decoded = []

for i, prediction in enumerate(predictions):
    predictions_decoded.append(class_names[np.argmax(prediction)])


predictions_decoded = pd.DataFrame({'Id': test_id,
                           'Species': predictions_decoded})

predictions_decoded.to_csv('predictions.csv', index=False)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")