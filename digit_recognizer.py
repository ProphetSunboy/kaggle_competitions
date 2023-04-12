import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_ds = pd.read_csv('train.csv')
test_ds = pd.read_csv('test.csv') / 255

train_y = train_ds.pop('label')
train_x = train_ds / 255
  
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=256, input_shape=[784], activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=10, activation='softmax'),
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=15, verbose=1)
print("Finished training the model")

predictions = pd.DataFrame({'Label': map(np.argmax, model.predict(test_ds))})
predictions.index += 1

csv_data = predictions.to_csv('predictions.csv', index_label='ImageId')