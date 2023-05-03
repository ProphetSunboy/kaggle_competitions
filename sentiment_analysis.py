import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np


train_ds = pd.read_csv('labeledTrainData.csv')
train_x = train_ds['review']
train_y = train_ds['sentiment']
test_ds = pd.read_csv('testData.csv')
test_x = test_ds.pop('review')
test_id = test_ds.pop('id').tolist()

VOCAB_SIZE = 3000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_ds['review'])
print(test_ds.isna().any())
print(train_x.shape)
print(test_x.head())

model_multi_lstm = tf.keras.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),
    encoder,
    tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_multi_lstm.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model_multi_lstm.fit(train_x, train_y, epochs=10)


predictions = model_multi_lstm.predict(test_x)
for i, prediction in enumerate(predictions):
    predictions[i] = int(round(prediction[0]))

predictions = predictions.reshape(-1).astype(int)
print(predictions)
predictions = pd.DataFrame({'id': test_id,
                           'sentiment': predictions})

predictions.to_csv('predictions.csv', index=False)