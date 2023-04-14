import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_ds = pd.read_csv('train.csv').fillna('')
test_ds = pd.read_csv('test.csv').fillna('')

train_y = train_ds.pop('target')
train_x = train_ds[['keyword', 'location', 'text']].agg('-'.join, axis=1)
print(train_x)
test_x = test_ds[['keyword', 'location', 'text']].agg('-'.join, axis=1)
test_id = test_ds.pop('id').tolist()

vocab_size = 2000
embedding_dim = 32
max_length = 80
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
 
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_x)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(train_x)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type,
                       truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(test_x)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
                               padding=padding_type, truncating=trunc_type)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, input_length=max_length, output_dim=64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit(padded, train_y, epochs=10)


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()
  
#plot_graphs(history, "accuracy")
#plot_graphs(history, "loss")

predictions = model.predict(testing_padded)
for i, prediction in enumerate(predictions):
    predictions[i] = int(round(prediction[0]))

predictions = predictions.reshape(-1).astype(int)
predictions = pd.DataFrame({'id': test_id,
                           'target': predictions})

predictions.to_csv('predictions.csv', index=False)