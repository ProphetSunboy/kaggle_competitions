import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import SGD

# load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


columns = [
            'MSZoning', 'LotArea', 'LotShape', 'Utilities',
            'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle',
            'OverallQual', 'OverallCond', 'YearBuilt', 'ExterQual',
            'ExterCond', 'Heating', 'HeatingQC', 'GrLivArea',
            'BedroomAbvGr', 'TotRmsAbvGrd', 'KitchenQual', 'GarageCars'
]

train_y = train_data['SalePrice']
train_x = train_data[columns]

test_id = test_data['Id']
test_data = test_data[columns]

print(train_data.head())
print(test_data.head())


def encode_cols(data, columns):
    for column in columns:
        le = LabelEncoder()
        le.fit(data[column].unique())
        data[column] = le.transform(data[column])
    return data

def normalize_numerical_cols(data, columns):
    for column in columns:
        data[column] /= data[column].max()
    return data

need_encode = [
            'MSZoning', 'LotShape', 'Utilities', 'Neighborhood',
            'Condition1', 'BldgType', 'HouseStyle', 'ExterQual',
            'ExterCond', 'Heating','HeatingQC', 'KitchenQual'
]

need_normalize = ['LotArea', 'YearBuilt', 'GrLivArea', 'Neighborhood']

train_x = encode_cols(train_x, need_encode)
test_data = encode_cols(test_data, need_encode)

train_x = normalize_numerical_cols(train_x, need_normalize)
test_data = normalize_numerical_cols(test_data, need_normalize)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, input_dim=train_x.shape[1], activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(train_x, train_y, epochs=100, batch_size=32)


predictions = model.predict(test_data)

predictions = predictions.reshape(-1)

predictions = pd.DataFrame({'Id': test_id,
                           'SalePrice': predictions})

predictions.to_csv('predictions.csv', index=False)