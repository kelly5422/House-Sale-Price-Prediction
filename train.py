import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Read dataset
train_df=pd.read_csv('ntut-ml-regression-2020/train-v3.csv')
test_df=pd.read_csv('ntut-ml-regression-2020/test-v3.csv')
valid_df=pd.read_csv('ntut-ml-regression-2020/valid-v3.csv')

# drop id and price
#train_dataset = train_df.sample(frac=0.8, random_state=0)
#test_dataset = train_df.drop(train_dataset.index)

train_features = train_df.copy()
valid_features = valid_df.copy()
#test_features = test_dataset.copy()

train_labels = train_features.pop('price')
valid_labels = valid_features.pop('price')
#test_labels = test_features.pop('price')

train_features = train_features.drop(['id'], axis=1)
valid_features = valid_features.drop(['id'], axis=1)
train_features = train_features.drop(['sale_yr'], axis=1)
valid_features = valid_features.drop(['sale_yr'], axis=1)
train_features = train_features.drop(['sale_day'], axis=1)
valid_features = valid_features.drop(['sale_day'], axis=1)
#print(valid_features)

# normalization
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(128, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(16, activation='relu'),
      layers.Dense(8, activation='relu'),
      layers.Dense(8, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))

  return model

dnn_model = build_and_compile_model(normalizer)

dnn_model.summary()

history = dnn_model.fit(
    train_features, train_labels,
    batch_size=128,
    validation_data=(valid_features, valid_labels),
    verbose=1, epochs=180)

dnn_model.save('./my_model')
print('train finish!')