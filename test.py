import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Read dataset
test_df=pd.read_csv('ntut-ml-regression-2020/test-v3.csv')
valid_df=pd.read_csv('ntut-ml-regression-2020/valid-v3.csv')
valid_features = valid_df.copy()
valid_labels = valid_features.pop('price')
valid_features = valid_features.drop(['id'], axis=1)
valid_features = valid_features.drop(['sale_yr'], axis=1)
valid_features = valid_features.drop(['sale_day'], axis=1)

# load model
reload_model = keras.models.load_model('./my_model')

test_results = {}
test_results['reload_model'] = reload_model.evaluate(valid_features, valid_labels, verbose=1)
print(pd.DataFrame(test_results, index=['Mean absolute error [price]']).T)

test_df = test_df.drop(['id'], axis=1)
test_df = test_df.drop(['sale_yr'], axis=1)
test_df = test_df.drop(['sale_day'], axis=1)
test_predictions = reload_model.predict(test_df).flatten()

id = np.array([])
for i in range(len(test_predictions)):
    id = np.append(id, i+1)

id = id.astype(int)

dict = {
    "id" : id,
    "price" : test_predictions
}

select_df = pd.DataFrame(dict)
select_df.to_csv("predict.csv", sep='\t',index=False)
