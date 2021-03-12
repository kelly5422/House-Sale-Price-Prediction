## Machine Learning@NTUT - Regression報告
House Sale Price Prediction Challenge

- 學生: 郭靜
- 學號: 108598068

---

## 做法說明
1. 分別將test,valid,test資料讀進來
2. 對資料做前處理,分割特徵與標籤
3. 對資料做正規化處理
4. 定義DNN模型
5. 訓練模型
6. 測試模型
7. 輸出csv檔


---

## 程式方塊圖與寫法

![](https://i.imgur.com/a1wQRtp.png)



#### 分別將 train valid test 資料讀進來
```
# Read dataset
train_df=pd.read_csv('ntut-ml-regression-2020/train-v3.csv')
test_df=pd.read_csv('ntut-ml-regression-2020/test-v3.csv')
valid_df=pd.read_csv('ntut-ml-regression-2020/valid-v3.csv')
```

#### 將 train valid 資料的特徵與標籤分開
#### 標籤為 price, 並把 id 從特徵中拿掉
```
train_features = train_df.copy()
valid_features = valid_df.copy()

train_labels = train_features.pop('price')
valid_labels = valid_features.pop('price')

train_features = train_features.drop(['id'], axis=1)
valid_features = valid_features.drop(['id'], axis=1)
```
#### 對資料做正規化
```
# normalization
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
```

#### 定義訓練模型
```
def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(128, activation='relu'),
      layers.Dense(128, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(8, activation='relu'),
      layers.Dense(8, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))

  return model
```

#### 開始訓練模型
```
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()
Model: "sequential_7"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
normalization_7 (Normalizati (None, 19)                39        
_________________________________________________________________
dense_45 (Dense)             (None, 128)               2560      
_________________________________________________________________
dense_46 (Dense)             (None, 128)               16512     
_________________________________________________________________
dense_47 (Dense)             (None, 64)                8256      
_________________________________________________________________
dense_48 (Dense)             (None, 64)                4160      
_________________________________________________________________
dense_49 (Dense)             (None, 8)                 520       
_________________________________________________________________
dense_50 (Dense)             (None, 8)                 72        
_________________________________________________________________
dense_51 (Dense)             (None, 1)                 9         
=================================================================
Total params: 32,128
Trainable params: 32,089
Non-trainable params: 39
```
```
history = dnn_model.fit(
    train_features, train_labels,
    batch_size=16,
    validation_data=(valid_features, valid_labels),
    verbose=1, epochs=100)
```

#### 測試預測結果
```
test_predictions = dnn_model.predict(test_df).flatten()
```
#### 輸出預測結果至csv檔
```
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
```

---

## 畫圖結果分析
* 下圖為訓練結果較不好的分析圖
訓練到最後fitting程度沒有很好
![](https://i.imgur.com/ppG604C.png)

* 下圖為訓練結果較好的分析圖
fitting程度較上圖好
![](https://i.imgur.com/H1uINvu.png)

---

## 討論預測值誤差很大的，是怎麼回事？
1. batch size與learning rate的值較難找到最佳值
2. epoch太大
3. features的欄位選擇不佳
4. DNN的層數與節點數設計不佳

---

## 如何改進？
1. batch size與learning rate可以多試不同數值
2. epoch的次數可以看fitting程度去做調整，若有使用到dropout要再增加一些
3. 針對features的欄位多去測試，有一些欄位看似關聯度不高，但其實還是影響很多
4. DNN的層數可以增加看看，節點數也可以多嘗試

---# House-Sale-Price-Prediction
