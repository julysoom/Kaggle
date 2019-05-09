# UCI-Default-of-Credit-Card
Predict whether a client will have a default payment next month by using a data set provided by UCI.

url : https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
## Requirements
* Python 3.5 (or later)
* scikit-learn
* keras
## Methods for Training and Prediction
* ```Logistic Regression```
* ```Deep Neural Network```
## How to use
### Data
* default of credit card clients.csv
### Feature Engineering
* Default user rate in data : 22.12%
```python
df["DEFAULT"].value_counts()
6636/30000

0.2212
```
* Average bill statement and payment
```python
df["BILL_AVG"] = df[["BILL_AMT9", "BILL_AMT8", "BILL_AMT7", "BILL_AMT6", "BILL_AMT5", "BILL_AMT4"]].mean(axis=1)
df["PAY_AVG"] = df[["PAY_AMT9", "PAY_AMT8", "PAY_AMT7", "PAY_AMT6", "PAY_AMT5", "PAY_AMT4"]].mean(axis=
```
* Binning age
```python
def f(df) :
    if (df["AGE"] >= 20) & (df["AGE"] < 30) :
        return "20s"
    elif (df["AGE"] >= 30) & (df["AGE"] < 40) :
        return "30s"
    elif (df["AGE"] >= 40) & (df["AGE"] < 50) :
        return "40s"
    elif (df["AGE"] >= 50) & (df["AGE"] < 60) :
        return "50s"
    elif (df["AGE"] >= 60) & (df["AGE"] < 70) :
        return "60s"
    else :
        return "70s"

df["AGE_GROUP"] = df.apply(lambda df: f(df), axis=1)
```
* Checking age group default distribution
```python
df.groupby("AGE_GROUP")["DEFAULT"].value_counts()
AGE_GROUP  DEFAULT
20s        0.0        7421
           1.0        2197
30s        0.0        8962
           1.0        2276
40s        0.0        4979
           1.0        1485
50s        0.0        1759
           1.0         582
60s        0.0         225
           1.0          89
70s        0.0          18
           1.0           7
Name: DEFAULT, dtype: int64
```
  * Older age group tends to delay their payments
### Data Pre-processing
* Categorify age group by using ```LabelEncoder```
  * rate_20s → 0, rate_30s → 1, ..., rate_70s → 5 
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["AGE_GROUP"] = le.fit_transform(df["AGE_GROUP"])
```
* Changed all dataset values to float
### Model
* Splitting data to train and test
```python
X = df.drop("DEFAULT", axis=1)
scaler = MinMaxScaler(feature_range = (0, 1))
X = scaler.fit_transform(X)
Y = np_utils.to_categorical(df["DEFAULT"], 2) # binary variable - 0: non-default 1: default
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)
```
* Parameter
```python
NB_EPOCH = 15
BATCH_SIZE = 64
VERBOSE = 1
OPTIMIZER = RMSprop(lr=.001)
```
* Model
  * Repeated following code 5 times to manually train model
```python
model = Sequential()
model.add(Dense(512, input_shape=(X.shape[1],)))
model.add(Activation("elu"))
model.add(Dropout(.3))
    
model.add(Dense(256))
model.add(Activation("elu"))
model.add(Dropout(.2))
    
model.add(Dense(128))
model.add(Activation("elu"))
model.add(Dropout(.1))
    
model.add(Dense(64))
model.add(Activation("elu"))
model.add(Dropout(.05))

model.add(Dense(2))
model.add(Activation("softmax"))

keras.regularizers.l1_l2(l1=.01, l2=.01)
keras.initializers.he_normal
print(model.summary())

model.compile(loss=keras.losses.binary_crossentropy, optimizer=OPTIMIZER, metrics=["accuracy"])
    


history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                   epochs=NB_EPOCH, verbose=VERBOSE, validation_split=.01)

plt.figure(figsize=(5, 3))
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Trends of accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
pred_proba = model.predict(X_test)
pred = (pred_proba > 0.5).astype(int)
print("\n")
print("ACC: ", accuracy_score(Y_test, pred))
print("AUC: ", roc_auc_score(Y_test, pred_proba))
print("Log Loss: ", log_loss(Y_test, pred_proba))
print("\n")
print("Confusion Matrix: ")
print(confusion_matrix(Y_test[:,1], pred[:,1]))
print("\n")
print("Classification Report: ")
print(classification_report(Y_test, pred))
acc = []
acc.append(accuracy_score(Y_test, pred))
```
### Evaluation


![image](https://user-images.githubusercontent.com/42960718/53013952-59e7b780-348a-11e9-8c61-0dd664d5ad63.png)
### Performance
```python
print(np.mean(acc))
print(np.std(acc))

0.8202666666666667
0.00525631894855028
```


