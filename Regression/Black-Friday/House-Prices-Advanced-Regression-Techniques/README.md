# House-Prices-Advanced-Regression-Techniques
* Predict the final price of each home in Ames, Iowa with 79 explanatory variables describing every aspect of a residential home provided by Kaggle.  
* url : https://www.kaggle.com/c/house-prices-advanced-regression-techniques 
## Requirements
* Python 3.5 (or later)
* scikit-learn
## Methods for Training and Prediction
* ```Ridge```
* ```Lasso```
* ```Elastic Net```
* ```Kernel Ridge```
* ```Gradient Boosting```
* ```Light GBM```
* ```XGBoost```
## How to use
### Data
* train.csv : training set
* test.csv : test set
### Data Pre-processing
* Concatenated train and test data for data pre-processing
* NA imputation 
  * Imputed mean value or value with most frequent value
* Feature Engineering
  * TotalSF : total square footage of the house floor
  * RemodYr : Year Built - Remodeled Year
* Changed ordinal category values to numeric to utilize as rank score
* Generated correlation matrix to visualize multicollinearity among variables 
* Used VIF to check multicollinearity
  * Deleted variables : TotalSF, TotalBsmtSF, GarageQual, PoolQC
 * Dummified categorical variables and merged them to original dataframe
* Normalized target and skewed variables by using `np.log1p`
  * `log(1+x)`
### EDA
* Splitted data to train and test set
* Extracted significant variables using XGBoost feature scores that are same or higher than 10
* Visualized data distribution of top variables using scatter plot
  * LotFrontage
  * GrLivArea
  * LotArea
 * Removed outliers based on scatter plot
### User Defined Function 
* Printing R2 and RMSE score   
```python
def score(prediction, labels):
    print("R2: {}".format(r2_score(prediction, labels)))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(prediction, labels))))
```
* Printing scores for train and test sets  
```python
def train_test(estimator, X_train, X_test, y_train, y_test):
    pred_train = estimator.predict(X_train)
    print(estimator)
    print("[Train]")
    score(pred_train, y_train)
    pred_test = estimator.predict(X_test)
    print("[Test]")
    score(pred_test, y_test)
```
### Modeling
* Used modeling methods mentioned above in **"Methods for Training and Prediction"**
* Chose ```XGBoost``` and ```Gradient Boosting``` for ensemble models since they had the highest accuracy rate
```python
[XG Boost]
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05,
       max_delta_step=0, max_depth=3, min_child_weight=1.7817,
       missing=None, n_estimators=2200, n_jobs=1, nthread=-1,
       objective='reg:linear', random_state=7, reg_alpha=0.464,
       reg_lambda=0.8571, scale_pos_weight=1, seed=None, silent=1,
       subsample=0.5213)
[Train]
R2: 0.9430596762181811
RMSE: 0.08561899289266175
[Test]
R2: 0.8991042576549411
RMSE: 0.10735037018364177
Accuracy: 0.9005 (0.0093)
```
```python

[Gradient Boosting]
GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.05, loss='huber', max_depth=4,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=15, min_samples_split=10,
             min_weight_fraction_leaf=0.0, n_estimators=3000,
             n_iter_no_change=None, presort='auto', random_state=5,
             subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0,
             warm_start=False)
[Train]
R2: 0.9852069847552438
RMSE: 0.044632928649333964
[Test]
R2: 0.9072047178566619
RMSE: 0.10237130369726542
Accuracy: 0.8984 (0.0127)
```
* Final model
```python
fin_model = (np.exp(pred_xgb) *.8) + (np.exp(pred_gb) * .2)
```
  * Gave higher weight to ```XGBoost``` since it had a higher accuracy rate comparing to ```Gradient Boosting```
* Export to CSV
## Final Evaluation
![kaggle_leaderboard](https://user-images.githubusercontent.com/42960718/52994755-ef1f8780-345b-11e9-8e86-90409f611300.PNG)
* **[1559 / 4214]** as of 2019-02-19
