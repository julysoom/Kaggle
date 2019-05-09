# UCI-Heart-Disease
Predict whether the observed patient might have heart disease based on the patient's medical records.

url : https://www.kaggle.com/ronitf/heart-disease-uci 
## Requirements
* Python 3.5 (or later)
* scikit-learn
* graphviz
## Methods for Training and Prediction
* ```Random Forest```
## How to use
### Data
* heart.csv
### Data Explanation
* `age`
* `sex` : 1 = male, 0 = female
* `cp` : chest pain type ; 1 = typical  angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic
* `trestbps` : resting blood pressure
* `cholserum` : cholestrol
* `fbs` : fasting blood suga ; if `fbs` > 120 mg/dl, 1 = true, 0 = false
*  `restecg` : resting electrocardiographic results ; 0 = normal, 1 = having ST=T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria
* `thalach` : maximum heart rate achieved
* `exang` : exercise induced angina ; 1 = yes, 0 = no
* `oldpeak` : ST depression induced by exercise relative to rest
* `slope` : the slope of the peak exercise ST segment ; 1 = upsloping, 2 = flat, 3 = downsloping
* `ca` : the number of major vessels (0 to 3) colored by flourosopy
* `tha` : a blood disorder called thalassemia ; 3 = normal, 6 = fixed defect, 7 = reversable defect
* `target` : heart disease ; 1 = yes, 0 = no
### Researched Medical Information
* According to the research, high cholestrol, high blood pressure, diabetes, weight, family history, and smoking will have critical impact on whether getting a heart disease
* Invariable variables : `age`, `sex` (male) , and family history
* Thalassemia is a rare anemia disease that is hereditary (highly influenced by family history)
* Number of major vessels : more number of major vessels indicate healthy heart condition
  * Heart diseases occure when blood vessels are clogged by fats, causing abnormal blood circulation
### Data Pre-processing
* Data 

![image](https://user-images.githubusercontent.com/42960718/57395600-78e12480-7203-11e9-9944-c61e4d8e1243.png)

* Changed the name of the columns and categorical data to add intuition

![image](https://user-images.githubusercontent.com/42960718/57395777-ed1bc800-7203-11e9-8602-9c100a5f41d4.png)

* Dummifying Data
```python
# drop_first=True 로 설정하면 중복되는 정보값에 더미 생성이 안됨
# 성별의 경우, female 과 male 칼럼을 각각 더미해서 생성하는게 아니라, male = 1 or male = 0 생성
df = pd.get_dummies(df, drop_first=True)
```
![image](https://user-images.githubusercontent.com/42960718/57397234-31f52e00-7207-11e9-866a-8cb54f1ed02b.png)

### Model
#### Splitting data to train and test
```python
X_train, X_test, y_train, y_test = train_test_split(df.drop("target", 1),
                                                   df["target"],
                                                   test_size=.2,
                                                   random_state=42)
```
* `Random Forest`
```python
# 랜덤포레스트랑 의사결정 나무 모델 돌리기
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz # 트리 모델 구조 시각화

# 데이터 결과 성능 측정
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
```
#### Grid-search
* Current model parameter
```python
from pprint import pprint
rf_clf = RandomForestClassifier()
print("Parameters currently in use:\n")
pprint(rf_clf.get_params())
# n_estimators = 랜덤포레스트 내 의사결정 나무 갯수
# max_features = 노트 (가지) 로 나눌 때 최대 고려 변수 (features)
# max_depth = 각 의사결정 나무 깊이 (층)
```
```python
Parameters currently in use:

{'bootstrap': True,
 'class_weight': None,
 'criterion': 'gini',
 'max_depth': None,
 'max_features': 'auto',
 'max_leaf_nodes': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 'warn',
 'n_jobs': None,
 'oob_score': False,
 'random_state': None,
 'verbose': 0,
 'warm_start': False}
 ```
 * RandomizedSearchCV
   * Faster processing speed compared to GridSearchCV
```python
   from sklearn.model_selection import RandomizedSearchCV
   rf_grid = {"n_estimators": [10, 20, 50, 100],
          "max_features": ["auto", "sqrt"],
          "max_depth": [3, 5, 10, 15]}
   pprint(rf_grid)
```
```python
rf_clf_grid = RandomizedSearchCV(estimator=rf_clf, param_distributions=rf_grid,
                                n_iter=20, cv=3, verbose=2, random_state=42,
                                n_jobs=-1)
rf_clf_grid.fit(X_train, y_train)
```
```python
RandomizedSearchCV(cv=3, error_score='raise-deprecating',
          estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False),
          fit_params=None, iid='warn', n_iter=20, n_jobs=-1,
          param_distributions={'n_estimators': [10, 20, 50, 100], 'max_features': ['auto', 'sqrt'], 'max_depth': [3, 5, 10, 15]},
          pre_dispatch='2*n_jobs', random_state=42, refit=True,
          return_train_score='warn', scoring=None, verbose=2)
```
#### User-defined function for performance difference after adjusting parameter
```python
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print("Average Error: {:0.4f} degrees.".format(np.mean(errors)))
    print("Accuracy = {:0.2f}%.".format(accuracy))
    return accuracy
```
```python
base_model = RandomForestClassifier(n_estimators=5, random_state=42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)
```
```python
best_random = rf_clf_grid.best_estimator_
best_random.fit(X_train, y_train)
random_accuarcy = evaluate(best_random, X_test, y_test)
```
### Evaluation
#### Visualization
* Tree plot
```python
estimator = best_random.estimators_[1] # estimator 에 파라미터 지정
feature_names = [i for i in X_train.columns]
y_train_str = y_train.astype("str")
y_train_str[y_train_str=="0"] = "no disease"
y_train_str[y_train_str=="1"] = "disease"
y_train_str = y_train_str.values
```
```python
# 시각화
export_graphviz(estimator, out_file="tree.dot", 
                feature_names = feature_names,
                class_names = y_train_str,
                rounded = True, proportion = True, 
                label="root",
                precision = 2, filled = True)

from subprocess import call
call(["dot", "-Tpng", "tree.dot", "-o", "tree.png", "-Gdpi=600"])

from IPython.display import Image
Image(filename = "tree.png")
```
![image](https://user-images.githubusercontent.com/42960718/57399142-a7fb9400-720b-11e9-859f-3afb89070a9a.png)
#### Confusion Matrix
```python
# Confusion Matrix (혼동행렬)
y_predict = best_random.predict(X_test)
# pred_proba: 각 분류값이 올바르게 분류된 확률
y_pred_proba = best_random.predict_proba(X_test)[:, 1]
y_pred_bin = best_random.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred_bin)
confusion_matrix
```
```python
# Result of Confusion Matrix
array([[25,  4],
       [ 5, 27]], dtype=int64)
```
* Sensitivity & Specificity
```python
total = sum(sum(confusion_matrix))

# 민감도: 질환에 실제 걸린 사람이 검사 받았을 때 양성을 판정 받는 비율
sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print("Sensitivity : ", sensitivity )

# 특이도: 질환에 걸리지 않은 정상인이 음성을 판정 받는 비율
specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print("Specificity : ", specificity)
```
### ROC Curve
```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams["font.size"] = 12
plt.title("ROC curve for diabetes classifier")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.grid(True)
```
![image](https://user-images.githubusercontent.com/42960718/57399302-f27d1080-720b-11e9-9fbd-695f006eb045.png)
#### Extracting Variable Importance
* Permutation Importance : Measuring the weight of each variables on accuracy by randomly shuffling data within the machine learning model
```python
import eli5 # for permutation importance
from eli5.sklearn import PermutationImportance
```
```python
pi = PermutationImportance(best_random, random_state=34).fit(X_test, y_test)
eli5.show_weights(pi, feature_names=X_test.columns.tolist())
```
![image](https://user-images.githubusercontent.com/42960718/57399481-45ef5e80-720c-11e9-83b5-956e69968862.png)










  
                                               
                                                  



