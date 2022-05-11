#0503
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris['data']
y = iris['target']

from sklearn.datasets import load_iris
## 분리 안한 경우. 정확도 100%
## GridSearch
from sklearn.model_selection import GridSearchCV

iris = load_iris()
data = X = iris['data']
label = y = iris['target']

X_train, X_test, y_train, y_test =  train_test_split(data, label, test_size=0.2, random_state= 121)

clf = DecisionTreeClassifier()

grid_dtree = GridSearchCV(clf, 
                         param_grid= {"max_depth": [1, 2, 3], "min_samples_split":[2, 3]}, 
                         cv= 3, refit=True)

grid_dtree.fit(X_train, y_train)

#grid_dtree.cv_results_
#
grid_dtree.best_params_

estimator = grid_dtree.best_estimator_

pred = estimator.predict(X_test)

accuracy_score(y_test, pred)

df = iris
# 표준화(Standardization)
import pandas as pd
f = lambda x : ( x - x.mean() ) / x.std()
df.iloc[:, :2].apply(f)

df = pd.DataFrame([[0, 1], [2, 2]])
df.apply(f)

df.std()

np.std(df.values)

#
from sklearn.preprocessing  import  StandardScaler
scaler = StandardScaler()
scaler.fit(df.iloc[:, :2])
scaler.transform(df.iloc[:, :2])
scaler.mean_


df1 = df.iloc[:, :2]
np.std(df1.values)  # 1.2183
np.mean(df1.values) #  1.625
df1.describe()


# 정규화(Normalization) : 0 ~ 1
g = lambda x : ( x - x.min() ) / (x.max() - x.min())
df.iloc[:, :2].apply(g)

from sklearn.preprocessing  import  MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df.iloc[:, :2])
scaler.transform(df.iloc[:, :2])

scaler.fit(X_train,y_train)
X = scaler.transform(X)
label = pd.get_dummies(label)
X_train, X_test, y_train, y_test =  train_test_split(X, label, test_size=0.2, random_state= 121)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

pred = clf.predict(X_test)
accuracy_score(y_test, pred)


##
# Null 처리 함수
from sklearn.preprocessing import LabelEncoder
def fillna(df):
    df['Age'].fillna(df['Age'].mean(),inplace=True)
    df['Cabin'].fillna('N',inplace=True)
    df['Embarked'].fillna('N',inplace=True)
    df['Fare'].fillna(0,inplace=True)
    return df

# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
    return df

# 레이블 인코딩 수행. 
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

from sklearn.base import BaseEstimator

class MyDummyClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        pred = np.zeros(( X.shape[0], 1))
        for i in range (X.shape[0]):
            if X['Sex'].iloc[i] == 1:
                pred[i] = 0
            else:
                pred[i] = 1
        return pred
    
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

titanic_df = pd.read_csv("C:/python/machine_Learning/pymldg-rev-master/2장/titanic_train.csv")
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis = 1)
X_titanic_df = transform_features(X_titanic_df)
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2 , random_state=0)

myclf = MyDummyClassifier()
myclf.fit(X_train,y_train)

mypredictions = myclf.predict(X_test)
accuracy_score(y_test, mypredictions)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class MyFakeClassifier(BaseEstimator):
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return np.zeros((len(X),1),dtype=bool)

digits = load_digits()

y = (digits.target == 7).astype(int)
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=11)

fakeclf = MyFakeClassifier()
fakeclf.fit(X_train,y_train)
fakepred = fakeclf.predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, fakepred)

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def get_clf_eval(y_test,pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    print('오차행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}'.format(accuracy,precision,recall))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

titanic_df = pd.read_csv("C:/python/machine_Learning/pymldg-rev-master/2장/titanic_train.csv")
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis = 1)
X_titanic_df = transform_features(X_titanic_df)
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2 , random_state=11)

lr_clf = LogisticRegression()

lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
get_clf_eval(y_test, pred)


### 피마 인디언 당뇨병 예측

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

diabet_data = pd.read_csv("C:/python/machine_Learning/pymldg-rev-master/3장/diabetes.csv")
print(diabet_data['Outcome'].value_counts())
diabet_data.head(3)

diabet_data.info()

# 피처 데이터 세트 X, 레이블 데이터 세트 y 를 추출
X = diabet_data.iloc[:, :-1]
y = diabet_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=156, stratify=y)

# 로지스틱 회귀로 학습, 예측 및 평가 수행

lr_clf = LogisticRegression()
lr_clf.fit(X_train,y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]

def get_clf_eval(y_test,pred, pred_proba):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
          F1:{3:.4f}, AUC:{4:.4f}'.format(accuracy,precision,recall,f1,roc_auc))


get_clf_eval(y_test, pred, pred_proba)

def precision_recall_curve_plot(y_test , pred_proba_c1):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출. 
    precisions, recalls, thresholds = precision_recall_curve( y_test, pred_proba_c1)
    
    # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 Plot 수행. 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary],label='recall')
    
    # threshold 값 X 축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    
    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend(); plt.grid()
    plt.show()

pred_proba_c1 = lr_clf.predict_proba(X_test)[:,1]
precision_recall_curve_plot(y_test, pred_proba_c1)

diabet_data.describe()

plt.hist(diabet_data['Glucose'], bins=10)

zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

total_count = diabet_data['Glucose'].count()

for feature in zero_features:
    zero_count = diabet_data[diabet_data[feature] == 0][feature].count()
    print('{0} 0 건수는 {1}, 퍼펙트는 {2:.2f} %'.format(feature, zero_count,100*zero_count/total_count))

# zero_features 리스트 내부에 저장된 개별 피처들에 대해서 0값을 평균값으로 대체
mean_zero_features = diabet_data[zero_features].mean()
diabet_data[zero_features]=diabet_data[zero_features].replace(0, mean_zero_features)

X = diabet_data.iloc[:,:-1]
y = diabet_data.iloc[:,-1]

# StandardScaler 클래스를 이용해 피처 데이터 세트에 일괄적으로 스케일링 적용
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2 , random_state=156, stratify=y)

# 로지스틱 회귀로 학습, 예측 및 평가 수행

lr_clf = LogisticRegression()
lr_clf.fit(X_train,y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:,1]

get_clf_eval(y_test, pred, pred_proba)

def get_clf_eval(y_test , pred):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}'.format(accuracy , precision ,recall))

from sklearn.preprocessing import Binarizer

def get_eval_by_threshold(y_test , pred_proba_c1, thresholds):
    # thresholds list객체내의 값을 차례로 iteration하면서 Evaluation 수행.
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1) 
        custom_predict = binarizer.transform(pred_proba_c1)
        print('임곗값:',custom_threshold)
        get_clf_eval(y_test , custom_predict)

thresholds = [0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.50]
pred_proba = lr_clf.predict_proba(X_test)
pred_proba_c1 = pred_proba[:,1].reshape(-1,1)

get_eval_by_threshold(y_test, pred_proba_c1 , thresholds)

binarizer = Binarizer(threshold=0.48)

pred_th_048 = binarizer.fit_transform(pred_proba_c1)

def get_clf_eval(y_test,pred, pred_proba):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
          F1:{3:.4f}, AUC:{4:.4f}'.format(accuracy,precision,recall,f1,roc_auc))


get_clf_eval(y_test, pred_th_048, pred_proba[:,1])

#########################################


import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

dt_clf = DecisionTreeClassifier(random_state=156)

iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2 , random_state=11)

dt_clf.fit(X_train,y_train)

from sklearn.tree import export_graphviz

export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names, \
                feature_names= iris_data.feature_names, impurity=True, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


### ensemble