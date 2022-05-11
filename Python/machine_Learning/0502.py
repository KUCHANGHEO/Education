import numpy as np

arr = np.array([3,1,9,5])

# np.sort
np.sort(arr)


arr.sort()


idx = np.argsort(arr)

arr[idx]



import pandas as pd
titanic = pd.read_csv("C:/python/machine_Learning/pymldg-rev-master/1장/titanic_train.csv")
titanic['Pclass']
titanic.value_counts()

titanic[['Pclass',"Sex"]]

titanic[['Pclass',"Sex"]].value_counts()

titanic = titanic.set_index("Pclass")
titanic

titanic.loc[3,:]

# ndarray, 리스트, 딕셔너리에서 데이터 프레임 만들기
# 데이터 프레임에서 ndarray, 리스트, 딕셔너리

df = titanic.iloc[:4,:4]
df

df.iloc[:,:2]
df.iloc[:, 2:4]

df.values

df.to_dict()

df.values.tolist()

df = pd.DataFrame(
    {'Passengerld':[1,2,3,4],
     'Survived':[0,1,1,1],
     'Sex':['male','female','female','female']})

df
df.to_dict()
df.to_dict('list')



# titanic 에서 age가 60이상인 행만 골라주세요
titanic.query('Age >= 60')
titanic[titanic['Age'] >= 60]

titanic[['Name', 'Age']].query('Age >= 60')
titanic.query('Age >= 60')[['Name', 'Age']]

titanic.query('Age >= 60 and Pclass ==1 ')
titanic[(titanic['Age']>= 60) & (titanic['Pclass'] == 1)]

cond1 = titanic['Age'] >= 60
cond2 = titanic['Pclass'] == 1
titanic[cond1 & cond2]

# groupby : split, apply, combine

titanic.groupby("Pclass")
titanic.groupby("Pclass").count()
titanic.groupby("Pclass")[['PassengerId','Survived']].count()
titanic.groupby("Pclass")[['PassengerId','Survived']].agg([max,min])

# 칼럼 별로 다른 함수를 적용
titanic.groupby('Pclass')[['PassengerId','Survived']].agg({"PassengerId":max,"Survived":min})

len(titanic['Name'])
titanic["Name"].apply(lambda x: len(x))

# 패러미터(매0개변수) = Argument(인수,인자) = 하이퍼 파라미터(여러 옵선)
# 딥러닝 : 파라미터 : 가중치
#        하이퍼 파라미터 : 파라미터
titanic
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 데이터 정제
iris = load_iris()

iris.keys()
data = X = iris['data']
label = y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(data,label,test_size=0.2,random_state=11)

dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train, y_train)

pred = dt_clf.predict(X_test)

from sklearn.metrics import accuracy_score

print('예측 정확도: {0:4f}'.format(accuracy_score(y_test,pred)))


## 교차검증

'''
1) K-fold cv,
2) startified k-fold cv : 계층하된 ... 불균형한(imblanced) 분포
   카드 이상 검출 : 99.9% 는 정상, 0.1% 이상
   train_test_split(df, test_size = 0.2)
   
'''

from sklearn.model_selection import cross_val_score, cross_validate

iris = load_iris()

data = X = iris['data']
label = y = iris['target']

clf = DecisionTreeClassifier()

scores = cross_val_score(clf, data, label, scoring="accuracy", cv= 3)
print(np.round(scores,4))
print(np.round(np.mean(scores),4))


### GridSearchCV
from sklearn.model_selection import GridSearchCV

iris = load_iris()

data = X = iris['data']
label = y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(data,label,test_size=0.2,random_state=11)

dtree = DecisionTreeClassifier()

parameters = {'max_depth':[1,2,3],'min_samples_split':[2,3]}

grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3,refit=True)

grid_dtree.fit(X_train,y_train)

scores_df = pd.DataFrame(grid_dtree.cv_results_)
scores_df[['params','mean_test_score','rank_test_score','split0_test_score','split1_test_score','split2_test_score']]

print("GridSearchCV 최적 파라미터:", grid_dtree.best_params_)
print("GridSearchCV 최적 파라미터:{0:4f}".format(grid_dtree.best_score_))


# GridSearchCV의 refit으로 이미 학습된 estimator 반환
estimator = grid_dtree.best_estimator_

pred = estimator.predict(X_test)
print('테스트 데이터 세트 정확도: {0:4f}'.format(accuracy_score(y_test,pred)))

# 표준화

df.iloc[:,:2].apply(lambda x : (x - x.mean() / x.std()))

# 정규화

df.iloc[:,:2].apply(lambda x : (x - x.mean() / x.max() - x.min()))

##### 피처 스케일링과 정규화

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)

print('feature 들의 평균 값\n',iris_df.mean())
print('\n feature 들의 분산 값 \n',iris_df.var())


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('feature 들의 평균 값 \n',iris_df_scaled.mean())
print('\n feature 들의 분산 값 \n',iris_df_scaled.var())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)

print('feature들의 최소값')
print(iris_df_scaled.min())
print('\nfeature들의 최댓값')
print(iris_df_scaled.max())


### 타이타닉 생존자 예측
from sklearn.preprocessing import LabelEncoder

# Null 처리 함수
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

# 앞에서 설정한 Data Preprocessing 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df

# 원본 데이터를 재로딩 하고, feature데이터 셋과 Label 데이터 셋 추출. 
titanic_df = pd.read_csv('C:/python/machine_Learning/pymldg-rev-master/2장/titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df= titanic_df.drop('Survived',axis=1)

X_titanic_df = transform_features(X_titanic_df)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_titanic_df, y_titanic_df, \
                                                  test_size=0.2, random_state=11)
    
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 결정트리, Random Forest, 로지스틱 회귀를 위한 사이킷런 Classifier 클래스 생성
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression()

# DecisionTreeClassifier 학습/예측/평가
dt_clf.fit(X_train , y_train)
dt_pred = dt_clf.predict(X_test)
print('DecisionTreeClassifier 정확도: {0:.4f}'.format(accuracy_score(y_test, dt_pred)))

# RandomForestClassifier 학습/예측/평가
rf_clf.fit(X_train , y_train)
rf_pred = rf_clf.predict(X_test)
print('RandomForestClassifier 정확도:{0:.4f}'.format(accuracy_score(y_test, rf_pred)))

# LogisticRegression 학습/예측/평가
lr_clf.fit(X_train , y_train)
lr_pred = lr_clf.predict(X_test)
print('LogisticRegression 정확도: {0:.4f}'.format(accuracy_score(y_test, lr_pred)))

from sklearn.model_selection import KFold

def exec_kfold(clf, folds=5):
    # 폴드 세트를 5개인 KFold객체를 생성, 폴드 수만큼 예측결과 저장을 위한  리스트 객체 생성.
    kfold = KFold(n_splits=folds)
    scores = []
    
    # KFold 교차 검증 수행. 
    for iter_count , (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
        # X_titanic_df 데이터에서 교차 검증별로 학습과 검증 데이터를 가리키는 index 생성
        X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
        y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]
        
        # Classifier 학습, 예측, 정확도 계산 
        clf.fit(X_train, y_train) 
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        scores.append(accuracy)
        print("교차 검증 {0} 정확도: {1:.4f}".format(iter_count, accuracy))     
    
    # 5개 fold에서의 평균 정확도 계산. 
    mean_score = np.mean(scores)
    print("평균 정확도: {0:.4f}".format(mean_score)) 
# exec_kfold 호출
exec_kfold(dt_clf , folds=5) 

from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt_clf, X_titanic_df , y_titanic_df , cv=5)
for iter_count,accuracy in enumerate(scores):
    print("교차 검증 {0} 정확도: {1:.4f}".format(iter_count, accuracy))

print("평균 정확도: {0:.4f}".format(np.mean(scores)))


from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[2,3,5,10],
             'min_samples_split':[2,3,5], 'min_samples_leaf':[1,5,8]}

grid_dclf = GridSearchCV(dt_clf , param_grid=parameters , scoring='accuracy' , cv=5)
grid_dclf.fit(X_train , y_train)

print('GridSearchCV 최적 하이퍼 파라미터 :',grid_dclf.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dclf.best_score_))
best_dclf = grid_dclf.best_estimator_

# GridSearchCV의 최적 하이퍼 파라미터로 학습된 Estimator로 예측 및 평가 수행. 
dpredictions = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test , dpredictions)
print('테스트 세트에서의 DecisionTreeClassifier 정확도 : {0:.4f}'.format(accuracy))



##### Regression: MSE, RMSE, MAE, MAPE
'''
 Classification: Accuracy, Presion, Recall,
 Sensitivity, Specificity, FPR, TPR, F1 score, ROC AUC
 
 Accuracy, Presion, F1 score
'''



