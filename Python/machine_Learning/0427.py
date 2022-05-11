import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
# 데이터 읽어 들이기
mr = pd.read_csv("C:/python/machine_Learning/pyml-rev-main/ch4/mushroom.csv", header=None)
# 데이터 내부의 분류 변수 전개하기(원핫인코딩)
label = []
data = []
attr_list = []
for row_index, row in mr.iterrows():
    label.append(row.loc[0])
    exdata = []
    for col, v in enumerate(row.loc[1:]):
        if row_index == 0:
            attr = {"dic": {}, "cnt":0}
            attr_list.append(attr)
        else:
            attr = attr_list[col]
        # 버섯의 특징 기호를 배열로 나타내기
        d = [0,0,0,0,0,0,0,0,0,0,0,0]
        if v in attr["dic"]:
            idx = attr["dic"][v]
        else:
            idx = attr["cnt"]
            attr["dic"][v] = idx
            attr["cnt"] += 1
        d[idx] = 1
        exdata += d
    data.append(exdata)

# dummies를 쓸수도 있다
pd.get_dummies(mr)
# 아니면 나눈뒤에 이렇게
train, test = train_test_split(mr)
X_train = train_data = train.iloc[:,1:]
y_train = train_label = train.iloc[:,:1]
X_test = test_data = test.iloc[:,1:]
y_test = test_label = test.iloc[:,:1]

train_data = pd.get_dummies(train_data)
train_label = pd.get_dummies(train_label)
test_data = pd.get_dummies(test_data)
test_label = pd.get_dummies(test_label)

# 원핫인코딩 array로 바꾸기
train_data = train_data.values
train_label = train_label.values
test_data = test_data.values
test_label = test_label.values

mr.isin(['c']).sum()
test.isin(['c']).sum()
train.isin(['c']).sum()




# 학습 전용 데이터와 테스트 전용 데이터로 나누기
data_train, data_test, label_train, label_test = \
train_test_split(data, label)
# 데이터 학습시키기
clf = RandomForestClassifier()
clf.fit(data_train, label_train)
# 데이터 예측하기
predict = clf.predict(data_test)
# 결과 테스트하기
ac_score = metrics.accuracy_score(label_test, predict)
print("정답률 =", ac_score)

#########################################
## apply, map

mr = pd.read_csv("C:/python/machine_Learning/pyml-rev-main/ch4/mushroom.csv", header=None)
train, test = train_test_split(mr)

f = lambda x: ord(x)
train.applymap(f)

f = lambda x: x.map(ord)
train.apply(f)


########################################

# 정규화 데이터 값을 0과 1사이에 값으로 바꿔주는것

df = pd.DataFrame({'height':[160,163,167],'weight':[54,58,60]})
f = lambda x: (x-x.min()) / (x.max()-x.min())
df.apply(f)

def min_max_normalize(x):
    return ( x - x.min()) / ( x.max() - x.min() )

def z_score_normalize(x):
    return (x - np.mean(x)) / np.std(x)

df.apply(min_max_normalize)


######################################
# cross validation cv
# parameter, argument hyper-parameter = weights
# 매개변수    인수,인자  하이퍼파라미터   가중치

from sklearn import svm, metrics
import random, re
# 붓꽃의 CSV 파일 읽어 들이기 --- (※1)
lines = open('C:/python/machine_Learning/iris.csv', 'r', encoding='utf-8').read().split("\n")
f_tonum = lambda n : float(n) if re.match(r'^[0-9\.]+$', n) else n
f_cols  = lambda li: list(map(f_tonum,li.strip().split(',')))
csv = list(map(f_cols, lines))
del csv[0] # 헤더 제거하기
random.shuffle(csv) # 데이터 섞기


# 데이터를 K개로 분할하기 --- (※2)
K = 5 
csvk = [ [] for i in range(K) ]
for i in range(len(csv)):
    csvk[i % K].append(csv[i])
# 리스트를 훈련 전용 데이터와 테스트 전용 데이터로 분할하는 함수
def split_data_label(rows):
    data = []; label = []
    for row in rows:
        data.append(row[0:4])
        label.append(row[4])
    return (data, label)


# 정답률 구하기 --- (※3)
def calc_score(test, train):
    test_f, test_l = split_data_label(test)
    train_f, train_l = split_data_label(train)
    # 학습시키고 정답률 구하기
    clf = svm.SVC()
    clf.fit(train_f, train_l)
    pre = clf.predict(test_f)
    return metrics.accuracy_score(test_l, pre)


# K개로 분할해서 정답률 구하기 --- (※4)
score_list = []
for testc in csvk:
    # testc 이외의 데이터를 훈련 전용 데이터로 사용하기
    trainc = []
    for i in csvk:
        if i != testc: trainc += i
    sc = calc_score(testc, trainc)
    score_list.append(sc)
print("각각의 정답률 =", score_list)
print("평균 정답률 =", sum(score_list) / len(score_list))

##################################################
import pandas as pd
from sklearn import svm, metrics, model_selection
import random, re
# 붓꽃의 CSV 데이터 읽어 들이기 --- (※1)
csv = pd.read_csv('C:/python/machine_Learning/iris.csv')

# 리스트를 훈련 전용 데이터와 테스트 전용 데이터로 분할하기 --- (※2)
data = csv[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
label = csv["Name"]

# 크로스 밸리데이션하기 --- (※3)
clf = svm.SVC()
scores = model_selection.cross_val_score(clf, data, label, cv=5)
print("각각의 정답률 =", scores)
print("평균 정답률 =", scores.mean())

#################################################
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
# MNIST 학습 데이터 읽어 들이기 --- (※1)
train_csv = pd.read_csv("./mnist/train.csv")
test_csv  = pd.read_csv("./mnist/t10k.csv")


# 필요한 열 추출하기 --- (※2)
train_label = train_csv.iloc[:, 0]
train_data  = train_csv.iloc[:, 1:577]
test_label  = test_csv.iloc[:, 0]
test_data   = test_csv.iloc[:, 1:577]
print("학습 데이터의 수 =", len(train_label))


# 그리드 서치 매개변수 설정 --- (※3)
params = [
    {"C": [1,10,100,1000], "kernel":["linear"]},
    {"C": [1,10,100,1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]}
]


# 그리드 서치 수행 --- (※4)
clf = GridSearchCV( svm.SVC(), params, n_jobs=-1 )
clf.fit(train_data, train_label)
print("학습기 =", clf.best_estimator_)


# 테스트 데이터 확인하기 --- (※5)
pre = clf.predict(test_data)
ac_score = metrics.accuracy_score(pre, test_label)
print("정답률 =",ac_score)

###################
'''
if (x1* w1) + (x2*w2) + (x3*w3) >b:
    print("구매")
else:
    print("구매하지않음")
'''

# TensorFlow 임포트 --- (※1)
import tensorflow as tf

# 상수 정의 --- (※2)
a = tf.constant(1234)
b = tf.constant(5000)


# 계산 정의 --- (※3)
@tf.function
def add_op(a, b):
    return a + b


# 세션 시작하기 --- (※4)
res = add_op(a, b).numpy()        # 식 평가하기
print(res)

##############################
# TensorFlow 읽어 들이기 --- (※1)
import tensorflow as tf

# 상수 정의하기 --- (※2)
a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(4)


# 연산 정의하기 --- (※3)
@tf.function
def calc1_op():
    return a + b * c

@tf.function
def calc2_op():
    return (a + b) * c


# 세션 시작하기 --- (※4)
res1 = calc1_op().numpy() # 식 평가하기
print(res1)
res2 = calc2_op().numpy() # 식 평가하기
print(res2)

