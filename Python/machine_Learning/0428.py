import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()

a = tf.constant(120, name="a")
b = tf.constant(130, name="b")
c = tf.constant(140, name="c")

v = tf.Variable(0, name = "v")

calc_op = a + b + c
assign_op = tf.assign(v, calc_op)

sess = tf.Session()
sess.run(assign_op)


print( sess.run(v))

##########################

a = tf.placeholder(tf.int32, [3])

b = tf.constant(2);
x2_op = a * b

sess = tf.Session()

r1 = sess.run(x2_op, feed_dict={ a:[1,2,3]})
print(r1)
r2 = sess.run(x2_op, feed_dict={ a:[10,20,10]})
print(r2)


#################################
a = tf.placeholder(tf.int32, [None])

b = tf.constant(10);
x10_op = a * b

sess = tf.Session()

r1 = sess.run(x10_op, feed_dict={ a:[1,2,3,4,5]})
print(r1)
r2 = sess.run(x10_op, feed_dict={ a:[10,20,10]})
print(r2)

#################################################
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()

# 키, 몸무게, 레이블이 적힌 CSV 파일 읽어 들이기 --- (※1)
csv = pd.read_csv("C:/python/machine_Learning/pyml-rev-main/ch5/bmi.csv")

# 데이터 정규화 --- (※2)
csv["height"] = csv["height"] / 200
csv["weight"] = csv["weight"] / 100

# 레이블을 배열로 변환하기 --- (※3)
# - thin=(1,0,0) / normal=(0,1,0) / fat=(0,0,1)
bclass = {"thin": [1,0,0], "normal": [0,1,0], "fat": [0,0,1]}
csv["label_pat"] = csv["label"].apply(lambda x : np.array(bclass[x]))

# 테스트를 위한 데이터 분류 --- (※4)
test_csv = csv[15000:20000]
test_pat = test_csv[["weight","height"]]
test_ans = list(test_csv["label_pat"])

# 데이터 플로우 그래프 구축하기 --- (※5)
# 플레이스홀더 선언하기
x  = tf.placeholder(tf.float32, [None, 2]) # 키와 몸무게 데이터 넣기
y_ = tf.placeholder(tf.float32, [None, 3]) # 정답 레이블 넣기

# 변수 선언하기 --- (※6)
W = tf.Variable(tf.zeros([2, 3])); # 가중치
b = tf.Variable(tf.zeros([3])); # 바이어스

# 소프트맥스 회귀 정의하기 --- (※7)
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 모델 훈련하기 --- (※8)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)

# 정답률 구하기
predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

# 세션 시작하기
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 변수 초기화하기

# 학습시키기
for step in range(3500):
    i = (step * 100) % 14000
    rows = csv[1 + i : 1 + i + 100]
    x_pat = rows[["weight","height"]]
    y_ans = list(rows["label_pat"])
    fd = {x: x_pat, y_: y_ans}
    sess.run(train, feed_dict=fd)
    if step % 500 == 0:
        cre = sess.run(cross_entropy, feed_dict=fd)
        acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
        print("step=", step, "cre=", cre, "acc=", acc)

# 최종적인 정답률 구하기
acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
print("정답률 =", acc)

#####################################################
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils

# MNIST 데이터 읽어 들이기 --- (※1)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 데이터를 float32 자료형으로 변환하고 정규화하기 --- (※2)
X_train = X_train.reshape(60000, 784).astype('float32')
X_test  = X_test.reshape(10000, 784).astype('float')
X_train /= 255
X_test  /= 255

# 레이블 데이터를 0-9까지의 카테고리를 나타내는 배열로 변환하기 --- (※2a)
y_train = utils.to_categorical(y_train, 10)
y_test  = utils.to_categorical(y_test, 10)

# 모델 구조 정의하기 --- (※3)
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

# 모델 구축하기 --- (※4)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy'])

# 데이터 훈련하기 --- (※5)
hist = model.fit(X_train, y_train)

# 테스트 데이터로 평가하기 --- (※6)
score = model.evaluate(X_test, y_test, verbose=1)
print('loss=', score[0])
print('accuracy=', score[1])

###################################################
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd, numpy as np

# BMI 데이터를 읽어 들이고 정규화하기 --- (※1)
csv = pd.read_csv("C:/python/machine_Learning/pyml-rev-main/ch5/bmi.csv")

# 몸무게와 키 데이터
csv["weight"] /= 100
csv["height"] /= 200
X = csv[["weight", "height"]]  # --- (※1a)

# 레이블
bclass = {"thin":[1,0,0], "normal":[0,1,0], "fat":[0,0,1]}
y = np.empty((20000,3))
for i, v in enumerate(csv["label"]):
    y[i] = bclass[v]

# 훈련 전용 데이터와 테스트 전용 데이터로 나누기 --- (※2)
X_train, y_train = X[1:15001], y[1:15001]
X_test,  y_test  = X[15001:20001], y[15001:20001] 

# 모델 구조 정의하기 --- (※3)
model = Sequential()
model.add(Dense(512, input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(3))
model.add(Activation('softmax'))

# 모델 구축하기 --- (※4)
model.compile(
    loss='categorical_crossentropy',
    optimizer="rmsprop",
    metrics=['accuracy'])

# 데이터 훈련하기 --- (※5)
hist = model.fit(
    X_train, y_train,
    batch_size=100,
    epochs=20,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
    verbose=1)

# 테스트 데이터로 평가하기 --- (※6)
score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])


######################


import konlpy

from konlpy.tag import Okt

okt = Okt()

phrase = "아버지 가방에 들어가신다"
okt.pos("아버지 가방에 들어가신다")
okt.pos(phrase,norm=True, stem=True)
okt.morphs(phrase)
okt.normalize(phrase)

###################################

from konlpy.tag import Hannanum

hannanum = Hannanum()

hannanum.pos(phrase)
hannanum.morphs(phrase)
hannanum.nouns(phrase)

##################################

from konlpy.tag import Kkma

kkma = Kkma()

kkma.pos(phrase)
kkma.morphs(phrase)
kkma.nouns(phrase)
kkma.sentences(phrase)

####################################
import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Okt
# utf-16 인코딩으로 파일을 열고 글자를 출력하기 --- (※1)
fp = codecs.open("C:/python/machine_Learning/pyml-rev-main/ch6/BEXX0003.txt", "r", encoding="utf-16")
soup = BeautifulSoup(fp, "html.parser")
body = soup.select_one("body > text")
text = body.getText()
# 텍스트를 한 줄씩 처리하기 --- (※2)
okt = Okt()
word_dic = {}
lines = text.split("\n")
for line in lines:
    malist = okt.pos(line)
    for word in malist:
        if word[1] == "Noun": #  명사 확인하기 --- (※3)
            if not (word[0] in word_dic):
                word_dic[word[0]] = 0
            word_dic[word[0]] += 1 # 카운트하기

# 많이 사용된 명사 출력하기 --- (※4)
keys = sorted(word_dic.items(), key=lambda x:x[1], reverse=True)
for word, count in keys[:50]:
    print("{0}({1}) ".format(word, count), end="")
print()


##########################################
# word 2 vec
import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from gensim.models import word2vec

# utf-16 인코딩으로 파일을 열고 글자를 출력하기 --- (※1)
fp = codecs.open("C:/python/machine_Learning/pyml-rev-main/ch6/BEXX0003.txt", "r", encoding="utf-16")
soup = BeautifulSoup(fp, "html.parser")
body = soup.select_one("body > text")
text = body.getText()

# 텍스트를 한 줄씩 처리하기 --- (※2)
otk = Okt()
results = []
lines = text.split("\r\n")
for line in lines:
    # 형태소 분석하기 --- (※3)
    # 단어의 기본형 사용
    malist = otk.pos(line, norm=True, stem=True)
    r = []
    for word in malist:
        # 어미/조사/구두점 등은 대상에서 제외 
        if not word[1] in ["Josa", "Eomi", "Punctuation"]:
            r.append(word[0])
    rl = (" ".join(r)).strip()
    results.append(rl)
    print(rl)

# 파일로 출력하기  --- (※4)
gubun_file = 'toji.gubun'
with open(gubun_file, 'w', encoding='utf-8') as fp:
    fp.write("\n".join(results))

# Word2Vec 모델 만들기 --- (※5)
data = word2vec.LineSentence(gubun_file)
model = word2vec.Word2Vec(data, 
    vector_size=200, window=10, hs=1, min_count=2, sg=1)
model.save("toji.model")
print("ok")


###########################################
model = word2vec.Word2Vec.load("toji.model")

model.wv.most_similar(positive="땅")
model.wv.most_similar(positive=["집"])
model.wv.most_similar(negative=["집"])
model.wv.most_similar(negative=["집","절"])
model.wv.most_similar(positive="땅",negative=["집","절"])

############################################

model = word2vec.Word2Vec.load("C:/python/machine_Learning/pyml_rev_data_20191204/위키피디아모델/wiki.model")

model.wv.most_similar(positive=["Python","파이썬"])

model.wv.most_similar(positive=["아빠","여성"],negative=["남성"])

def md(x,y):
    return model.wv.most_similar(positive=x,negative=y)

    
md(["왕자","여성"],["남성"])

md(["서울","일본"],["한국"])

md(["서울","중국"],["한국"])

md(["오른쪽","남자"],["왼쪽"])[0]

md(["서울","맛집"],None)[0:5]

model["고양이"]
