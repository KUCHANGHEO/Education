import os, glob, json
root_dir = "./newstext"
dic_file = root_dir + "/word-dic.json"
data_file = root_dir + "/data.json"
data_file_min = root_dir + "/data-mini.json"
# 어구를 자르고 ID로 변환하기 ---(※1)
word_dic = { "_MAX": 0 }
def text_to_ids(text):
    text = text.strip()
    words = text.split(" ")
    result = []
    for n in words:
        n = n.strip()
        if n == "": continue
        if not n in word_dic:
            wid = word_dic[n] = word_dic["_MAX"]
            word_dic["_MAX"] += 1
            print(wid, n)
        else:
            wid = word_dic[n]
        result.append(wid)
    return result
# 파일을 읽고 고정 길이의 배열 리턴하기 ---(※2)
def file_to_ids(fname):
    with open(fname, "r") as f:
        text = f.read()
        return text_to_ids(text)
# 딕셔너리에 단어 모두 등록하기 --- (※3)
def register_dic():
    files = glob.glob(root_dir+"/*/*.gubun", recursive=True)
    for i in files:
        file_to_ids(i)
# 파일 내부의 단어 세기 --- (※4)
def count_file_freq(fname):
    cnt = [0 for n in range(word_dic["_MAX"])]
    with open(fname,"r") as f:
        text = f.read().strip()
        ids = text_to_ids(text)
        for wid in ids:
            cnt[wid] += 1
    return cnt
# 카테고리마다 파일 읽어 들이기 --- (※5)
def count_freq(limit = 0):
    X = []
    Y = []
    max_words = word_dic["_MAX"]
    cat_names = []
    for cat in os.listdir(root_dir):
        cat_dir = root_dir + "/" + cat
        if not os.path.isdir(cat_dir): continue
        cat_idx = len(cat_names)
        cat_names.append(cat)
        files = glob.glob(cat_dir+"/*.gubun")
        i = 0
        for path in files:
            print(path)
            cnt = count_file_freq(path)
            X.append(cnt)
            Y.append(cat_idx)
            if limit > 0:
                if i > limit: break
                i += 1
    return X,Y
# 단어 딕셔너리 만들기 --- (※5)
if os.path.exists(dic_file):
    word_dic = json.load(open(dic_file))
else:
    register_dic()
    json.dump(word_dic, open(dic_file,"w"))
# 벡터를 파일로 출력하기 --- (※6)
# 테스트 목적의 소규모 데이터 만들기
X, Y = count_freq(20)
json.dump({"X": X, "Y": Y}, open(data_file_min,"w"))
# 전체 데이터를 기반으로 데이터 만들기
X, Y = count_freq()
json.dump({"X": X, "Y": Y}, open(data_file,"w"))
print("ok")

############################################
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn import model_selection, metrics
import json
max_words = 56681 # 입력 단어 수: word-dic.json 파일 참고
nb_classes = 6 # 6개의 카테고리
batch_size = 64 
nb_epoch = 20
# MLP 모델 생성하기 --- (※1)
def build_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


# 데이터 읽어 들이기--- (※2)
data = json.load(open("C:/python/machine_Learning/newstext/word-dic.json")) 
#data = json.load(open("./newstext/data.json"))
X = data["X"] # 텍스트를 나타내는 데이터
Y = data["Y"] # 카테고리 데이터
# 학습하기 --- (※3)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
Y_train = np_utils.to_categorical(Y_train, nb_classes)
print(len(X_train),len(Y_train))
model = KerasClassifier(
    build_fn=build_model, 
    nb_epoch=nb_epoch, 
    batch_size=batch_size)
model.fit(X_train, Y_train)
# 예측하기 --- (※4)
y = model.predict(X_test)
ac_score = metrics.accuracy_score(Y_test, y)
cl_report = metrics.classification_report(Y_test, y)
print("정답률 =", ac_score)
print("리포트 =\n", cl_report)

###########################################
# 레벤슈타인 거리 구하기
def calc_distance(a, b):
    ''' 레벤슈타인 거리 계산하기 '''
    if a == b: return 0
    a_len = len(a)
    b_len = len(b)
    if a == "": return b_len
    if b == "": return a_len
    # 2차원 표 (a_len+1, b_len+1) 준비하기 --- (※1)
    matrix = [[] for i in range(a_len+1)]
    for i in range(a_len+1): # 0으로 초기화
        matrix[i] = [0 for j in range(b_len+1)]
    # 0일 때 초깃값을 설정
    for i in range(a_len+1):
        matrix[i][0] = i
    for j in range(b_len+1):
        matrix[0][j] = j
    # 표 채우기 --- (※2)
    for i in range(1, a_len+1):
        ac = a[i-1]
        for j in range(1, b_len+1):
            bc = b[j-1]
            cost = 0 if (ac == bc) else 1
            matrix[i][j] = min([
                matrix[i-1][j] + 1,     # 문자 삽입
                matrix[i][j-1] + 1,     # 문자 제거
                matrix[i-1][j-1] + cost # 문자 변경
            ])
    return matrix[a_len][b_len]
# "가나다라"와 "가마바라"의 거리 --- (※3)
print(calc_distance("가나다라","가마바라"))
# 실행 예
samples = ["신촌역","신천군","신천역","신발","마곡역"]
base = samples[0]
r = sorted(samples, key = lambda n: calc_distance(base, n))
for n in r:
    print(calc_distance(base, n), n)
    
############################################
def ngram(s, num):
    res = []
    slen = len(s) - num + 1
    for i in range(slen):
        ss = s[i:i+num]
        res.append(ss)
    return res
def diff_ngram(sa, sb, num):
    a = ngram(sa, num)
    b = ngram(sb, num)
    r = []
    cnt = 0
    for i in a:
        for j in b:
            if i == j:
                cnt += 1
                r.append(i)
    return cnt / len(a), r
a = "오늘 강남에서 맛있는 스파게티를 먹었다."
b = "강남에서 먹었던 오늘의 스파게티는 맛있었다."
# 2-gram
r2, word2 = diff_ngram(a, b, 2)
print("2-gram:", r2, word2)
# 3-gram
r3, word3  = diff_ngram(a, b, 3)
print("3-gram:", r3, word3)

#######################################

import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Twitter
import urllib.request
import os, re, json, random
# 마르코프 체인 딕셔너리 만들기 --- (※1)
def make_dic(words):
    tmp = ["@"]
    dic = {}
    for word in words:
        tmp.append(word)
        if len(tmp) < 3: continue
        if len(tmp) > 3: tmp = tmp[1:]
        set_word3(dic, tmp)
        if word == ".":
            tmp = ["@"]
            continue
    return dic
# 딕셔너리에 데이터 등록하기 --- (※2)
def set_word3(dic, s3):
    w1, w2, w3 = s3
    if not w1 in dic: dic[w1] = {}
    if not w2 in dic[w1]: dic[w1][w2] = {}
    if not w3 in dic[w1][w2]: dic[w1][w2][w3] = 0
    dic[w1][w2][w3] += 1
# 문장 만들기 --- (※3)
def make_sentence(dic):
    ret = []
    if not "@" in dic: return "no dic" 
    top = dic["@"]
    w1 = word_choice(top)
    w2 = word_choice(top[w1])
    ret.append(w1)
    ret.append(w2)
    while True:
        w3 = word_choice(dic[w1][w2])
        ret.append(w3)
        if w3 == ".": break
        w1, w2 = w2, w3
    ret = "".join(ret)
    # 띄어쓰기
    params = urllib.parse.urlencode({
        "_callback": "",
        "q": ret
    })
    # 네이버 맞춤법 검사기를 사용합니다.
    data = urllib.request.urlopen("https://m.search.naver.com/p/csearch/ocontent/util/SpellerProxy?" + params)
    data = data.read().decode("utf-8")[1:-2]
    data = json.loads(data)
    data = data["message"]["result"]["html"]
    data = soup = BeautifulSoup(data, "html.parser").getText()
    # 리턴
    return data
   
def word_choice(sel):
    keys = sel.keys()
    return random.choice(list(keys))
# 문장 읽어 들이기 --- (※4)
toji_file = "toji.txt"
dict_file = "markov-toji.json"
if not os.path.exists(dict_file):
    # 토지 텍스트 파일 읽어 들이기
    fp = codecs.open("C:/python/machine_Learning/pyml-rev-main/ch6/BEXX0003.txt", "r", encoding="utf-16")
    soup = BeautifulSoup(fp, "html.parser")
    body = soup.select_one("body > text")
    text = body.getText()
    text = text.replace("…", "") # 현재 koNLPy가 …을 구두점으로 잡지 못하는 문제 임시 해결
    # 형태소 분석
    twitter = Twitter()
    malist = twitter.pos(text, norm=True)
    words = []
    for word in malist:
        # 구두점 등은 대상에서 제외(단 마침표는 포함)
        if not word[1] in ["Punctuation"]:
            words.append(word[0])
        if word[0] == ".":
            words.append(word[0])
    # 딕셔너리 생성
    dic = make_dic(words)
    json.dump(dic, open(dict_file,"w", encoding="utf-8"))
else:
    dic = json.load(open(dict_file,"r"))
# 문장 만들기 --- (※6)
for i in range(3):
    s = make_sentence(dic)
    print(s)
    print("---")

###########################################
import codecs
from bs4 import BeautifulSoup
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random, sys
fp = codecs.open("C:/python/machine_Learning/pyml-rev-main/ch6/BEXX0003.txt", "r", encoding="utf-16")
soup = BeautifulSoup(fp, "html.parser")
body = soup.select_one("body")
text = body.getText() + " "
print('코퍼스의 길이: ', len(text))

# 문자를 하나하나 읽어 들이고 ID 붙이기
chars = sorted(list(set(text)))
print('사용되고 있는 문자의 수:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars)) # 문자 → ID
indices_char = dict((i, c) for i, c in enumerate(chars)) # ID → 문자

# 텍스트를 maxlen개의 문자로 자르고 다음에 오는 문자 등록하기
maxlen = 20
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('학습할 구문의 수:', len(sentences))
print('텍스트를 ID 벡터로 변환합니다...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# 모델 구축하기(LSTM)
print('모델을 구축합니다...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# 후보를 배열에서 꺼내기
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# 학습시키고 텍스트 생성하기 반복
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('반복 =', iteration)
    model.fit(X, y, batch_size=128, epochs=10) # 
    # 임의의 시작 텍스트 선택하기
    start_index = random.randint(0, len(text) - maxlen - 1)
    # 다양한 다양성의 문장 생성
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('--- 다양성 = ', diversity)
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('--- 시드 = "' + sentence + '"')
        sys.stdout.write(generated)
        # 시드를 기반으로 텍스트 자동 생성
        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.
            # 다음에 올 문자를 예측하기
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            # 출력하기
            generated += next_char
            sentence = sentence[1:] + next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
##############################################
import codecs
from bs4 import BeautifulSoup
import urllib.request
from konlpy.tag import Twitter
import os, re, json, random
dict_file = "chatbot-data.json"
dic = {}
twitter = Twitter()
# 딕셔너리에 단어 등록하기 --- (※1)
def register_dic(words):
    global dic
    if len(words) == 0: return
    tmp = ["@"]
    for i in words:
        word = i[0]
        if word == "" or word == "\r\n" or word == "\n": continue
        tmp.append(word)
        if len(tmp) < 3: continue
        if len(tmp) > 3: tmp = tmp[1:]
        set_word3(dic, tmp)
        if word == "." or word == "?":
            tmp = ["@"]
            continue
    # 딕셔너리가 변경될 때마다 저장하기
    json.dump(dic, open(dict_file,"w", encoding="utf-8"))
# 딕셔너리에 글 등록하기
def set_word3(dic, s3):
    w1, w2, w3 = s3
    if not w1 in dic: dic[w1] = {}
    if not w2 in dic[w1]: dic[w1][w2] = {}
    if not w3 in dic[w1][w2]: dic[w1][w2][w3] = 0
    dic[w1][w2][w3] += 1
# 문장 만들기 --- (※2)
def make_sentence(head):
    if not head in dic: return ""
    ret = []
    if head != "@": ret.append(head)        
    top = dic[head]
    w1 = word_choice(top)
    w2 = word_choice(top[w1])
    ret.append(w1)
    ret.append(w2)
    while True:
        if w1 in dic and w2 in dic[w1]:
            w3 = word_choice(dic[w1][w2])
        else:
            w3 = ""
        ret.append(w3)
        if w3 == "." or w3 == "？ " or w3 == "": break
        w1, w2 = w2, w3
    ret = "".join(ret)
    # 띄어쓰기
    params = urllib.parse.urlencode({
        "_callback": "",
        "q": ret
    })
    # 네이버 맞춤법 검사기를 사용합니다.
    data = urllib.request.urlopen("https://m.search.naver.com/p/csearch/ocontent/util/SpellerProxy?" + params)
    data = data.read().decode("utf-8")[1:-2]
    data = json.loads(data)
    data = data["message"]["result"]["html"]
    data = soup = BeautifulSoup(data, "html.parser").getText()
    # 리턴
    return data

def word_choice(sel):
    keys = sel.keys()
    return random.choice(list(keys))
# 챗봇 응답 만들기 --- (※3)
def make_reply(text):
    # 단어 학습시키기
    if not text[-1] in [".", "?"]: text += "."
    words = twitter.pos(text)
    register_dic(words)
    # 사전에 단어가 있다면 그것을 기반으로 문장 만들기
    for word in words:
        face = word[0]
        if face in dic: return make_sentence(face)
    return make_sentence("@")
# 딕셔너리가 있다면 읽어 들이기
if os.path.exists(dict_file):
    dic = json.load(open(dict_file,"r"))
    
############################################

from PIL import Image
import numpy as np

def average_hash(fname, size = 16):
    img = Image.open(fname)
    img = img.convert('L')
    img = img.resize((size, size))
    pixel_data = img.getdata()
    pixels = np.array(pixel_data)
    pixels = pixels.reshape((size,size))
    avg = pixels.mean()
    diff = 1 * ( pixels > avg )
    return diff

def np2hash(ahash):
    bhash = []
    for nl in ahash.tolist():
        s1 = [str(i) for i in nl]
        s2 = "".join(s1)
        i = int(s2, 2)
        bhash.append("%04x" % i)
    return "".join(bhash)

ahash = average_hash('C:/python/machine_Learning/pyml-rev-main/ch7/tower.jpg')
print(ahash)
print(np2hash(ahash))    

###########################################

from PIL import Image
import numpy as np
import os, re
# 파일 경로 지정하기
search_dir = ".\\image\\101_ObjectCategories"
cache_dir = ".\\image\\cache_avhash"
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)
# 이미지 데이터를 Average Hash로 변환하기 --- (※1)
def average_hash(fname, size = 16):
    fname2 = fname[len(search_dir):]
    # 이미지 캐시하기
    cache_file = cache_dir + "\\" + fname2.replace('\\', '_') + ".csv"
    if not os.path.exists(cache_file): # 해시 생성하기
        img = Image.open(fname)
        img = img.convert('L').resize((size, size), Image.ANTIALIAS)
        pixels = np.array(img.getdata()).reshape((size, size))
        avg = pixels.mean()
        px = 1 * (pixels > avg)
        np.savetxt(cache_file, px, fmt="%.0f", delimiter=",")
    else: # 캐시돼 있다면 읽지 않기
        px = np.loadtxt(cache_file, delimiter=",")
    return px

# 해밍 거리 구하기 --- (※2)
def hamming_dist(a, b):
    aa = a.reshape(1, -1) # 1차원 배열로 변환하기
    ab = b.reshape(1, -1)
    dist = (aa != ab).sum()
    return dist

# 모든 폴더에 처리 적용하기 --- (※3)
def enum_all_files(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            fname = os.path.join(root, f)
            if re.search(r'\\.(jpg|jpeg|png)$', fname):
                yield fname


# 이미지 찾기 --- (※4)
def find_image(fname, rate):
    src = average_hash(fname)
    for fname in enum_all_files(search_dir):
        dst = average_hash(fname)
        diff_r = hamming_dist(src, dst) / 256
        # print("[check] ",fname)
        if diff_r < rate:
            yield (diff_r, fname)

# 찾기 --- (※5)
srcfile = search_dir + "\\chair\\image_0016.jpg"
html = ""
sim = list(find_image(srcfile, 0.25))
sim = sorted(sim, key=lambda x:x[0])
for r, f in sim:
    print(r, ">", f)
    s = '<div style="float:left;"><h3>[ 차이 :' + str(r) + '-' + \
        os.path.basename(f) + ']</h3>'+ \
        '<p><a href="' + f + '"><img src="' + f + '" width=400>'+ \
        '</a></p></div>'
    html += s

# HTML로 출력하기
html = """<html><head><meta charset="utf8"></head>
<body><h3>원래 이미지</h3><p>
<img src='{0}' width=400></p>{1}
</body></html>""".format(srcfile, html)
with open(".\\avhash-search-output.html", "w", encoding="utf-8") as f:
    f.write(html)
print("ok")

#####################################################
from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
# 분류 대상 카테고리 선택하기 --- (※1)
caltech_dir = "./image/101_ObjectCategories"
categories = ["chair","camera","butterfly","elephant","flamingo"]
nb_classes = len(categories)
# 이미지 크기 지정 --- (※2)
image_w = 64 
image_h = 64
pixels = image_w * image_h * 3
# 이미지 데이터 읽어 들이기 --- (※3)
X = []
Y = []
for idx, cat in enumerate(categories):
    # 레이블 지정 --- (※4)
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # 이미지 --- (※5)
    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f) # --- (※6)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)
X = np.array(X)
Y = np.array(Y)
# 학습 전용 데이터와 테스트 전용 데이터 구분 --- (※7)
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./image/5obj.npy", xy)
print("ok,", len(Y))
#####################################################
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np

# 카테고리 지정하기
categories = ["chair","camera","butterfly","elephant","flamingo"]
nb_classes = len(categories)
# 이미지 크기 지정하기
image_w = 64 
image_h = 64

# 데이터 불러오기 --- (※1)
X_train, X_test, y_train, y_test = np.load("./image/5obj.npy", allow_pickle = True)
# 데이터 정규화하기
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float")  / 256
print('X_train shape:', X_train.shape)

# 모델 구축하기 --- (※2)
model = Sequential()
model.add(Convolution2D(32, 3, 3, padding='same',input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))

model.add(Flatten()) # --- (※3) 
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# 모델 훈련하기 --- (※4)
model.fit(X_train, y_train, batch_size=32, epochs=50)
    
# 모델 평가하기--- (※5)
score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])

#################################################################
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import h5py
from PIL import Image
import numpy as np
import os

# 카테고리 지정하기
categories = ["chair","camera","butterfly","elephant","flamingo"]
nb_classes = len(categories)

# 이미지 크기 지정하기
image_w = 64 
image_h = 64

# 데이터 불러오기 --- (※1)
X_train, X_test, y_train, y_test = np.load("./image/5obj.npy", allow_pickle = True)
# 데이터 정규화하기
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float")  / 256
print('X_train shape:', X_train.shape)

# 모델 구축하기 --- (※2)
model = Sequential()
model.add(Convolution2D(32, 3, 3, 
    padding='same',
    input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))

model.add(Flatten()) # --- (※3) 
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

# 모델 훈련하기 --- (※4)
hdf5_file = "./image/5obj-model.hdf5"
if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 읽어 들이기
    model.load_weights(hdf5_file)
else:
    # 학습한 모델을 파일로 저장하기
    model.fit(X_train, y_train, batch_size=32, epochs=50)
    model.save_weights(hdf5_file)
    
# 모델 평가하기--- (※5)
# 예측하기
pre = model.predict(X_test)
# 예측 결과 테스트 하기
for i,v in enumerate(pre):
    pre_ans = v.argmax() # 예측한 레이블
    ans = y_test[i].argmax() # 정답 레이블
    dat = X_test[i] # 이미지 데이터
    if ans == pre_ans: continue
    # 예측이 틀리면 무엇이 틀렸는지 출력하기
    print("[NG]", categories[pre_ans], "!=", categories[ans])
    print(v)
    # 이미지 출력하기
    fname = "image/error/" + str(i) + "-" + categories[pre_ans] + \
        "-ne-" + categories[ans] + ".png"
    dat *= 256
    img = Image.fromarray(np.uint8(dat))
    img.save(fname)
score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])


############################################################
import cv2
import sys

# 입력 파일 지정하기
image_file = "./pakutas/photo1.jpg"

# 캐스케이드 파일의 경로 지정하기 --- (※1)
cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"

# 이미지 읽어 들이기 --- (※2)
image = cv2.imread(image_file)

# 그레이스케일로 변환하기
image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 인식 특징 파일 읽어 들이기 --- (※3)
cascade = cv2.CascadeClassifier(cascade_file)

# 얼굴 인식 실행하기
face_list = cascade.detectMultiScale(image_gs,
    scaleFactor=1.1,
    minNeighbors=1,
    minSize=(150,150))
if len(face_list) > 0:
    # 인식한 부분 표시하기 --- (※4)
    print(face_list)
    color = (0, 0, 255)
    for face in face_list:
        x,y,w,h = face
        cv2.rectangle(image, (x,y), (x+w, y+h), color, thickness=8)
    # 파일로 출력하기 --- (※5)
    #cv2.imwrite("facedetect-output.PNG", image)
    cv2.imwrite("./pakutas/photo1-facedetect.PNG", image)
else:
    print("no face")
    
################################################################

import cv2, sys, re

# 입력 파일 지정하기 --- (※1)

image_file = ("C:/python/machine_Learning/pakutas/photo1.jpg")

# 출력 파일 이름
output_file = re.sub(r'\.jpg|jpeg|PNG$', '-mosaic.jpg', image_file)
print(output_file)
mosaic_rate = 30 

# 캐스캐이드 파일 경로 지정하기
cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"

# 이미지 읽어 들이기 --- (※2)
image = cv2.imread(image_file)
image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 그레이스케일 변환

# 얼굴 인식 실행하기 --- (※3)
cascade = cv2.CascadeClassifier(cascade_file)
face_list = cascade.detectMultiScale(image_gs,
    scaleFactor=1.1,
    minNeighbors=1,
    minSize=(100,100))
if len(face_list) == 0:
    print("no face")
    quit()

# 확인한 부분에 모자이크 걸기 -- (※4)
print(face_list)
color = (0, 0, 255)
for (x,y,w,h) in face_list:
    # 얼굴 부분 자르기 --- (※5)
    face_img = image[y:y+h, x:x+w]
    # 자른 이미지를 지정한 배율로 확대/축소하기 --- (※6)
    face_img = cv2.resize(face_img, (w//mosaic_rate, h//mosaic_rate))
    # 확대/축소한 그림을 원래 크기로 돌리기 --- (※7)
    face_img = cv2.resize(face_img, (w, h), 
        interpolation=cv2.INTER_AREA)
    # 원래 이미지에 붙이기 --- (※8)
    image[y:y+h, x:x+w] = face_img

# 렌더링 결과를 파일에 출력
cv2.imwrite(output_file, image)
