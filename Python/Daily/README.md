# Daily Python

> 중앙정보학원

### 01일차 강의 요약

- 숫자형의 데이터 타입, 기본 연산
- 문자열의 표현 방식과 문자열의 연산
- 인덱싱과 슬라이싱

### 02일차 강의 요약

- strip()함수
- split()함수
- 리스트, 튜플, 딕셔너리

### 03일차 강의 요약

- 리스트 인덱싱 및 리스트 기초 연산
- 딕셔너리 인덱싱 및 딕셔너리 기초 연산
- 변수의 타입이 갖는 특징 확인
- 리스트 comprehension
- def, lambda 예약어 사용 및 사용자 정의 함수 생성
- 파일 입출력 기초

### 04일차 강의 요약

- 파일 입출력
- 파이썬 내장 모듈 및 함수
- 데이터 분석 라이브러리
    - Numpy, Pandas, Matplotlib
    - Numpy의 array 연산 및 관련 함수

### 05일차 강의 요약

- numpy를 통한 인덱스값 불러오기
- 행렬 만들기
- matplotlib을 통해 시각화 표현

### 06일차 강의 요약

- numpy 5일차 복습
- 분산과 표준편차 계산
- 팬시 색인, 불리언 색인

### 07일차 강의 요약

- iris 데이터 분리 평균값 구하기 평균값을 꺽은선 그래프로 그리기 복습 
- array의 팬시 색인, 불리언 색인, 조건식 응용 
- np.where 활용
- np.random.rand(0~1사이값), randn(0 중심의 정규분포)

### 08일차 강의 요약

- 절대 값 계산 np.abs 함수
- matplotlib의 hist, axis, scatter, colorbar, imshow
- numpy, imshow 를 이용한 이미지 가공 후 출력

### 09일차 강의 요약

- numpy를 이용한 이미지 변환, 이미지 파일의 numpy배열안 숫자의 의미 이해

### 10일차 강의 요약

- numpy를 이용한 이미지 reshape,array를 plt로 그래프 그리기,표준 정규분포

### 11일차 강의 요약

- numpy 행렬 연산
- iris 예제를 통한 정규화 연습 및 matplotlib.pyplot(이하 plt)를 활용한 시각화
- plt의 서브플랏과 범례 작성
- pandas 라이브러리 기초
- 데이터 프레임 생성, 수정, 삭제, 복사 등
- csv 파일을 읽어 들여 데이터 프레임으로 변환

### 12일차 강의 요약

- pandas 라이브러리 응용
- 데이터 프레임의 속성 확인 및 인덱싱
- pd.DataFrame 클래스가 가진 함수 적용
- 데이터 프레임을 plt를 활용한 그래프 시각화
- 데이터 프레임에 통계 함수 또는 numpy의 함수 적용

### 13일차 강의 요약

- 데이터 프레임을 map과 lambda를 이용한 데이터 추출, 변형, 열추가
- 데이터 프레임의 데이터를 조건을 이용한 인덱싱
- 데이터 프레임의 원소 값을 보는 value_counts()
- 데이터 프레임 정렬 sort_index, sort_values
- 데이터 프레임 결측치 처리, 특정 칼럼에 적용하는 apply
- 데이터 프레임 merge, 그룹별 연산 groupby

### 14일차 강의 요약

- 데이터 프레임을 산점도 그래프로 표시
- 데이터 프레임 산점도 중 특정 원소값을 표시
- 데이터 프레임 정렬 sort_index, sort_values
- 데이터 프레임 groupby와 apply
- 데이터 프레임을 plot으로 시각화

### 15일차 강의 요약

- iris 데이터 사용
- 점들 간의 거리 구하기
- 산점도 그리기

### 16일차 강의 요약

- world_happiness_report 데이터 사용
- LinearRegression 모델을 사용
- 가중치와 기울기를 이용하여 예측값 구하기
- matplotlib을 이용한 시각화

### 17일차 강의 요약

### 18일차 강의 요약

### 19일차 강의 요약

- 머신러닝
- KNN(K-Nearest Neighbor) 
    - iris 데이터를 활용해 패턴 생성
    - plt.scatter()를 통해 산점도 그래프 그리기
    - 복습 문제로 100까지의 행으로 versicolor, virginica 2가지 속성만을 다루고, mglearn 이라는 라이브러리를 통해 시각화

### 20일차 강의 요약

- 19일차와 마찬가지로 iris 데이터 사용
- Train 데이터와 Test 데이터 분리 
    - from sklearn.model_selection import train_test_split
- 유방암 데이터 
    - from sklearn.datasets import load_breast_cancer
    - cancer = load_breast_cancer()
    - load_...()를 통해 데이터셋 로드, iris의 경우 load_iris()
    - KNN 통해 실습하고 정규화까지 적용하여 시각화

### 21일차 강의 요약

- 유방암 데이터셋 활용
    - 정규화 X_norm = (X-X.mean(axis=0)) / X.std(axis=0)
    - train_test_split 사용하여 분류
    - SVM 모델 사용
    - SVC의 옵션을 사용하여 튜닝
    > C: 각포인트의 중요도를 제한하는 매개변수
  
    > gamma: 하나의 훈련 샘플이 미치는 영향의 범위를 결정하는 매개변수

### 22일차 강의 요약

- iris 데이터셋 활용
    - sepal_length와 sepal_width 속성을 사용하여 X, y 생성
    - SVM 모델 사용
    - np.corrcoef를 사용하여 피어슨 상관계수 값 계산
    - 분류 알고리즘
    > KNN, SVM, Decision Tree 등등
    - 모델 시각화를 위하여 graphviz 설치

### 23일차 강의 요약

- 행복 데이터를 머신 러닝 결정트리를 이용한 분류후 시각화
    - np.corrcoef(): 상관관계 분석
    - mglearn, graphviz 시각화 도구
- 암 데이터를 주 성분 분석(PCA)분석 후 시각화
    - pca.components_.shape 주성분 개수, 방향
- 숫자 필기체 데이터를 PCA 처리후 시각화

### 24일차 강의 요약

- 숫자 필기체 데이터를 주 성분 분석, plt를 이용하여 산점도 시각화
    - round() 함수를 이용해 주성분 반올림
- PCA를 이용한 숫자 필기체 이미지 복원
    - np.matmul(X_pca[:,:2], pca.components_[:2]): 행렬 곱 함수
- 유방암 데이터를 로지스틱 회귀를 이용하여 학습 후 평가
    - coef_ : 가중치 (w)
    - intercept_ : 절편값 (b)
 
### 25일차 강의 요약


### 26일차 강의 요약


### 27일차 강의 요약

- numpy를 활용한 신경망 구성
- world_happiness_report_2021 데이터를 사용한 신경망
    - sigmoid 사용자 정의 함수
    - def sigmoid(x) : return 1 / (1 + np.exp(-x) )
    - 이후 각 층마다 가중치를 설정하여 신경망 구성
    - 신경망을 통해 얻은 예측값은 mse(평균제곱오차)를 통해 평가
- breast_cancer 데이터를 사용한 신경망
    - 마찬가지로 numpy를 통해 가중치와 신경망을 구성
    - binary_crossentropy를 통해 오차 계산
    - entropy = (-y*np.log(pred_y)-(1-y)*np.log(1-pred_y)).mean()

### 28일차 강의 요약

- 임의의 가중치로 선형회귀를 적용한 iris 데이터
- 활성화 함수로 소프트 맥스를 사용하고 크로스엔트로피를 계산

### 29일차 강의 요약

### 30일차 강의 요약

### 31일차 강의 요약

- 행복 데이터를 표준화 후 Keras.Sequential 모델에 넣어 학습
- 전체 데이터중 대한민국  위치 시각화
- 학습중 loss 변화율 시각화

### 01 신경망 강의 요약

- 데이터 전처리: X_train = X_train.reshape(-1,28*28)/255.
- 원 핫 인코딩 y_train = np.eye(10)[y_train] (test도 똑같이)
- 신경망 구조, 비용함수, 활성화 함수
- 단층 신경망 ( 로지스틱 회귀)
    -  model = Sequential()
    -  model.add(Dense(10, input_shape=(784,), activation='softmax'))
    -  model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    -  history = model.fit(X_train, y_train, epochs=10)
    -  history에 저장해둔 metrics를 plot으로 시각화
    -  predict로 평가
    -  가중치 저장  W, b = model.get_weights()
    -  W 값을 plt로 시각화
- 중간층 추가
    - model = Sequential()
    - model.add(Dense(128, activation='relu', input_shape=(784,)))
    - model.add(Dense(256, activation='relu'))
    - model.add(Dense(256, activation='relu'))
    - model.add(Dense(10, activation='softmax'))
    - 단층 신경망 보다 정답률 증가
- 모델 저장 model.save('mnist_model_01.h5')
- 출력값을 1차원으로 풀어주는 Flatten
    - model.add(Flatten(input_shape=(28,28)))
- 고의적으로 중간 단계의 출력값들을 누락 시키는 Dropout
    - 여러 가지 예외사항에 대처가 가능한 강건한 모델을 만들 수 있다
    - model.add(Dropout(0.5))
- 이진 분류
    - Dense(1, activation='sigmoid') 출력층 하나와 sigmoid 함수
    - loss='binary_crossentropy' categorical이 아닌 binary

### 02 신경망 강의 요약

- 이미지 처리에 있어서 기존 신경망의 문제점
    - 이미지 데이터를 1차원 배열로 처리
    - 실제 이미지는 상화좌우의 픽셀들과 연관되어있음
- CNN(Convolutional Neural Network)
    - Convolution: 이미지에 필터를 적용해 변환하는 기술
    - Pooling: 이미지의 크기를 줄임
    - 채널: 컬러 이미지는 RGB 3개의 채널이지만, 다수의 필터를 적용하여 많은 채널을 생성
    - CNN에선 필터의 값들이 곧 가중치 이다.
    - 즉, CNN은 이미지를 잘 표현하는 필터를 찾아내는 것이 목표
- MNIST에 CNN 적용
    - model = keras.models.Sequential()
    - model.add(keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(28,28,1)))
    - model.add(keras.layers.MaxPool2D(2))
    - model.add(keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    - model.add(keras.layers.MaxPool2D(2))
    - model.add(keras.layers.Flatten())
    - model.add(keras.layers.Dense(10, activation='softmax'))
    - 기존 신경망 보다 정답률이 높아진것을 
