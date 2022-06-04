# Numpy

- Numerical Python
- 강력한 N 차원 배열 객체.list 보다 빠르다
- 정교한 브로드캐스팅 기능. 반복문을 쓰지 않음
- 유용한 선형 대수학, 푸리에 변환 및 난수 기능
- C/C ++ 및 포트란 코드 통합 도구

- Numpy는 python코드가 아니라 C코드로 작성되어있다.
- 메모리 저장구조가 다르다

## 임포트하기

- import numpy

## 함수 이름을 바로 쓰고 싶을 때

- from snumpy import var

## Array 만들기

- 각 차원을 축(axis)라고 합니다.
- 축의 개수는 차원의 개수인데 rank 라고도 합니다.
- 배열의 차원을 shape라고 하고 tuple로 표시합니다.
- shape안의 숫자는 각 차원에 있는 원소의 개수입니다.
- 전체 원소의 개수는 size라고 합니다.

## 2차원, 3차원 배열 만들기

- np.array()함수를 써서 만들기
- arange()와 reshape()를 써서 만들기

## 배열의 연산
- 리스트의 연산과 비슷하지만,약간 다르다
- numpy함수를 쓰는것 보다 객체.함수()가 편리하다
- Numpy의 연산은 원소들 끼리 이루어진다. element wise
- array의 형태(shape)가 안 맞으면, 자동으로 맞추어 주기도 한다. broadcasting

## Numpy 내장함수를 이용하여 array 만들기

- np.linspace 시작과 끝 원소 갯수를 정의
- np.ones 1로채워진 차원을 만듬
- np.zeros 0으로 채워진 차원을 만듬
- np.diag 입력받은 숫자를 대각선으로 나열
- np.full 입력한 숫자로 차원을 만듬
- np.eyes 입력한 숫자만큼 원핫 인코딩
- np.tile 입력받은 shape를 반복
- np.empty null값으로 채움
- np.zeros_like 다른 배열의 형태와 데이터 타입만 가져와서 0으로 채움 
- np.ones_like 다른 배열의 형태와 데이터 타입만 가져와서 1으로 채움

## copy와 view
- 기본 배열로부터 새로운 배열을 생성하기 위해서는 copy함수로 명시적으로 사용해야 함
- copy 함수로 복사한 배열은 원본 배열과 완전히 다른 새로운 객체입니다.
- slice,indexing은 새로운 객체가 아닌 기존 배열의 뷰(View)입니다.
- 반환한 배열의 값을 변경하면 원본 배열이 변경된다.

## 배열의 결합

- hstack
- vstack
- dstack
- concatenate

## 배열의 분리
- spliting
- 수평분리 hsplit
- 수직분리 vsplit

## Numpy의 shape 변경 함수

- reshape 배열을 원하는 shape로 바꿔줌
- flatten 다차원 배열을 1차원으로 바꿔줌
- ravel 다차원 배열을 연산할 때 1차원으로 바꿔줌
- transpose 행렬의 곱셈을 가능하게 바꾸워줌

## 배열의 비교
- 같다(==)
- 크다,작다
- 배열 전체를 하나로 비교(np.array_equal)

## 정렬하기
- numpy.sort()와 객체.sort()는 다르다
- 1차원 배열의 정렬, 역순으로 정렬
- 2차원 배열의 정렬과 축기준으로 정렬하기

## indexing

## slicing

## Fancy Indexing.position

- 정수배열 indexer를 사용해서 indexing하는 방법

## broad casting
- 차원(dimension)이 다른 두 배열의 연산에서
- 낮은 차원의 배열이 차원을 맞추어 주도록 변화한다
- 데이터의 복사를 하지 않으므로 빠르다
