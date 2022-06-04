# Pandas

- 통합 인덱싱을 활용한 데이터 조작을 가능하게 하는 데이터프레임(DataFrame) 오브젝트
- 인메모리(in-memory) 데이터 구조와 다양한 파일 포맷들 간의 데이터 읽기/쓰기 환경 지원
- 데이터 결측치의 정렬 및 처리
- 데이터셋의 재구조화 및 피보팅(pivoting)
- 레이블 기반의 슬라이싱, 잘 지원된 인덱싱, 대용량 데이터셋에 대한 서브셋 지원
- 데이터 구조의 칼럼 추가 및 삭제
- 데이터셋의 분할-적용-병합을 통한 GroupBy 엔진 지원
- 데이터셋 병합(merging) 및 조인(joining) 지원
- 저차원 데이터에서의 고차원 데이터 처리를 위한 계층적 축 인덱싱 지원

## Pandas의 1차원 자료 Series

- index와 value로 쌍을 이룸, 딕셔너리와 비슷
- pd.Series()

## Pandas의 2차원 자료 DataFrame

- 행과 열로 이루어져 있다. 엑셀과 같다. 행렬은 Numpy
- pd.DataFrame()

## 데이터프레임 만드는 4가지 방법
- array
- dictionary
- list
- data, index, columns 직접 만들기

## 데이터 프레임의 행과 열 다루기

- drop 행,열, 데이터 삭제
- indexing 
    - iloc 행,열의 이름으로 찾기
    - loc 형,열의 순서로 찾기
- rename 행,열의 이름 바꾸기
- describe() 데이터 요약
- info() 행열별 데이터 요약

## 행 열 추가

- df.loc[새 행] = [parameter]
- df[새 열] = [parameter]

## 파일 읽기

- pd.read_csv() csv 파일 읽기
- pd.read_excel() 엑셀 파일 읽기

## 빈도수 세기

- count() 열 별 원소의 갯수
- value_counts() 원소 내용의 갯수
- unique() 유일한 값 찾기
- corr() 상관계수

## 결측치

- isnull() 데이터를 True, False로 바꿔줌 보통.sum()과 같이 씀
- dropna() NaN값이 있는 행 삭제 ( axis = 1 은 열 삭제)
- fillna() 결측치 대체
- duplicated() 중복행 찾기
- drop_duplicates() 중복행 제거

## 데이터 합치기

- concat
    - 결과값이 Series나 DataFrame으로 나옴
- merge
    - 특정 열 기준으로 합치기 가능
- how =
    - outer: NaN값 까지 전부 표시
    - inner: NaN값 없게끔 합치기
    - left: 왼쪽 데이터 프레임 기준으로 합치기
    - right: 오른쪽 데이터 프레임 기준으로 합치기

## 그룹화

-  groupby 특정 행 별로 묶음
-  뒤에 .mean() 같은 요약 통계량 함수와 연계 할수 있음

## 필터

- df[df['math'] > 80] 같이 조건문을 넣어줌
- 조건문을 객체에 넣고 df에서 부르는 mask 기법도 가능 ex) df[mask]

## 쿼리

- df.query('조건문')
- and, or, &, | 사용 가능
- A in B 도 가능
- 칼럼 이름 바로 사용
