# R


## 1.단축키 및  규칙

- Crtl + L : 화면 지우기
- Crtl + R : 스크립트에서 실행
- Crtl + Enter : R스튜디오 에서 스크립트 실행
- a = b 를 R에서는 a ←로 쓴다 단축키는 Alt -
- a ← c(1, 2, 3) R에서는 묶을때 앞에 c라는 함수를 쓴다
- shift + Alt + 아래키 : 해당줄 아래로 복사
- rm() : 객체 제거하기
- Crtl shift M : %>%

### R의 데이터 타입

- 숫자: numeric
- 문자: character   ‘홑따옴표’ 나 “쌍따옴표” 로 표현한다 하지만 ‘혼용”은 안된다
- 참거짓: logical  참거짓은 TRUE FALSE T F 등 대문자로만 표현

### 변수

- obj <- c(3, 5, 7)
- name_1 <- c(3, 5, 7)
- name.2 <- c(3, 5, 7)
- .name2 <- c(3, 5, 7)
- _name2 <- c(3, 5, 7) 에러 이름앞에 _가 오면 안된다
- 2name <- c(3, 5, 7) 에러 이름앞에 숫자가 오면 안된다

그래서 x_2name <- c(3, 5, 7) 앞에 아무의미없는 문자를 준다

### 객체의 타입

if, else for 같은 특정문자는 객체의 이름으로 쓸수 없다

a <- c(1, 2)
mode(a)  numeric

a <- c(1, 2, "a", "b")
mode(a)   character 문자 하나만있어도 char가 나옴

a <- c(1, 2, T, F)
mode(a)  numeric

데이터 타입 우선순위 character > numeric > logical

하나의 벡터에는 하나의 타입만 가져야 한다

### 괄호안의 숫자

matrix(1:12) <- R은 다른 언어와 다르게 1부터 12까지란 뜻이다 11이아니다

### 참거짓

```r
A <- T
B <- F
C <- c(T, T)
D <- c(F, T)

A & B # [1] FALSE
C & D # [1] FALSE  TRUE
A && B # [1] FALSE
C && D # [1] FALSE

A || B # [1] TRUE
C || D # [1] TRUE

A <- c(3, 4)
B <- c(5, 4)
A < B # [1]  TRUE FALSE
```

&&  나 || 은 각 벡터의 첫번째 원소만 비교하여 결과를 낸다

비교 하는 객체가 두쌍이면 결과도 두개 나온다

행 row record 관측치 속성

열 colum variable 변수 field

## 2.행렬(matrix)
``` `matrix(1:12)`
[,1]
[1,]    1
[2,]    2
[3,]    3
[4,]    4
[5,]    5
[6,]    6
[7,]    7
[8,]    8
[9,]    9
[10,]   10
[11,]   11
[12,]   12
```

``` `matrix(1:12, nrow = 3)`
[,1] [,2] [,3] [,4]
[1,]    1    4    7   10
[2,]    2    5    8   11
[3,]    3    6    9   12
```

`byrow = T` 옵션은 데이터가 행 먼저 들어가는 기능이다

``` mat <- matrix(1:12, nrow = 3, byrow = T)
mat
[,1] [,2] [,3] [,4]
[1,]    1    2    3    4
[2,]    5    6    7    8
[3,]    9   10   11   12
```

행이나 열의 이름을 설정할땐 rownames(), colnames()를 사용한다

``` rownames(mat) <- c("국어", "영어", "수학")
colnames(mat) <- c("a1", "a2", "a3", "a4")
mat
a1 a2 a3 a4
국어  1  2  3  4
영어  5  6  7  8
수학  9 10 11 12
```

벡터와 마찬가지로 하나의 데이터 타입만 가능하다

``` mode(mat)
[1] "numeric"
```

``` class(mat)
[1] "matrix" "array"
```

행렬에서 특정 원소만 출력하고 싶을 때 인덱스를 활용한다.

행렬에서 인덱스를 활용할때는 대괄호[]를 사용한다

특정 행 또는 열을 제외하고 싶을땐 (-)마이너스를 사용한다

``` mat[3,4]
[1] 12
mat[ ,4]
국어 영어 수학
4    8   12
mat[3, ]
a1 a2 a3 a4
9 10 11 12
mat[-2, ]
a1 a2 a3 a4
국어  1  2  3  4
수학  9 10 11 12
mat[-1,]
a1 a2 a3 a4
영어  5  6  7  8
수학  9 10 11 12
mat[-3,]
a1 a2 a3 a4
국어  1  2  3  4
영어  5  6  7  8
mat[,2:4]
a2 a3 a4
국어  2  3  4
영어  6  7  8
수학 10 11 12
mat[1:2,2:4]
a2 a3 a4
국어  2  3  4
영어  6  7  8
mat[c(1,3), c(2,4)]
a2 a4
국어  2  4
수학 10 12
mat[-2,c(2,4)]
a2 a4
국어  2  4
수학 10 12
```

``` t(mat) # 행과 열을 바꾼다(전치행렬)
국어 영어 수학
a1    1    5    9
a2    2    6   10
a3    3    7   11
a4    4    8   12
```

## 3.데이터 프레임(Data Frame)


```x1 <- c(100, 80, 60, 40, 20)
x2 <- c("A", "B", "C", "A", "B")
a <- cbind(x1, x2)
a
x1    x2
[1,] "100" "A"
[2,] "80"  "B"
[3,] "60"  "C"
[4,] "40"  "A"
[5,] "20"  "B"
```

```rbind(x1, x2)
[,1]  [,2] [,3] [,4] [,5]
x1 "100" "80" "60" "40" "20"
x2 "A"   "B"  "C"  "A"  "B"
```

```mode(a)
[1] "character"
class(a)
[1] "matrix" "array"
```

데이터 프레임은 행렬과 구조가 동일하나 각 열마다 다른 타입(num,char)의 데이터를 구성할수 있다.

```A <- data.frame(a)
class(A)
[1] "data.frame"
```

```mode(A)
[1] "list"
```

```rownames(A) <- c("a1","a2","a3","a4","a5")
colnames(A) <- c('score','grade') # 꼭 열의 갯수와 숫자를 맞춰야한다
A
score grade
a1   100     A
a2    80     B
a3    60     C
a4    40     A
a5    20     B
```

위 방식 말고 바로 data.frame()함수 안에 값을 넣어 만들수 도 있다

```x1 <- c(100, 80, 60, 40, 20)
x2 <- c("A", "B", "C", "A", "B")
df <- data.frame(score=x1,grade=x2)
df
score grade
1   100     A
2    80     B
3    60     C
4    40     A
5    20     B
```

데이터 프레임에서 인덱스를 활용 할땐 달러 기호($)를 사용할수 있다

$는 열(column)을 나타낸다.

```df[,1]
[1] 100  80  60  40  20
```

```df$score
[1] 100  80  60  40  20
```

str() : 데이터 구조 보기

```str(df)
'data.frame':	5 obs. of  2 variables:
$ score: num  100 80 60 40 20
$ grade: chr  "A" "B" "C" "A" ...
```

character 타입은 자동으로 Factor 타입으로 변환된다

즉 문자를 범주형으로 바꾸는 stringsAsFactors = T 가 기본값

행렬과 같이 df[ , ]으로 접근한다

```df <-  data.frame(score=x1,grade=x2,stringsAsFactors = T)
df
score grade
1   100     A
2    80     B
3    60     C
4    40     A
5    20     B
```

aaa.csv 라는 데이터 프레임이 있을때

"A", "B", "C", "A", "B"

stringAsFactors = FALSE 일땐 글자로만 읽는다

stringAsFactors = TRUE 일땐 등급으로 따진다

등급별로,성별별로,세대별로,부서별로,학년별로 등등을 '범주'라고한다

## 4.입력과 출력(I.O)

`getwd()` : 는 현재 지정된 작업 폴더의 경로를 출력한다.

`setwd()` : 는 새로운 작업 폴더의 경로를 설정할수 있으며 폴더 구분은  / 또는 \\ 로 한다.

`dir()` : 로 작업 폴더 내 파일 이름을 출력한다.

`install.packages("readxl")` : readxl 패키지 설치
`library(readxl)` : readxl 패키지 로드

`df_ex <- read_excel("excel_exam.xlsx")` 

`mean(df_ex$math)`

[1] 57.45

csv (comma separated values)

`df_ex2 <- read.csv("csv_exam.csv")`
```
| id | class | math | english | science |
| --- | --- | --- | --- | --- |
| 1 | 1 | 50 | 98 | 50 |
| 2 | 1 | 60 | 97 | 60 |
| 3 | 1 | 45 | 86 | 78 |
| 4 | 1 | 30 | 98 | 58 |
| 5 | 2 | 25 | 80 | 65 |
| 6 | 2 | 50 | 89 | 98 |
| 7 | 2 | 80 | 90 | 45 |
| 8 | 2 | 90 | 78 | 25 |
| 9 | 3 | 20 | 98 | 15 |
| 10 | 3 | 50 | 98 | 45 |
| 11 | 3 | 65 | 65 | 65 |
| 12 | 3 | 45 | 85 | 32 |
| 13 | 4 | 46 | 98 | 65 |
| 14 | 4 | 48 | 87 | 12 |
| 15 | 4 | 75 | 56 | 78 |
| 16 | 4 | 58 | 98 | 65 |
| 17 | 5 | 65 | 68 | 98 |
| 18 | 5 | 80 | 78 | 90 |
| 19 | 5 | 89 | 68 | 87 |
| 20 | 5 | 78 | 83 | 58 |
```

`df_ex2 <- df_ex2[1:10,]`

```
| id | class | math | english | science |
| --- | --- | --- | --- | --- |
| 1 | 1 | 50 | 98 | 50 |
| 2 | 1 | 60 | 97 | 60 |
| 3 | 1 | 45 | 86 | 78 |
| 4 | 1 | 30 | 98 | 58 |
| 5 | 2 | 25 | 80 | 65 |
| 6 | 2 | 50 | 89 | 98 |
| 7 | 2 | 80 | 90 | 45 |
| 8 | 2 | 90 | 78 | 25 |
| 9 | 3 | 20 | 98 | 15 |
| 10 | 3 | 50 | 98 | 45 |
```

`write.csv(df_ex2,"csv_exam_test.csv")`
`dir()`

객체 그대로 보내고 싶을떄

```
x1 <- c(100, 80, 60, 40, 20)
x2 <- c("A", "B", "C", "A", "B")
df <- data.frame(score=x1,grade=x2)
```

save(객체,’파일명’) 명령어를 이용해 rda파일로 저장 

`save(df, file = 'df_midterm.rda')`

읽기 : read.csv read_excel

저장 : csv, rda

excel_exam.xlsx 파일을 df_exam 이름으로 읽어 봅시다.

```
library(readxl)
df_exam = read_excel("excel_exam.xlsx")
```

`head(df_exam)` # 앞에서 부터 일부행만 출력

`head(df_exam, 2)` # 2행만 출력

`tail(df_exam)` # 뒤에서 부터 출력

`View(df_exam)` # 데이터 뷰어창 열기

`dim(df_exam)` # 행, 열 출력

`str(df_exam)` # 데이터 속성 확인

`summary(df_exam)` # 요약 통계량

`nrow(df_exam)` # 행의 갯수 출력

`ncol(df_exam)` # 열의 갯수 출력

## 5.dplyr

### 데이터 가공

```r
inner_join(test3,test4) # AND와 같이 모두 TRUE 여야 합침
left_join(test3,test4) # 왼쪽열 기준으로 합침
right_join(test3,test4) # 오른쪽열 기준으로 합침
full_join(test3,test4) # OR과 같이 모두 TRUE가 아니더라도 합침
```

### 5 verbs

- filter() # 행 추출
- select() # 열 추출
- arrange() # 정렬
- mutate() # 변수 추가
- summarise() # 통계치 산출

- group_by() # 집단별로 나누기
- left_join() # 데이터 합치기(열)
- bind_rows() # 데이터 합치기(행)

```r
df_exam[df_exam$class=="1",] # 행의 조건을 df_exam의 class 1 

filter(df_exam,class == 1) # filter를 이용한 방법
filter(df_exam,class == 2) # 2반
filter(df_exam,class != 1) # 1반이 아닌것
filter(df_exam,class==1 | class==2)# 1반 아니면 2반
filter(df_exam,math >= 50 & english>=80)
# 수학이 50점 이상이고 영어도 80점 이상
```

### Pipe

```r
### %>% 단축기 Crtl shift M
df_exam %>% filter(class == 1)
```

```r
# 유닉스 계열에서 쓰이는 기법 |
df_exam %>%
  filter(class ==1| class == 2) %>%
  filter(math > 50) %>%
  filter(english >= 90)
```

- filter(): 행추출 (행이 줄어든다)
- select() 열 추출
- arrange() 줄세우기
- mutate 파생변수( 변수 추가, 열 추가)

### 요약 통계량 함수

```r
df_exam$math #예시
sum(df_exam$math) # 합계
sd(df_exam$math) # 표준 편차
mean(df_exam$math) # 평균
median(df_exam$math) # 중앙값
min(df_exam$math) # 최솟값
max(df_exam$math) # 최댓값
n(df_exam$math) # 빈도
```

- summarise() 집단별로 요약하기
- join() 열 합치기
- bind_row() 행 합치기

