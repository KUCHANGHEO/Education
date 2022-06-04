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

```
id class  math english science
<dbl> <dbl> <dbl>   <dbl>   <dbl>
1     1     1    50      98      50
2     2     1    60      97      60
3     3     1    45      86      78
4     4     1    30      98      58
5     5     2    25      80      65
6     6     2    50      89      98
7     7     2    80      90      45
8     8     2    90      78      25
9     9     3    20      98      15
10    10     3    50      98      45
11    11     3    65      65      65
12    12     3    45      85      32
13    13     4    46      98      65
14    14     4    48      87      12
15    15     4    75      56      78
16    16     4    58      98      65
17    17     5    65      68      98
18    18     5    80      78      90
19    19     5    89      68      87
20    20     5    78      83      58
```

`head(df_exam)` # 앞에서 부터 일부행만 출력

```
id class  math english science
<dbl> <dbl> <dbl>   <dbl>   <dbl>
1     1     1    50      98      50
2     2     1    60      97      60
3     3     1    45      86      78
4     4     1    30      98      58
5     5     2    25      80      65
6     6     2    50      89      98
```

`head(df_exam, 2)` # 2행만 출력

```
id class  math english science
<dbl> <dbl> <dbl>   <dbl>   <dbl>
1     1     1    50      98      50
2     2     1    60      97      60
```

`tail(df_exam)` # 뒤에서 부터 출력

```
id class  math english science
<dbl> <dbl> <dbl>   <dbl>   <dbl>
1    15     4    75      56      78
2    16     4    58      98      65
3    17     5    65      68      98
4    18     5    80      78      90
5    19     5    89      68      87
6    20     5    78      83      58
```

`View(df_exam)` # 데이터 뷰어창 열기

`dim(df_exam)` # 행, 열 출력


[1] 20 5

`str(df_exam)` # 데이터 속성 확인

```
tibble [20 x 5] (S3: tbl_df/tbl/data.frame)
$ id     : num [1:20] 1 2 3 4 5 6 7 8 9 10 ...
$ class  : num [1:20] 1 1 1 1 2 2 2 2 3 3 ...
$ math   : num [1:20] 50 60 45 30 25 50 80 90 20 50 ...
$ english: num [1:20] 98 97 86 98 80 89 90 78 98 98 ...
$ science: num [1:20] 50 60 78 58 65 98 45 25 15 45 ...
```

`summary(df_exam)` # 요약 통계량

```
id            class        math
Min.   : 1.00   Min.   :1   Min.   :20.00
1st Qu.: 5.75   1st Qu.:2   1st Qu.:45.75
Median :10.50   Median :3   Median :54.00
Mean   :10.50   Mean   :3   Mean   :57.45
3rd Qu.:15.25   3rd Qu.:4   3rd Qu.:75.75
Max.   :20.00   Max.   :5   Max.   :90.00
english        science
Min.   :56.0   Min.   :12.00
1st Qu.:78.0   1st Qu.:45.00
Median :86.5   Median :62.50
Mean   :84.9   Mean   :59.45
3rd Qu.:98.0   3rd Qu.:78.00
Max.   :98.0   Max.   :98.00
```

`nrow(df_exam)` # 행의 갯수 출력

[1] 20

`ncol(df_exam)` # 열의 갯수 출력

[1] 5

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

### filter(): 행추출 (행이 줄어든다)

```r
# 1,2,3 반에 해당하면 추출
df_exam %>% 
  filter(class == 1|class == 2|class == 3)
df_exam %>% # %in% 으로도 표현가능 
  filter(class %in% c(1,2,3))
# A in B 는 A가 B에 들어가 있는거지만
# A %in% B 은 B가 A안에 있는 것이다
```

### select() 열 추출

```r
select(df_exam, math, english, science)
# 파이프 응용
df_exam %>%
  select(math,english,science)

df_exam %>% # math 제외
  select(-math)

df_exam %>%
  filter(class == 1) %>% # class가 1인 행 추출
  select(class,math) # english,math 열 추출
  ```
  
### arrange() 줄세우기

```r
df_exam %>% arrange(math) #오름차순
df_exam %>% arrange(desc(math)) #내림차순

# 반, 수학은 내림차순 정렬, 영어는 오름차순 정렬

df_exam %>% 
  select(class,math,english)%>% 
  arrange(desc(math),english)
```

### mutate 파생변수( 변수 추가, 열 추가)

```r
df_exam %>% 
  mutate(total = math+english+science) # 종합 변수 추가

df_exam %>% 
  mutate(total = math+english+science,
         mean = total/3) %>% 
  arrange(desc(mean)) %>% 
  head(6)

# ifelse(조건, 참일때, 거짓일때)
df_exam %>% 
  mutate(test = ifelse(science >= 60,"pass,","fail"))
```

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

### summarise() 집단별로 요약하기

```r
df_exam %>% 
  summarise(수학평균 = mean(math))
# 이렇게 쓰여서는 summarise를 쓰는 의미가 없다

# 반별 수학 평균은?

df_exam %>% 
  group_by(class) %>% # summarise는 group_by와 같이 쓰인다
  summarise(수학평균 = mean(math))

df_exam %>% 
  group_by(class) %>% 
  summarise(mean_math = mean(math),
            sum_math = sum(math),
            median_math = median(math),
            빈도수 = n())
```

### ggplot에서 제공하는 mpg

```r
library(ggplot2)
mpg
str(mpg) # 살펴보기

# 제조사 열 보기
mpg$manufacturer
unique(mpg$manufacturer) # 원소 중첩 제거
table(mpg$manufacturer) # 원소 갯수
# 열 보기
colnames(mpg) 

# 연습
mpg %>% 
  group_by(manufacturer) %>% # 회사별로 분리
  filter(class == "suv") %>% # suv 추출
  mutate(tot = (cty + hwy)/2) %>% # 통합 연비 변수
  summarise(mean_tot = mean(tot)) %>% # 통합 연비 평균 
  arrange(desc(mean_tot)) %>% # 내림차순 정렬
  head(5) # 일부 출력

cheatsheet # cheatsheet=답안지
```

### join() 열 합치기

```r
# 중간고사 데이터 생성
test1 <- data.frame(id = c(1,2,3,4,5),
                    midterm = c(60,80,70,90,85))
# 기말고사 데이터 생성
test2 <- data.frame(id = c(1,2,3,4,5),
                    final = c(70,83,65,95,80))
# test값 출력
test1 
test2
# id를 기준으로 합쳐 total에 할당
total <- left_join(test1,test2, by = "id")
total # total 출력
# 반별 선생 이름
name <- data.frame(class = c(1,2,3,4,5),
                   teacher = c("kim", "lee", "park","choi","woo"))
name
# join을 이용하여 생성한 이름 매치
exam_new <- left_join(df_exam, name, by = "class")
exam_new
```

### bind_row() 행 합치기

```r
rbind(test1,test2, make.row.name=F) #에러
cbind(test1, test2) # 성공, 열은 상관없지만 행은 같은 열이여야 합쳐진다
colnames(test2)[2] =  "midterm" # 열 맞추기
test2
test1
rbind(test1,test2)
test2 <- test2 %>% rename(final = midterm)
test2
rbind(test1,test2)
bind_rows(test1,test2)
# rbind는 열의 이름이 다르면 실행이 안돼지만
# bind_rows는 열의 이름이 달라도 실행이 된다.
bind_cols(test1,test2)

test3 <- data.frame(id = c(1,2,3,4,5,6),
                    midterm = c(60,80,70,90,85,90))

test4 <- data.frame(id = c(1,2,3,7,8,9),
                    final = c(70,83,65,95,80,99))
```

### 데이터 프레임 합치기

```r
데이터 프레임 합치기
```
