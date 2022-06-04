# R

## R의 기본 명령어
- Crtl + L : 화면 지우기
- Crtl + R : 스크립트에서 실행
- Crtl + Enter : R스튜디오 에서 스크립트 실행
- a = b 를 R에서는 a ←로 쓴다 단축키는 Alt -
- a ← c(1, 2, 3) R에서는 묶을때 앞에 c라는 함수를 쓴다
- shift + Alt + 아래키 : 해당줄 아래로 복사
- rm() : 객체 제거하기
- Crtl shift M : %>%

## R의 데이터 타입

- 숫자: numeric
- 문자: character   ‘홑따옴표’ 나 “쌍따옴표” 로 표현한다 하지만 ‘혼용”은 안된다
- 참거짓: logical  참거짓은 TRUE FALSE T F 등 대문자로만 표현

## 변수

- obj <- c(3, 5, 7)
- name_1 <- c(3, 5, 7)
- name.2 <- c(3, 5, 7)
- .name2 <- c(3, 5, 7)
- _name2 <- c(3, 5, 7) 에러 이름앞에 _가 오면 안된다
- 2name <- c(3, 5, 7) 에러 이름앞에 숫자가 오면 안된다

그래서 x_2name <- c(3, 5, 7) 앞에 아무의미없는 문자를 준다

## 객체의 타입

if, else for 같은 특정문자는 객체의 이름으로 쓸수 없다

a <- c(1, 2)
mode(a)  numeric

a <- c(1, 2, "a", "b")
mode(a)   character 문자 하나만있어도 char가 나옴

a <- c(1, 2, T, F)
mode(a)  numeric

데이터 타입 우선순위 character > numeric > logical

하나의 벡터에는 하나의 타입만 가져야 한다

## 괄호안의 숫자

matrix(1:12) <- R은 다른 언어와 다르게 1부터 12까지란 뜻이다 11이아니다

## 참거짓

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
