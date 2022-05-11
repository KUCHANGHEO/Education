x <- 9
# x <- 12
x

y <- 10

# 객체 제거하기
remove(y)

# R의 사칙연산
5+10/(2+3)
(1/2+1/3)^2/(1/3^2)
5*7


#R의 데이터 타입
#숫자형:numeric
#문자형:char
#참거짓형:logical

a <- 1

typeof(a)
mode(a)

b = "Hello"
typeof(b)
mode(b)

c <- TRUE
mode(c)
typeof(c)

# 이름 설정
obj <- c(3, 5, 7)
name_1 <- c(3, 5, 7)
name.2 <- c(3, 5, 7)
.name2 <- c(3, 5, 7)
# _name2 <- c(3, 5, 7) _가 앞에오면 안된다
# 2name <- c(3, 5, 7) 숫자가 앞에오면 안된다
x_2name <- c(3, 5, 7) # 그래서 의미없는 문자 뒤에 적어준다다

# R에서는 묶음을 할때 c()함수로 묶어준다

a <- c(1, 2) 
mode(a) #"numeric"

a <- c(1, 2, "a", "b")
mode(a) #"character"

a <- c(1, 2)
mode(a)

a <- c(1, 2, T, F)
mode(a) #"numeric"

# 데이터 타입 우선순위 character > numeric > logical

# R의 논리 연산자
A <- T
B <- F
C <- c(T, T)
D <- c(F, T)

A & B #FALSE
C & D #FALSE TRUE R에서는 
A && B #FALSE
C && D #FALSE

A || B #TRUE
C || D #TRUE

# &&  나 || 은 각 벡터의 첫번째 원소만 비교하여 결과를 낸다

A <- c(3, 4)
B <- c(5, 4)
A < B

