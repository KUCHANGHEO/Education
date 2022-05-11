# 데이터 가공
install.packages("dplyr") # d ply r :디플라이어
library(dplyr)
library(readxl)
df_exam = read_excel("excel_exam.xlsx")
# 5 verbs 
filter() # 행 추출
select() # 열 추출
arrange() # 정렬
mutate() # 변수 추가
summarise() # 통계치 산출

group_by() # 집단별로 나누기
left_join() # 데이터 합치기(열)
bind_rows() # 데이터 합치기(행)


df_exam[df_exam$class=="1",] # 행의 조건을 df_exam의 class 1 

filter(df_exam,class == 1) # filter를 이용한 방법
filter(df_exam,class == 2) # 2반
filter(df_exam,class != 1) # 1반이 아닌것
filter(df_exam,class==1 | class==2)# 1반 아니면 2반
filter(df_exam,math >= 50 & english>=80)
# 수학이 50점 이상이고 영어도 80점 이상

# pipe
# %>% 단축기 Crtl shift M
df_exam %>% filter(class == 1)
# 유닉스 계열에서 쓰이는 기법 |
df_exam %>%
  filter(class ==1| class == 2) %>%
  filter(math > 50) %>%
  filter(english >= 90)
# filter(): 행추출 (행이 줄어든다)
# 1,2,3 반에 해당하면 추출
df_exam %>% 
  filter(class == 1|class == 2|class == 3)
df_exam %>% # %in% 으로도 표현가능 
  filter(class %in% c(1,2,3))
# A in B 는 A가 B에 들어가 있는거지만
# A %in% B 은 B가 A안에 있는 것이다

# select() 열 추출
select(df_exam, math, english, science)
# 파이프 응용
df_exam %>%
  select(math,english,science)

df_exam %>% # math 제외
  select(-math)

df_exam %>%
  filter(class == 1) %>% # class가 1인 행 추출
  select(class,math) # english,math 열 추출

# 줄세우기 arrange
df_exam %>% arrange(math) #오름차순
df_exam %>% arrange(desc(math)) #내림차순

# 반, 수학은 내림차순 정렬, 영어는 오름차순 정렬

df_exam %>% 
  select(class,math,english)%>% 
  arrange(desc(math),english)

# 파생변수 mutate (변수 추가 ,열 추가)

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

# 요약 통계량 함수

df_exam$math
sum(df_exam$math) # 합계
sd(df_exam$math) # 표준 편차
mean(df_exam$math) # 평균
median(df_exam$math) # 중앙값
min(df_exam$math) # 최솟값
max(df_exam$math) # 최댓값
n(df_exam$math) # 빈도

# summarise 집단별로 요약하기

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

# ggplot에서 제공하는 mpg
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

filter(mtcars, mpg>20)

# join 데이터 프레임 합치기

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

inner_join(test3,test4) # AND와 같이 모두 TRUE 여야 합침
left_join(test3,test4) # 왼쪽열 기준으로 합침
right_join(test3,test4) # 오른쪽열 기준으로 합침
full_join(test3,test4) # OR과 같이 모두 TRUE가 아니더라도 합침
