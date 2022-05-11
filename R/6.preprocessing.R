df <-  data.frame(sex = c("M","F",NA,"M","F"),
                  score = c(5,4,3,4,NA))
is.na(df) # 결측치 확인
sum(is.na(df$sex)) # 숫자로 나옴

table(is.na(df)) # 결측치 빈도 출력

# 처리방법 1 데이터를 지워버린다
table(is.na(df$sex)) # sex 결측치 빈도 출력
table(is.na(df$score)) # score 결측치 빈도 출력
summary(df) # 결측치가 나옴
na.omit(df) # NA가 보이기만 하면 행을 다 날린다

mean(df$score) # 평균 산출 : NA라고 뜸
sum(df$score) # 합계 산출 : NA라고 뜸

mean(df$score, na.rm = T) # NA를 지워서 계산됨
sum(df$score, na.rm = T)

df %>% filter(is.na(score)) # score가 NA인 데이터
df %>% filter(!is.na(score)) # score 결측치 제거

# 처리방법 2 다른 값으로 채워넣는다
#다른 값: 평균, 최빈 값, 직전 값, 직후 값 

# 처리방법 3 무시한다
df_nomiss <- df %>% filter(!is.na(score))
mean(df_nomiss$score) # score 평균 산출


# 이상치
outlier <-  data.frame(sex = c(1,2,1,3,2,1),
                       score = c(5,4,3,4,2,16))

table(outlier$sex)
table(outlier$score)

# sex가 3이면 NA 할당
3 <- NA
outlier$sex[4] <-  NA
outlier
outlier$sex[4]
outlier[4,1] 

# score가 5이면 NA 할당
outlier$score <- ifelse(outlier$score > 5 ,NA , outlier$score)
outlier$score

# NA값 무시후 출력
outlier %>%  
  filter(!is.na(sex)& !is.na(score)) %>% 
  group_by(sex) %>% 
  summarise(mean_score = mean(score))

# 상자 그림 통계치 출력
library(ggplot2)
boxplot(mpg$hwy)
boxplot(mpg$cty)
boxplot(mpg$manufacturer) # 숫자가 아니라 출력 불가
str(mpg)
# 12~37 벗어나면 NA 할당
mpg$hwy <- ifelse(mpg$hwy < 12 | mpg$hwy > 37 , NA, mpg$hwy)
table(is.na(mpg$hwy))
# 구동방식(drv)별로 고속도로 주행연비(hwy)의 평균을 구하세요.
mpg %>% 
  group_by(drv) %>% 
  summarise(mean_hwy = mean(hwy, na.rm = T))
# filter를 이용해도 된다
mpg %>%filter(!is.na(hwy)) %>% 
  group_by(drv) %>% summarise(mean(hwy))

# 초기화
rm(mpg)
mpg

### 연습 ###


# displ 4이하 5이상 일때 hwy평균
mpg %>% 
  filter(displ <= 4 | displ >= 5) %>% 
  summarise(mean(hwy)) # mean(hwy) = 24.5 값이 하나로 나온다
# 따로 보고싶다면 따로 실행한다
d4 <- mpg %>% filter(displ <= 4)
d5 <- mpg %>% filter(displ >= 5)
mean(d4$hwy);mean(d5$hwy) # ;같은 줄 이지만 \n 후 실행을 의미한다

# audi,ford 제조사의 cty 평균
mpg %>% filter(manufacturer == 'audi'|manufacturer=='ford') %>% 
  group_by(manufacturer) %>% summarise(mean_cty=mean(cty))
# hwy cty값을 합친 값을 추가하고 내림차순으로 정렬
mpg %>%
  mutate(total = hwy + cty) %>% 
  arrange(desc(total)) %>% 
  head()
  
mpg_new <- select(mpg, class, cty)
# suv나 compact의 cty 평균
mpg_new %>% 
  filter(class == 'suv' | class == 'compact') %>% 
  group_by(class) %>% 
  summarise(mean(cty))
# 제조사 별 compact 차종 빈도수
mpg %>% 
  filter(class == 'compact') %>% 
  group_by(manufacturer) %>% 
  summarise(n = sum(n())) %>% 
  arrange(desc(n))
