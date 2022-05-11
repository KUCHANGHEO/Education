#한 셀에 여러가지 원소가 들어간것 벡터

#여러셀에 나눠저 들어가는것 데이터 프레임
 #구조는 행렬(matrix)와 비슷하나 각 열마다 다른 데이터 타입을 구상할수 있다
# 각열의 길이는 모두 같아야 한다
- c(100, 80, 60, 40, 20)
x2 <- c("A", "B", "C", "A", "B")
a <- cbind(x1, x2)
rbind(x1, x2)
mode(a)
cla #"character"ss(a)
A < #"matrix" "array" - data.frame(a)
class(A)

ro #"data.frame"wrownames(A) <- c("a1","a2","a3","a4","a5")
colnames(A) <- c('score','grade')e(A)


# # "list" �# data.frame 으로 한번에 데이터 프레임 만들기�0, 80, 60, 40, 20)
x2 <- c("A", "B", "C", "A", "B")
df <- data.frame(score=x1,grade=x2)
df[,1]
df$score
 # 행렬과 같이 []로 접근str(df)
d # $은 열(column)을 나타낸다f <-  da # 데이터 구조 보기t#stringAsFactors = FALSE 일땐 글자로만 읽는다
#stringAsFactors = TRUE 일땐 등급으로 따진다
#등급별로,성별별로,세대별로,부서별로,학년별로 등등을 '범주'라고한다
a.frame(score=x1,grade=x2,stringsAsFactors = T)

rm(list = # 모든 객체 지우기
ls())
