
# Matrix 행렬

matrix(1:12) # R에서는 (1:12)가 1부터 11까지가 아니라 1부터 12까지다

# 명령어 살펴보기
help(matrix)
?matrix

matrix(1:12, nrow = 3)
# byrow = T 옵션은 데이터가 행 먼저 들어가는 기능이다
mat <- matrix(1:12, nrow = 3, byrow = T)
#행이나 열의 이름을 설정할땐 rownames(), colnames()를 사용한다
rownames(mat) <- c("국어", "영어", "수학") 
colnames(mat) <- c("a1", "a2", "a3", "a4")
mode(mat) # "numeric"
class(mat) # "matrix" "array" 

#행렬에서 특정 원소만 출력하고 싶을 때 인덱스를 활용한다.
#행렬에서 인덱스를 활용할때는 대괄호[]를 사용한다
mat[3,4]
mat[ ,4]
mat[3, ]
#특정 행 또는 열을 제외하고 싶을땐 (-)마이너스를 사용한다
mat[-2, ]
mat[-1,]
mat[-3,]
mat[,2:4]
mat[1:2,2:4]
mat[c(1,3), c(2,4)]
mat[-2,c(2,4)]

t(mat) # 행과 열을 바꾼다(전치행렬)