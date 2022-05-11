

# 임포트하기
import numpy as np


np.sum([1,2,3,4])
sum([1,2,3,4])



# 함수 이름을 바로 쓰고 싶을 때
from numpy import var
np.var([1,2,3,4])
var([1,2,3,4])

## Array 만들기

a = np.array([0,1,2,3])
a


a = np.array([0,1.0,2,3])
a


a = [0,1,2,3]
np.array(a)

a = np.array([0,1,2,3])
a.dtype
a.ndim
a.shape
a.size


# 3/4 =8bytes = 64bit


np.array([0,1,2,3,])
np.array([0,1,2.2,3,])

np.array([0,1,2,3], float)
np.array([0,1,2.2,3], int)

np.array([0,1,2,3], 'int8')
np.array([0,1,2,3], dtype = 'int16')


np.array([0,1,2,3], 'float32')
np.array([0,1,2,3], dtype = 'float64')



arr1 = np.array([[1,2],[3,4]])
arr2 = np.array([[3,4],[5,6]])



# 2x2x2 3차원 배열을 만들어 봅시다

np.array([[[1,2],[3,4]],[[5,6],[7,8]]])


for i in range(3):
    print(i)

list(range(3))

for i in np.arange(3):
    print(i)

np.arange(3) # array로 출력
a = np.arange(4)
a.shape

a = np.arange(8)
a.shape
a.reshape(2,4)

np.arange(8).reshape(2,4)



# 10 부터 21 까지의 숫자로 3행 4열 짜리를 만들어 주세요

np.arange(10,22).reshape( (3,4) )


np.arange(6, dtype = np.float32).reshape(2, 3)

# reshape(z,y,x)
np.arange(8).reshape(2,4)
np.arange(8).reshape(2,2,2)
np.arange(24).reshape(2,3,4)
np.arange(24).reshape(3,4,2)
np.arange(24).reshape(4,2,3)




## 배열의 연산



a = [1,2,3,4]
b = [5,6,7,8]

np.array(a) + np.array(b) # element wise

# row wise, colum wise

a = np.array(a)
b = np.array(b)

a*3
type(a)

a + b
np.add(a, b)

c = np.array([10])

a + c


arr3 = np.array([[[1,2],
                  [3,4]],
                 
                 [[5,6],
                  [7,8]]])


arr1 = np.arange(1,5).reshape(2,2)
arr2 = np.arange(5,9).reshape(2,2)
arr1 + arr2
arr1 - arr2
arr1 * arr2
arr1 / arr2

arr1 @ arr2 # 행렬의 곱셈
np.dot(arr1,arr2)

np.arange(36).reshape(3, 4, 3)





np.arange(1, 9 ,2)
a = np.array(9, dtype= np.float32)

np.linspace(0, 1, 5, endpoint= False)

np.ones((3,3))

np.zeros((3, 3))
b[2,0] = 3
b # sparse

np.diag([1,2,3])


np.ones((3,3),int)*8
np.full((3,3),8)

np.diag([0,4,8])

c = np.arange(9).reshape(3,3)
np.diag(c)

np.identity() # 단위 행

np.diag([1,1])
np.identity(2)

np.eye(2)
np.eye(3)
np.eye(3,4)

np.tile([1,2,3], 2)
np.tile(np.array([1,2,3]), 3)

np.tile([1,2,3], (3, 2))

np.empty((2,3))

### like  형태와 데이터 타입은 유지

a = np.arange(6).reshape(2,3)

np.zeros_like(a)

b = np.arange(3, dtype = np.float16)

np.zeros_like(b)
np.ones_like(a)
np.ones_like(b)


### copy와 view

a = np.arange(1,6)
b = a.copy()

b[3] = 0

c = a
c[3] = 0

a = np.arange(1,6)


b = a[ 2:4]
b[0] = 0


np.arange(1,9,2)
np.linspace(0, 1, 6)
np.ones((3,3),int)


### stack

a = np.arange(5)
a*10

np.vstack([a,a * 10])
np.vstack([a,a * 10, a * 20, a -10])
np.vstack([a,a * 10])
np.vstack([a,a * 10])

np.hstack([a,a * 10])


a = np.array([1,2,3])
b = np.array([4,5,6])

np.vstack([a,b])
np.hstack([a,b])
np.dstack([a,b])


a = np.arange(1, 25).reshape((4,6))

np.vsplit(a,2)
type(a)
np.array(np.vsplit(a,2)).shape

np.vsplit(a,3)

np.hsplit(a,3)
np.hsplit(a,2)
np.hsplit(a,4)


np.arange(24).reshape(3,8)
np.arange(24).reshape(3, )
np.arange(24).reshape(3,-1)
np.arange(24).reshape(4,-1)
np.arange(24).reshape(-1,6)

# flatten
a = np.arange(24).reshape(3,8)
a.flatten().shape

# ravel

b = a.ravel()
b[0] = 100
b
a # 같이 바뀌었다 view


a = np.arange(24).reshape(3,8)
a @ a.T


## 배열의 비

a = np.array([1,2,3,4])
b = np.array([3,4,5,6])

a == b

a > b

np.array_equal(a, b)

A = np.array([2,1,4,3])
B = np.sort(A)
B
A

A.sort() # A를 정렬하고 바로 저장
A

a= [2,1,4,3]
b = np.sort(a)
a

sorted(a) # sorted : python의 내장함수
a.sort()
a

a = np.array([4,2,6,5,1,3,0])
a[ : :2]
a[ : :-1]
np.sort(a)[::-1]


x = np.array([[2,1,6],
              [0,7,4],
              [5,3,2]])

np.sort(x)
np.sort(x, axis = 0)
np.sort(x, axis = 1)

## indexing

a = np.array([0,1,2,3])

a.ndim
a.size
a.nbytes
a.dtype

a[0] = 10
a[0]

### 2d indexing
a = np.array([[0,1,2,3],[10,11,12,13]])
a.ndim
a.shape
a.size

a[1,3]
a[1,3] = -1

a[1]
a[1,]

#a[,1] << 빈칸을 인식을 못함/ 대신:로 빈칸의미
a[:,1]

## slicing

### slicing은 원본 View

a = np.arange(25).reshape(5,5)
a[1:3]
a[4::]
a[:,1::2]
a[1::2,0:3:2]



# Fancy Indexing.position

a = np.arange(0, 80, 10)

a[[0,1]]
a[[0,1,3,5]]

indices = [0,1,3,5]

a[indices]
a[indices] = 9


a= np.arange(36).reshape(6,6)
a[1]
a[2]
a[[1,2]]
a[indices]
a[indices, indices]

a[[0,1,3,5],[0,1,3,5]]

a[[0,1,2,2],[1,2,0,3]]


a = np.arange(0,80,10)

a[a>40]
a[(a>40)|(a<30)]

mask = a>40
a[mask]



a= np.arange(36).reshape(6,6)

mask = [a>20,(a<14)|(a>26)]

a[mask]

a = np.arange(25).reshape(5,5)

mask1 = np.array((1,1,1,1,0), dtype = bool)
mask2 = np.array((0,1,1,1,1), dtype = bool)

a[mask1,mask2]


# broad casting


a = np.array([1,2,3])
a + 1


a = np.ones((3,5))
b = np.ones((5, ))
a + b
a

# 배열 수학

a = np.array([[1,2,3],[4,5,6]])

sum(a)
np.sum(a)
np.sum(a, axis = 0)
np.sum(a, axis = 1)

a
np.min(a)
np.min(a, axis = 0)
np.min(a, axis = 1)

b = np.array([[4,5,6],[1,2,3]])
np.minimum(a,b)
np.argmin(a) # 결과 값은 인덱스 번호 
np.argmax(a)  

c = np.array([2,1,3,5,4,2])
np.max(c)
np.argmax(c) # index 위치


a
a > 4

a[a>4]

np.where( a> 4)
np.where( a> 4, "H", "h")


a = np.arange(-15,15).reshape(5,6) ** 2


# list 두개를 더해서 새로 리스트를 만들자

li1= [1,2,3,4]
li2= [1,2,3,4]
li3 = []

for i in range(len(li1)):
    li3.append(li1[i]+li2[i])

print(li3)

import timeit

num = 100000


def test():
    li1 = list(range(1,400000))
    li2 = list(range(1,400000))
    li3 = []
    for i in range(len(li1)):
        li3.append(li1[i]+li2[i])
    print(li3)

def test1():    
    a4= np.arange(1,400000)
    a5= np.arange(1,400000)
    a6 = a4 + a5
    print(a6)
    
timeit.timeit(test1, number = 10)
