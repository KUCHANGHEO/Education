import time

def Hello(count):
    if count == 0:
        return
    print("Hello")
    count -= 1
    Hello(count)
    
Hello(5)



def factorial(num):
    if num == 1:
        return 1
    return num * factorial(num - 1)

print(factorial(5))



def fib(n):
    if n == 1 or n == 2:
        return 1
    return fib(n-1) + fib(n-2)

for i in range(1, 15):
    print(fib(i), end=" ")