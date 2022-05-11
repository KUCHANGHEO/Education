import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

a = np.array(np.random.randint(10,size=40).reshape(5,8))
b = np.array(np.random.randint(10,size=56).reshape(8,7))

a @ b

def sig(x):
    return 1 / (1 + np.exp(-x))

np.sign(1.2)
np.sign(-2.2)


# [-1, 3] 10개 점

x = np.linspace(-1, 3, 20)


plt.plot(x,x**x)

plt.plot(x,x**3 - 3 * x**2 + x)

def f(x):
    return x**2

x = np.linspace(0, 3, 20)

plt.plot(x, np.f(x))


def Relu(x):
    if x >= 0:
        return x
    else:
        return 0

x = np.linspace(-3, 3, 20)

plt.plot(x,Relu(x))
plt.plot(x, np.maximum(x,0))



np.e

np.exp(3)
np.exp(0.1)
np.exp(0)
np.exp(-0.1)
np.exp(-1)

np.exp(x)

plt.plot(x, np.exp(x))

np.log(x)
plt.plot(x, np.log(x))



def f(x):
    return 


