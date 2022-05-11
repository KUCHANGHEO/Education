# Thread 쓰레드
# porcess "exe"
'''
from concurrent.futures import thread
import time

def long_task():
    for i in range(5):
        time.sleep(1)
        print("workingL%s\n" % i)

print("Start")

for i in range(5):
    long_task()

print("End")
'''
import threading
import time
def long_task2():
    for i in range(5):
        time.sleep(1)
        print("working:%s" % i)
print("Start")

thread = []

for i in range(5):
    t = threading.Thread(target=long_task2)
    thread.append(t)

for t in thread:
    t.start()

for t in thread:
    t.join()

print("End")