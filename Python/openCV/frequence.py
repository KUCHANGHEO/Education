#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# In[2]:


t = np.arange(0, 1, 0.001)    # Time vector
Hz = [1, 2, 10, 100]
gs = [np.sin(2 * np.pi * t * h) for h in Hz]


# In[5]:


plt.figure(figsize=(10,5))
for i, g in enumerate(gs):
    plt.subplot(2, 2, i+1), plt.plot(t, g)
    plt.xlim(0, 1),  plt.ylim(-1, 1 )
    plt.title("frequency: %3d Hz" % Hz[i])
plt.tight_layout()
plt.show()

