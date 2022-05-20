#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# In[2]:


t = np.arange(0, 3, 0.001)    # Time vector
g = [0] *5
g[0] = np.sin(2 * np.pi * t )
g[1] = np.sin(2 * np.pi * t * 3 )
g[2] = np.sin(2 * np.pi * t * 5 )
g[3] = g[0] + g[1] + g[2]
g[4] = 0.3* g[0] - 0.7 * g[1] + 0.5* g[2]


# In[3]:


titles = ['1Hz', '3Hz','5Hz','sum','weighted sum']
plt.figure(figsize=(13,6))
for i, title in enumerate(titles):
    plt.subplot(2, 3, i+1), plt.plot(t, g[i]), plt.title(title)
plt.tight_layout()
plt.show()


# In[ ]:




