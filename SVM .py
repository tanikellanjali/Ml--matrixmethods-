#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
# from sklearn import cross_validation
from sklearn import preprocessing 
from sklearn import svm 
from sklearn import metrics 
from matplotlib import colors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

import seaborn as sns


# In[2]:


df = pd.read_csv('Network_Slicing_Recognition.csv')


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[6]:


plt.figure(figsize=(16, 30))
ax = sns.displot(data=df, x="Packet Loss Rate", y="Packet delay", kind = 'kde',height = 7 )
g.ax.axline(xy1=(10, 10), slope=.2, color="b", dashes=(5, 2))


# In[7]:


plt.figure(figsize=(16, 30))
ax = sns.displot(data=df,x="Time", kind = 'kde',height = 7 )
g.ax.axline(xy1=(10, 10), slope=.2, color="b", dashes=(5, 2))


# In[8]:


plt.figure(figsize=(16, 30))
ax = sns.displot(data=df,x="Packet delay", kind = 'kde',height = 7 )
g.ax.axline(xy1=(10, 10), slope=.2, color="b", dashes=(5, 2))


# In[9]:


plt.figure(figsize=(16, 30))
ax = sns.displot(data=df,x="LTE/5g Category", kind = 'kde',height = 7 )
g.ax.axline(xy1=(10, 10), slope=.2, color="b", dashes=(5, 2))


# In[10]:


corr = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="Blues", annot=True)


# In[11]:


X = df.iloc[:, [2,4]].values
y = df.iloc[:, [3]].values


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)


# In[13]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[14]:


classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


# In[15]:


y_pred = classifier.predict(X_test)


# In[16]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[17]:


plt.figure(figsize = (10,10))
plt.plot(cm)


# In[18]:


plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')
p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title('SVM')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[ ]:




