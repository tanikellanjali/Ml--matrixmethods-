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


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


corr = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="Blues", annot=True)


# In[8]:


df['Smartphone'].hist()


# In[9]:


fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
sns.histplot(df, ax=axes[0], x="LTE/5g Category", kde=True, color='r')
sns.histplot(df, ax=axes[1], x="Time", kde=True, color='b')
sns.histplot(df, ax=axes[2], x="Packet Loss Rate", kde=True)
sns.histplot(df, ax=axes[3], x="Packet delay", kde=True)


# In[10]:


def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))
    return prior


# In[11]:


def calculate_likelihood_categorical(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    p_x_given_y = len(df[df[feat_name]==feat_val]) / len(df)
    return p_x_given_y


# In[12]:


def naive_bayes_categorical(df, X, Y):
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_categorical(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 


# In[13]:


df.iloc[:,-1]


# In[ ]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.5, random_state=41)

X_test = test.iloc[:,:-1].values
Y_test = test.iloc[:,-1].values
Y_pred = naive_bayes_categorical(train, X=X_test, Y='Smartphone')


# In[ ]:


from sklearn.metrics import confusion_matrix, f1_score
print(confusion_matrix(Y_test, Y_pred))
print(f1_score(Y_test, Y_pred))


# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(Y_test, Y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(Y_pred), max(Y_test))
p2 = min(min( Y_pred), min(Y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title('Categorical Naive Bayes')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


# In[ ]:




