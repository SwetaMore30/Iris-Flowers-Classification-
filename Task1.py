#!/usr/bin/env python
# coding: utf-8

# # >LetsGrowMore
# **>Name - Sweta More**
# 
# **>Data Science Internship**
# 
# **>Task-1**
# **Iris Flowers Classification ML Project**

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import sklearn 


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# In[3]:


url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'


# In[4]:


column_name = ['sepal_length' , 'sepal_width', 'petal_length' , 'petal_width', 'class' ]


# In[5]:


dataset = pd.read_csv(url, names= column_name)


# In[6]:


dataset.shape


# In[7]:


dataset.head()


# In[8]:


dataset.info()


# In[9]:


dataset.describe()


# In[10]:


dataset['class'].value_counts()


# In[11]:


sns.violinplot(y='class', x='sepal_length', data= dataset, inner='quartile')
plt.show()
sns.violinplot(y='class', x='sepal_width', data= dataset, inner='quartile')
plt.show()
sns.violinplot(y='class', x='petal_length', data=dataset, inner='quartile')
plt.show()
sns.violinplot(y='class', x='petal_width', data=dataset, inner='quartile')
plt.show()


# In[12]:


sns.pairplot(dataset, hue='class' , markers='+')
plt.show()


# In[13]:


plt.figure(figsize=(7,5))
sns.heatmap(dataset.corr(),
annot=True , cmap='cubehelix_r')
plt.show


# In[14]:


x = dataset.drop(['class'],axis=1)
y = dataset['class']
print(f'x shape: {x.shape} | y shape:{y.shape} ')


# In[15]:


x_train, x_test ,y_train, y_test = train_test_split(x,y, test_size=0.20 , random_state=1)


# In[16]:


models = []

models.append(('LR',
LogisticRegression()))
models.append(('LDA',
LinearDiscriminantAnalysis()))
models.append(('KNN',
KNeighborsClassifier()))
models.append(('CART',
DecisionTreeClassifier()))
models.append(('NB',
GaussianNB()))
models.append(('SVC',
SVC(gamma='auto')))

# evaluate each model in turn

results = []

model_names = []

for name, model in models:
 kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

results.append(cv_results)

model_names.append(name)

print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[17]:


model = SVC(gamma='auto')
model.fit(x_train , y_train)

prediction= model.predict(x_test)


# In[18]:


print(f'Test Accuracy: {accuracy_score(y_test, prediction)}')
print(f'Classification Report:\n {classification_report(y_test,prediction)}')

