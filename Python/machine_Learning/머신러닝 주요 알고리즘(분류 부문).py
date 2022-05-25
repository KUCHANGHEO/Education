#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("diabetes.csv")


# In[3]:


df


# In[4]:


X = df.drop('Outcome', axis = 1)


# In[5]:


y = df['Outcome']


# In[6]:


X


# In[7]:


y


# In[8]:


scaler = StandardScaler()


# In[11]:


scaler_X = scaler.fit_transform(X)


# In[13]:


scaled_X = pd.DataFrame(scaler_X, columns = X.columns)


# In[80]:


scaled_X.describe()


# In[17]:


fig, ax = plt.subplots(1,2, figsize=(12,4))
X.plot(kind='kde',title='Raw Data', ax= ax[0])
scaled_X.plot(kind='kde',title='StandardScaler', ax= ax[1])
plt.show()


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(scaled_X,y, test_size=0.25, random_state=0)


# In[37]:


X_train.shape


# In[38]:


X_test.shape


# # Decision Tree

# In[39]:


from sklearn.tree import DecisionTreeClassifier # 결정 트리 분류기를 불러옴
from sklearn.model_selection import StratifiedKFold, cross_val_score


# In[40]:


Classifier = DecisionTreeClassifier()


# In[41]:


Classifier.fit(X_train,y_train)


# In[42]:


y_pred = Classifier.predict(X_test)


# In[43]:


skf = StratifiedKFold(n_splits=10, shuffle=True)


# In[45]:


accuracies = cross_val_score(estimator=Classifier, X= X_train, y = y_tain, cv = skf)


# In[52]:


print("Accuracy : {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation : {:.2f} %".format(accuracies.std()*100))


# # Random Forest

# In[53]:


from sklearn.ensemble import RandomForestClassifier


# In[60]:


Classifier = RandomForestClassifier(n_estimators= 100)


# In[61]:


Classifier.fit(X_train,y_train)


# In[62]:


y_pred = Classifier.predict(X_test)


# In[63]:


skf = StratifiedKFold(n_splits=10, shuffle=True)


# In[64]:


accuracies = cross_val_score(estimator=Classifier, X= X_train, y = y_tain, cv = skf)


# In[65]:


print("Accuracy : {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation : {:.2f} %".format(accuracies.std()*100))


# # 가우시안 나이브 베이즈

# In[66]:


from sklearn.naive_bayes import GaussianNB
Classifier = GaussianNB()
Classifier.fit(X_train,y_train)
y_pred = Classifier.predict(X_test)
skf = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = cross_val_score(estimator=Classifier, X= X_train, y = y_tain, cv = skf)
print("Accuracy : {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation : {:.2f} %".format(accuracies.std()*100))


# # K-NN

# In[67]:


from sklearn.neighbors import KNeighborsClassifier
Classifier = KNeighborsClassifier()
Classifier.fit(X_train,y_train)
y_pred = Classifier.predict(X_test)
skf = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = cross_val_score(estimator=Classifier, X= X_train, y = y_tain, cv = skf)
print("Accuracy : {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation : {:.2f} %".format(accuracies.std()*100))


# # AdaBoost

# In[68]:


from sklearn.ensemble import AdaBoostClassifier
Classifier = AdaBoostClassifier()
Classifier.fit(X_train,y_train)
y_pred = Classifier.predict(X_test)
skf = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = cross_val_score(estimator=Classifier, X= X_train, y = y_tain, cv = skf)
print("Accuracy : {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation : {:.2f} %".format(accuracies.std()*100))


# # 이차 판별 분석

# In[69]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
Classifier = QuadraticDiscriminantAnalysis()
Classifier.fit(X_train,y_train)
y_pred = Classifier.predict(X_test)
skf = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = cross_val_score(estimator=Classifier, X= X_train, y = y_tain, cv = skf)
print("Accuracy : {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation : {:.2f} %".format(accuracies.std()*100))


# # SVM

# In[70]:


from sklearn.svm import SVC
Classifier = SVC(kernel='linear')
Classifier.fit(X_train,y_train)
y_pred = Classifier.predict(X_test)
skf = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = cross_val_score(estimator=Classifier, X= X_train, y = y_tain, cv = skf)
print("Accuracy : {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation : {:.2f} %".format(accuracies.std()*100))


# # SVM -RBF

# In[73]:


from sklearn.svm import SVC
Classifier = SVC(kernel='rbf')
Classifier.fit(X_train,y_train)
y_pred = Classifier.predict(X_test)
skf = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = cross_val_score(estimator=Classifier, X= X_train, y = y_tain, cv = skf)
print("Accuracy : {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation : {:.2f} %".format(accuracies.std()*100))


# # Votting

# In[76]:


from sklearn.ensemble import VotingClassifier

clf1 = AdaBoostClassifier()
clf2 = RandomForestClassifier()
clf3 = SVC(kernel= 'linear')
Classifier = VotingClassifier(estimators=[('lr',clf1),('rf',clf2),('gnb',clf3)])
Classifier.fit(X_train,y_train)
y_pred = Classifier.predict(X_test)
skf = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = cross_val_score(estimator=Classifier, X= X_train, y = y_tain, cv = skf)
print("Accuracy : {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation : {:.2f} %".format(accuracies.std()*100))


# # Bagging

# In[77]:


from sklearn.ensemble import BaggingClassifier
Classifier = BaggingClassifier()
Classifier.fit(X_train,y_train)
y_pred = Classifier.predict(X_test)
skf = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = cross_val_score(estimator=Classifier, X= X_train, y = y_tain, cv = skf)
print("Accuracy : {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation : {:.2f} %".format(accuracies.std()*100))


# # 여러 알고리즘 성능을 한눈에 비교하기

# In[78]:


# 사용할 분류기를 모두 지정합니다.
classifiers = [DecisionTreeClassifier(),
               RandomForestClassifier(),
               GaussianNB(),
               KNeighborsClassifier(),
               SVC(kernel = 'linear'),
               SVC(kernel = 'rbf'),
               AdaBoostClassifier(),
               QuadraticDiscriminantAnalysis(),
               VotingClassifier(estimators=[('1', AdaBoostClassifier()), 
                                            ('2', RandomForestClassifier()), 
                                            ('3', SVC(kernel = 'linear'))]),
               BaggingClassifier(base_estimator=clf3, n_estimators=10, random_state=0)
              ]

# 각 분류기의 이름을 지정합니다. 
classifier_names = ['D_tree',
                    'RF', 
                    'GNB', 
                    'KNN', 
                    'Ada',
                    'QDA',
                    'SVM_l',
                    'SVM_k',
                    'Voting',
                    'Bagging'
                   ]

# 결과가 저장될 리스트를 만듭니다.
modelaccuracies = []
modelmeans = []
modelnames = []

# 각 분류기를 실행하여 결과를 저장합니다. 
classifier_data=zip(classifier_names, classifiers)
for classifier_name, classifier in classifier_data:
    # 계층별 교차 검증 환경을 설정합니다. 
    skf=StratifiedKFold(n_splits=10, shuffle=True)
    # 교차 검증을 통해 정확도를 계산합니다. 
    accuracies=cross_val_score(classifier, X = X_train, y = y_train, cv = skf)
    # 정확도의 평균값을 출력합니다.
    print("Mean accuracy of", classifier_name, ": {:.2f} %".format(accuracies.mean()*100))
    # 결과를 저장합니다.
    modelaccuracies.append(accuracies)
    modelnames.append(classifier_name)
    modelmeans.append(accuracies.mean()*100)  

# 각 분류기별 정확도의 평균값을 막대 그래프로 출력합니다.
plt.figure(figsize=(10,5))    
plt.ylim([60, 80])
plt.bar(modelnames, modelmeans);

# 각 분류기별 결과를 Box 그래프로 출력합니다.
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.boxplot(modelaccuracies)
ax.set_xticklabels(modelnames)
plt.show()


# In[ ]:




