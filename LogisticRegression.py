#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,RidgeCV,LassoCV,ElasticNet,ElasticNetCV,LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import pickle


# In[4]:


from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score


# In[7]:


df= pd.read_csv(r"C:\Users\nagar\OneDrive\Desktop\Ineuron\diabets\diabetes (1).csv")


# In[8]:


df


# In[9]:


df.dtypes


# In[10]:


df.isna().sum()



# In[40]:


df.describe()


# In[41]:


pf=ProfileReport(df)
pf.to_widgets()
#filling Zeros using mean imputation
df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())
df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())
df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].mean())

# In[13]:





# In[42]:


df.columns
   
    


# In[37]:





# In[38]:





# In[43]:




# In[45]:


sns.boxplot(df)


# In[49]:
#lets treat the outliers using winsarization technique
#!pip install feature_engine
from feature_engine.outliers import Winsorizer
winsor_iqr=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables='Insulin')
df['Insulin']=winsor_iqr.fit_transform(df[['Insulin']])
# In[51]:
winsor_iqr=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables='Pregnancies')
df['Pregnancies']=winsor_iqr.fit_transform(df[['Pregnancies']])

winsor_iqr=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables='BloodPressure')
df['BloodPressure']=winsor_iqr.fit_transform(df[['BloodPressure']])  

winsor_iqr=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables='SkinThickness')
df['SkinThickness']=winsor_iqr.fit_transform(df[['SkinThickness']])    
sns.boxplot(df)

winsor_iqr=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables='BMI')
df['BMI']=winsor_iqr.fit_transform(df[['BMI']])    
sns.boxplot(df)

winsor_iqr=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables='DiabetesPedigreeFunction')
df['DiabetesPedigreeFunction']=winsor_iqr.fit_transform(df[['DiabetesPedigreeFunction']])    
sns.boxplot(df)

winsor_iqr=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables='Age')
df['Age']=winsor_iqr.fit_transform(df[['Age']])    
sns.boxplot(df)

df


p=ProfileReport(df)
p.to_file("outputAfter.html")  
import os
os.getcwd()
sns.boxplot(df)


x=df.drop(columns='Outcome')
x
y=df['Outcome']
y=pd.DataFrame(y)

column_names=df.columns.tolist()
scaler=StandardScaler()
df_new=scaler.fit_transform(df)
scaled_df=pd.DataFrame(df_new,columns=column_names)
scaled_df.columns
sns.boxplot(scaled_df)
# In[ ]:




winsor_iqr=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables='BloodPressure')
scaled_df['BloodPressure']=winsor_iqr.fit_transform(scaled_df[['BloodPressure']])
    
winsor_iqr=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables='BMI')
scaled_df['BMI']=winsor_iqr.fit_transform(scaled_df[['BMI']])  

# we can create the function for winsrizor as well
def winsarization_func(culumnName,dataFrame) :
    winsor_iqr=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=[culumnName])
    dataFrame[culumnName]=winsor_iqr.fit_transform(dataFrame[[culumnName]])
    return dataFrame

scaled_df=winsarization_func('BMI', scaled_df)
for i in scaled_df.columns:
    scaled_df=winsarization_func(i, scaled_df)
    
scaled_df
sns.boxplot(scaled_df)
p=ProfileReport(scaled_df)
p.to_file("outputAfterscaled.html") 

X=scaled_df.drop(columns='Outcome')
X
Y=df['Outcome']
Y=pd.DataFrame(Y)
 


#Lets find the inflation factor for original data set

def vif_score(n):
    scaler=StandardScaler()
    arr=scaler.fit_transform(n)
    return pd.DataFrame([[n.columns[i],variance_inflation_factor(arr,i)] for i in range(arr.shape[1])], columns=['FEATURE','VIF'])


vif_score(X)

#split the training and test data

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=144)
x_train
x_test
y_train
y_test

#create Model
model=LogisticRegression()
model.fit(x_train,y_train)
predictions=model.predict(x_test)
predictions
#evaluate the model
from sklearn.metrics import accuracy_score,r2_score,recall_score,f1_score,pair_confusion_matrix,auc,roc_curve
accuracy=accuracy_score(y_test,predictions)
accuracy
rscore=r2_score(y_test, predictions)
rscore
conf_matrix=confusion_matrix(y_test, predictions)
conf_matrix
#lets draw ROC curve
y_prob=model.predict_proba(x_test)[:,1]
fpr,tpr,thresholds=roc_curve(y_test,y_prob)
#calculate roc score
roc_auc=roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,color='darkorange' )
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
