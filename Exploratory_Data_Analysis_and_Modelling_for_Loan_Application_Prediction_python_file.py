#!/usr/bin/env python
# coding: utf-8

# # Prediction of status for Loan application / Mohammed Ismail Ghouse / 4036280

# # Generating first algorithm's index:

# In[1]:


Student_ID = 4036280;
Primary_algorithm_index = Student_ID % 6
print(Primary_algorithm_index)


# # Data preperation: Importing and Cleaning dataset

# In[2]:


pip install missingno


# In[3]:


pip install imblearn


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import warnings
import os
import scipy
import sklearn.metrics as metrics


from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


# # Importing dataset

# In[5]:


dataset = pd.read_csv('E:\MSc_Data_Science\Semester 2_V2.0\Statistical_Analysis\Coursework1\Coursework_1\Dataset\Coursework1_dataset.csv')

print(dataset)


# # Data Exploration

# In[6]:


dataset.head()


# In[7]:


dataset.tail()


# In[8]:


dataset.shape


# In[9]:


dataset.info()


# In[10]:


dataset.columns


# # Checking for null values is the dataset

# In[11]:


dataset.value_counts(dropna= True)


# # Dealing with null values

# In[12]:


dataset.isnull().sum()


# # Dealing with null values of attributes which has Categorical datatype

# In[13]:


dataset["Gender"].fillna(dataset["Gender"].mode()[0], inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0],inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)
dataset['SelfEmployed'].fillna(dataset['SelfEmployed'].mode()[0],inplace=True)
dataset['CreditHistory'].fillna(dataset['CreditHistory'].mode()[0],inplace=True)
dataset['LoanAmountTerm'].fillna(dataset['LoanAmountTerm'].mode()[0],inplace=True)


# # Dealing with null values of attributes which has Numerical datatype

# In[14]:


dataset['Loan_Amount'].fillna(dataset['Loan_Amount'].mean(),inplace=True)


# # Checking for null values again

# In[15]:


dataset.isnull().sum()


# In[16]:


dataset[['Applicant_Income','Coapplicant_Income','Loan_Amount']].describe()


# # Data Analysis:

# # Analyzing with single attributes

# # Exploring the Gender attribute

# In[17]:


sns.countplot( x="Gender",data=dataset, hue="Married")


# In[18]:


Num_male_applicants = len(dataset[dataset.Gender=="Male"])
print (Num_male_applicants)
Num_female_applicants = len(dataset[dataset.Gender=="Female"])
print (Num_female_applicants)


# In[19]:


married_applicants = dataset.groupby(["Gender","Married"]).size()
print (married_applicants)


# # Exploring Loan term attribute

# In[20]:


sns.countplot(y="LoanAmountTerm", data= dataset, palette="tab10")


# In[21]:


dataset.LoanAmountTerm.value_counts(dropna= True)


# In[22]:


term12 = len(dataset[dataset.LoanAmountTerm == 12.0])
term36 = len(dataset[dataset.LoanAmountTerm == 36.0])
term60 = len(dataset[dataset.LoanAmountTerm == 60.0])
term84 = len(dataset[dataset.LoanAmountTerm == 84.0])
term120 = len(dataset[dataset.LoanAmountTerm == 120.0])
term180 = len(dataset[dataset.LoanAmountTerm == 180.0])
term240 = len(dataset[dataset.LoanAmountTerm == 240.0])
term300 = len(dataset[dataset.LoanAmountTerm == 300.0])
term360 = len(dataset[dataset.LoanAmountTerm == 360.0])
term480 = len(dataset[dataset.LoanAmountTerm == 480.0])
termNull = len(dataset[dataset.LoanAmountTerm.isnull()])

print("Percentage of 12: {:.2f}%".format((term12 / (len(dataset.LoanAmountTerm))*100)))
print("Percentage of 36: {:.2f}%".format((term36 / (len(dataset.LoanAmountTerm))*100)))
print("Percentage of 60: {:.2f}%".format((term60 / (len(dataset.LoanAmountTerm))*100)))
print("Percentage of 84: {:.2f}%".format((term84 / (len(dataset.LoanAmountTerm))*100)))
print("Percentage of 120: {:.2f}%".format((term120 / (len(dataset.LoanAmountTerm))*100)))
print("Percentage of 180: {:.2f}%".format((term180 / (len(dataset.LoanAmountTerm))*100)))
print("Percentage of 240: {:.2f}%".format((term240 / (len(dataset.LoanAmountTerm))*100)))
print("Percentage of 300: {:.2f}%".format((term300 / (len(dataset.LoanAmountTerm))*100)))
print("Percentage of 360: {:.2f}%".format((term360 / (len(dataset.LoanAmountTerm))*100)))
print("Percentage of 480: {:.2f}%".format((term480 / (len(dataset.LoanAmountTerm))*100)))
print("Missing values percentage: {:.2f}%".format((termNull / (len(dataset.LoanAmountTerm))*100)))


# # Analyzing with two or more attributes

# # Applying crosstabs to find relations

# In[23]:


pd.crosstab(dataset.PropertyArea,dataset.LoanStatus).plot(kind="bar", stacked=True, figsize=(5,5), color=['#f64f59','#12c2e9'])
plt.title('Comparision of Loan Status with repesct to areas ')
plt.xlabel('Property Area')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


# In[24]:


pd.crosstab(dataset.SelfEmployed,dataset.CreditHistory).plot(kind="bar", stacked=True, figsize=(5,5), color=['#333333','#dd1818'])
plt.title('Credit history comparision within workingand Self-employed applicants')
plt.xlabel('Self_Employed_Applicants')
plt.ylabel('Counts')
plt.legend(["Bad_Credit_Score", "Good_Credit_Score"])
plt.xticks(rotation=0)
plt.show()


# # Data exploration using Histogram

# In[25]:


sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

sns.histplot(data=dataset, x="Applicant_Income", kde=True, ax=axs[0, 0], color='red')
sns.histplot(data=dataset, x="Coapplicant_Income", kde=True, ax=axs[0, 1], color='black')
sns.histplot(data=dataset, x="Loan_Amount", kde=True, ax=axs[1, 0], color='blue');


# # Data Exploration using Boxplot

# In[26]:


sns.boxplot(x="LoanStatus", y="Applicant_Income", data=dataset, palette="RdPu");


# In[27]:


sns.boxplot(x="Coapplicant_Income", y="LoanStatus", data=dataset, palette="PuRd");


# In[28]:


sns.boxplot(x="Loan_Amount", y="LoanStatus", data=dataset, palette="RdPu");


# In[29]:


sns.boxplot(x="LoanStatus", y="CreditHistory", data=dataset, palette="mako");


# In[30]:


sns.boxplot(x="LoanStatus", y="LoanAmountTerm", data=dataset, palette="mako");


# # Pearson's Correlation

# In[31]:


dataset.plot(x='Applicant_Income', y='Coapplicant_Income', style='*',color='black')  
plt.title('Correlation between Applicants and Co-applicants Incomes')  
plt.xlabel('ApplicantIncome')
plt.ylabel('CoapplicantIncome')  
plt.show()
print('Correlation:', dataset['Applicant_Income'].corr(dataset['Coapplicant_Income']))
print(stats.ttest_ind(dataset['Applicant_Income'], dataset['Coapplicant_Income']))


# # Removing and sepearting features from the dataset

# In[32]:


dataset1 = dataset.drop(['LoanID'], axis = 1)


# In[33]:


dataset1.shape


# # Applying Heatmap to find a Correlation

# In[34]:


plt.figure(figsize=(10,7))
sns.heatmap(dataset.corr(), annot=True, cmap='viridis');


# # Applying One-hot Encoding Before dealing with outliers

# In[35]:


dataset1 = pd.get_dummies(dataset1)

dataset1 = dataset1.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 
              'SelfEmployed_No', 'LoanStatus_N'], axis = 1)

new = {'Gender_Male': 'Gender', 'Married_Yes': 'Married', 
       'Education_Graduate': 'Education', 'SelfEmployed_Yes': 'SelfEmployed',
       'LoanStatus_Y': 'LoanStatus'}
       
dataset1.rename(columns=new, inplace=True)


# # Dealing with Skewness by applying Square-root Transformation

# In[36]:


dataset1.Applicant_Income = np.sqrt(dataset1.Applicant_Income)
dataset1.Coapplicant_Income = np.sqrt(dataset1.Coapplicant_Income)
dataset1.Loan_Amount = np.sqrt(dataset1.Loan_Amount)


# # Removing Outliers from the dataset

# In[37]:


Q1 = dataset1.quantile(0.25)
Q3 = dataset1.quantile(0.75)
IQR = Q3 - Q1

dataset2 = dataset1[~((dataset1<(Q1-1.5*IQR))|(dataset1>(Q3+1.5*IQR))).any(axis=1)]


# # Normalizing data before applying models

# In[38]:


A=dataset2.drop(['LoanStatus'], axis=1)
b=dataset2['LoanStatus']


# In[39]:


A= MinMaxScaler().fit_transform(A)


# # Applying Smote to deal with overfitting

# In[40]:


A, b = SMOTE().fit_resample(A, b)


# In[41]:


sns.set_theme(style="whitegrid")
sns.countplot(y= b, data=dataset2)
plt.ylabel('Loan Status')
plt.xlabel('Total counts')
plt.show()


# In[42]:


sns.set(style="whitegrid")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

sns.histplot(data=dataset2, x="Applicant_Income", kde=True, ax=axs[0, 0], color='red')
sns.histplot(data=dataset2, x="Coapplicant_Income", kde=True, ax=axs[0, 1], color='black')
sns.histplot(data=dataset2, x="Loan_Amount", kde=True, ax=axs[1, 0], color='blue');


# # Splitting dataset for training and testing before applying models

# In[43]:


A_train, A_test, b_train, b_test = train_test_split(A, b, test_size = 0.2,random_state=1)


# # Training dataset through primary and secondary algorithm

# # 1. Primary algorithm- Support Vector Machine [SVM]

# In[59]:


SVCclassifier = SVC(kernel='rbf', max_iter=500)
SVCclassifier.fit(A_train, b_train)

SVM_pred = SVCclassifier.predict(A_test)
print("Classification Report:")
print(classification_report(b_test, SVM_pred))
print("Confusion Matrix:")
SVM_confmatrix=metrics.confusion_matrix(b_test, SVM_pred)
SVM_CM_display = metrics.ConfusionMatrixDisplay(confusion_matrix = SVM_confmatrix, display_labels = [False, True])
SVM_CM_display.plot()
plt.show()
from sklearn.metrics import accuracy_score
SVCAcc = accuracy_score(SVM_pred,b_test)
print('SVC accuracy: {:.2f}%'.format(SVCAcc*100))


# In[45]:


fpr1,tpr1,_ = metrics.roc_curve(b_test,SVM_pred)
plt.plot(fpr1,tpr1)
plt.ylabel("True positive")
plt.xlabel("False Positive")
plt.show()


# # 2. Secondary algorithm- Logistic Regression

# In[58]:


LRclassifier = LogisticRegression(solver='saga', max_iter=500)
LRclassifier.fit(A_train, b_train)

LR_pred = LRclassifier.predict(A_test)

print("Classification Report:")
print(classification_report(b_test, LR_pred))
print("Confusion Matrix:")
LR_confmatrix=metrics.confusion_matrix(b_test, LR_pred)
LR_CM_display = metrics.ConfusionMatrixDisplay(confusion_matrix = LR_confmatrix, display_labels = [False, True])
LR_CM_display.plot()
plt.show()
LRAcc = accuracy_score(LR_pred,b_test)
print('LR accuracy: {:.2f}%'.format(LRAcc*100))


# In[47]:


fpr2,tpr2,_ = metrics.roc_curve(b_test,LR_pred)
plt.plot(fpr2,tpr2)
plt.ylabel("True positive")
plt.xlabel("False Positive")
plt.show()


# # Comparing primary and secondary algorithms

# In[56]:


compare = pd.DataFrame({'Model': ['Support Vector Machines','Logistic Regression'], 
                        'Accuracy': [SVCAcc*100,LRAcc*100]})
compare.sort_values(by='Accuracy')


# # Plotting Multiple ROC curve

# In[62]:


plt.figure(0).clf()
auc1 = metrics.roc_auc_score(b_test, SVM_pred)
plt.plot(fpr1,tpr1,label="SVM")
auc2 = metrics.roc_auc_score(b_test, LR_pred)
plt.plot(fpr2,tpr2,label="LR")
plt.legend(loc=0)

