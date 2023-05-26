#!/usr/bin/env python
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier


# Get the path of the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory to the folder where the script is located
os.chdir(script_dir)

#Load the dataset
df = pd.read_csv('diabetes.csv')

df['isdiab']=df['diabetes'].map({'Diabetes':1, 'No diabetes':0})
df['isdiab'].value_counts()[1]

# print(df.head())

#Removing the unimportatnt features
df.drop(['waist','hip','waist_hip_ratio',],inplace=True, axis=1)


#Exploratory data analysis
# sns.lineplot(y ='glucose',x='weight', hue='diabetes', data =df)
# plt.show()

# sns.lineplot(y ='glucose',x='age', hue='gender', data =df)
# plt.show()

# figure, axis = plt.subplots(1,2,figsize=(8,6))
# sns.lineplot(ax=axis[0],x='systolic_bp', y='glucose',hue='diabetes',data=df)
# sns.lineplot(ax=axis[1],x='diastolic_bp', y='glucose',hue='diabetes',data=df)
# plt.show()

# sns.lineplot(x=df.hdl_chol,y= df.glucose,hue=df.diabetes,data=df)
# plt.show()

# sns.countplot(x='diabetes',hue='gender',data=df)
# plt.show()

# sns.lineplot(x=df.cholesterol,y= df.glucose,hue=df.diabetes,data=df)
# plt.show()
#In general, It is seen that higher weight and old Age are two major factor causing diabetes.
#What do we conclude from these graphs?
#1- Diabetic patients have Higher glucose rate and higher weight as comapred to Non -diabetic ones
#2- The Age is not directly realted but higher gluose level in oldies can be a cause of Diabetes in them, also the males of age 40 to 80 have higher blood glucose level than females.
#3- The BP is not directly related to the diabetes, as patients have highest BP are Found to be Non-diabetic.
#4- Diabetic patients have lower HDL-Cholesterol
#5- Females being diabetic are more than the males being diabetic.
#6- Higher cholestrol is seen in the patients having diabetes



#Removing the unimortant features
df1=df[['patient_number','cholesterol','glucose','hdl_chol','age','gender','weight','systolic_bp','diastolic_bp','isdiab']]


#Splitting the dataset into X and Y
X= df1[['cholesterol','glucose','hdl_chol','age','weight','systolic_bp','diastolic_bp']]
y=df['isdiab']


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.4,random_state=1)


'''
The Highest K score is 5 with accuraccy of 91.67%
score = []
for i in range(1,20) :
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    score.append(accuracy)
    print('KNN accuracy with {0} neighbors is : {1}'.format(i, accuracy))

plt.figure()
plt.plot(range(1,20), score)
plt.show()
'''
#KNN Model
kNN = KNeighborsClassifier(n_neighbors=5)
kNN.fit(X_train, y_train)
kNN_pred=kNN.predict(X_test)


#SVM Model
svm = svm.SVC(kernel='linear')
svm.fit(X_train, y_train)
predict = svm.predict(X_test)
# accuraccy 92.94%
# print('svm accuracy : ', accuracy_score(y_test, predict))


'''
Neural Network gives accuraccy of 83.3%, so it won't be used 

nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
nn.fit(X_train, y_train)
nnpredict = nn.predict(X_test)
print('nn accuracy : ', accuracy_score(y_test, nnpredict))
'''



def predict_diabetes_KNN(cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp):
    user_inputs = np.array([cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp]).reshape(1, -1)
    user_df = pd.DataFrame(user_inputs, columns=['cholesterol', 'glucose', 'hdl_chol', 'age', 'weight', 'systolic_bp', 'diastolic_bp'])
    
    prediction = kNN.predict(user_df)
    
    return prediction[0]

def predict_diabetes_SVM(cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp):
    user_inputs = np.array([cholesterol, glucose, hdl_chol, age, weight, systolic_bp, diastolic_bp]).reshape(1, -1)
    user_df = pd.DataFrame(user_inputs, columns=['cholesterol', 'glucose', 'hdl_chol', 'age', 'weight', 'systolic_bp', 'diastolic_bp'])
    
    prediction = svm.predict(user_df)
    
    return prediction[0]

# print(predict_diabetes_KNN(203, 299, 43, 38, 288, 136, 83))