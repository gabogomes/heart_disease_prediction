import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn import svm
from matplotlib import pyplot

dataset=pd.read_csv("../dataset/heart_disease_data.csv")

"""

Age: age of the patient [years]
Sex: sex of the patient [M: Male, F: Female]
ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
RestingBP: resting blood pressure [mm Hg]
Cholesterol: serum cholesterol [mm/dl]
FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
Oldpeak: oldpeak = ST [Numeric value measured in depression]
ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
HeartDisease: output class [1: heart disease, 0: Normal]

Age , Sex , Chest Pain Type , Resting BP , Cholesterol , Fasting BS , Resting ECG , Max HR , Exercise Angina , Oldpeak , ST_Slope , Heart Disease

"""

# Parameters Sex and Exercise Angina are binary data but are not given in numerical values, as 
# it is the case of Fasting BS. We need to use label encoding to transform them

dataset["Sex"]=LabelEncoder().fit_transform(dataset["Sex"])
dataset["ExerciseAngina"]=LabelEncoder().fit_transform(dataset["ExerciseAngina"])

# Transform dataset from pandas dataframe into array for data preprocessing and 
# application of ML algorithms

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

# Changing the order of the columns so that the columns to be One-Hot-Encoded are 
# all to the left (remind that passthrough keeps only columns to the right!)

x[:,0:12]=x[:,[2,6,10,1,5,8,0,3,4,7,9]]

"""

First three (2,6,10) have more than 2 possible values , other three (1,5,8) are binary and remaining are numerical

"""

# Applying One-Hot-Encoding to the dataset and making sure it is an array

ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2])],remainder='passthrough')
x=np.array(ct.fit_transform(x))

# Train-Test split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# Performing variable standardization or normalization to improve results. 
# Needs to be done after train-test-split so that test set results do not impact
# the training set results and cause bias

scaler=StandardScaler()
x_train[:,[13,14,15,16,17]]=scaler.fit_transform(x_train[:,[13,14,15,16,17]])
x_test[:,[13,14,15,16,17]]=scaler.fit_transform(x_test[:,[13,14,15,16,17]])

# Creating an object of the Support Vector Classifier

#Several kernel options, such as:
#linear
#poly
#rbf (standard if none is given)
#sigmoid
#precomputed

classifier=svm.SVC(probability=True, kernel='rbf')

# Training the model with the training dataset

classifier.fit(x_train,y_train)

# Using trained model to test the predictions on the test dataset

y_pred=classifier.predict(x_test)

# Evaluating model performance

"""

Confusion matrix is composed by:

 [True positives   False positives]
 [False negatives   True negatives]

 Accuracy = (True positives + True negatives) / (True positives + True negatives + False positives + False negatives)

 ROC AUC = Receiver Operating Characteristics Area Under Curve. We also plot the ROC Curve

"""

cm=confusion_matrix(y_test,y_pred)
model_accuracy=accuracy_score(y_test,y_pred)
classifier_probs=classifier.fit(x_train,y_train).predict_proba(x_test)
classifier_roc_auc=roc_auc_score(y_test,classifier_probs[:,1]) # We use column one to choose only the probability values of the positive outcome
model_roc_curve_fpr , model_roc_curve_tpr , _ = roc_curve(y_test,classifier_probs[:,1])

print("Confusion matrix:")
print(cm)
print("Accuracy:")
print(model_accuracy)
print("ROC AUC:")
print(classifier_roc_auc)

pyplot.plot(model_roc_curve_fpr, model_roc_curve_tpr, '--o')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.show()

# The line below (when uncommented) shows two matrices: the predicted values by the model and the true values of the dataset, concatenated

# print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

"""

Up to now, we considered one single training process with 80% of the total dataset. We will now
consider a technique to refine our analysis. It is called K-Fold Cross Validation. Basically, it
consists in separating the dataset in K groups. The data contained in the training set change, as
well as the data contained in the test set. This can be done with the model selection method of the
SKLearn package, as below. We will use both the accuracy and the F1 score to evaluate model performance.

"""

cv_number=20

scores_accuracy_mean=(cross_val_score(classifier, x, y, cv=cv_number, scoring="accuracy")).mean()
scores_accuracy_std=(cross_val_score(classifier, x, y, cv=cv_number, scoring="accuracy")).std()


scores_f1_mean=(cross_val_score(classifier, x, y, cv=cv_number, scoring="f1")).mean()
scores_f1_std=(cross_val_score(classifier, x, y, cv=cv_number, scoring="f1")).std()

print("Accuracy mean and standard deviation of K-Fold Cross Validation procedure:")
print(scores_accuracy_mean,scores_accuracy_std)
print("F1 mean and standard deviation of K-Fold Cross Validation procedure:")
print(scores_f1_mean,scores_f1_std)