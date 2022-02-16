import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn import svm
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

# Parameters Sex and Exercise Angina are binary data but are not given in numerical values, differently from
# the case of Fasting BS. We need to use label encoding to transform them

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

# Choosing the classifier

"""

Option 1: Logistic Regression

Penalties are l1, l2, elasticnet or none. Default is l2
Solvers can be chosen according to the penalty added. They are: 
newton-cg - [l2, none]
lbfgs - [l2, none]
liblinear - [l1, l2]
sag - [l2, none]
saga - [elasticnet, l1, l2, none]

classifier=LogisticRegression(penalty='l2',random_state=0,solver='newton-cg')

Option 2: Decision Tree

classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)

Option 3: Naive Bayes

classifier=GaussianNB()

Option 4: Random Forest Ensemble Classifier

Ensemble of decision trees. If bootstrap=True (default), each tree uses a sub-sample of the total sample size, where the
size of the sub-samples is controled with max_samples (default='None', corresponding to X.shape[0] samples). Otherwise, 
if bootstrap=False , the whole dataset is used to build each tree. n_estimators is the number of decision trees. Default 
number is 100. criterion can be gini (default) or entropy.

classifier=RandomForestClassifier(max_samples='None', bootstrap='True', n_estimators=100)

"""

n_neighbors=20
weights='uniform' # uniform or distance

classifier=neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

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