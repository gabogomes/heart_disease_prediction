import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

# Taking out samples for which cholesterol and resting BP value were zero

cholesterol_threshold=0.0        # 85.0
bp_threshold=0.0                 # 80.0

dataset=dataset[dataset["Cholesterol"] > cholesterol_threshold]
dataset=dataset[dataset["RestingBP"] > bp_threshold]

# Transform dataset from pandas dataframe into array for data preprocessing and 
# application of ML algorithms

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

# Changing the order of the columns so that the columns to be One-Hot-Encoded are 
# all to the left (remind that passthrough keeps only columns to the right!)

x[:,0:12]=x[:,[2,6,10,1,5,8,0,3,4,7,9]]

# First three (2,6,10) have more than 2 possible values , other three (1,5,8) are binary and remaining are numerical

# Applying One-Hot-Encoding to the dataset and making sure it is an array

ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2])],remainder='passthrough')
x=np.array(ct.fit_transform(x))

classifier_lr_pure=LogisticRegression(penalty='none',random_state=0,solver='newton-cg')
classifier_lr_l2=LogisticRegression(penalty='l2',random_state=0,solver='newton-cg')
classifier_lr_l1=LogisticRegression(penalty='l1',random_state=0,solver='liblinear')
classifier_decision_trees=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier_naive_bayes=GaussianNB()
classifier_naive_bayes_bernoulli=BernoulliNB()

# Using K-Fold Cross Validation method

cv_number=20

classifier_list=[classifier_lr_pure, classifier_lr_l1, classifier_lr_l2, classifier_decision_trees, classifier_naive_bayes, classifier_naive_bayes_bernoulli]

scores_list=[]

for classifier in classifier_list:

# See more scoring metrics for classification in https://scikit-learn.org/stable/modules/model_evaluation.html

    scores_accuracy_mean=(cross_val_score(classifier, x, y, cv=cv_number, scoring="accuracy")).mean()
    scores_accuracy_std=(cross_val_score(classifier, x, y, cv=cv_number, scoring="accuracy")).std()
    scores_f1_mean=(cross_val_score(classifier, x, y, cv=cv_number, scoring="f1")).mean()
    scores_f1_std=(cross_val_score(classifier, x, y, cv=cv_number, scoring="f1")).std()
    scores_neg_log_loss_mean=(cross_val_score(classifier, x, y, cv=cv_number, scoring="neg_log_loss")).mean()
    scores_neg_log_loss_std=(cross_val_score(classifier, x, y, cv=cv_number, scoring="neg_log_loss")).std()
    scores_roc_auc_mean=(cross_val_score(classifier, x, y, cv=cv_number, scoring="roc_auc")).mean()
    scores_roc_auc_std=(cross_val_score(classifier, x, y, cv=cv_number, scoring="roc_auc")).std()

    scores_list.append(scores_accuracy_mean)
    scores_list.append(scores_accuracy_std)
    scores_list.append(scores_f1_mean)
    scores_list.append(scores_f1_std)
    scores_list.append(scores_neg_log_loss_mean)
    scores_list.append(scores_neg_log_loss_std)
    scores_list.append(scores_roc_auc_mean)
    scores_list.append(scores_roc_auc_std)

models_list=['Pure Logistic Regression', 'L1 Logistic Regression', 'L2 Logistic Regression', 'Decision Tree', 'Gaussian Naive Bayes', 'Bernoulli Naive Bayes']

results_summary={'Model': models_list, 'Mean accuracy': scores_list[0:len(scores_list)-7:8], 'Mean accuracy std dev': scores_list[1:len(scores_list)-6:8],
'Mean F1 score': scores_list[2:len(scores_list)-5:8], 'Mean F1 std dev': scores_list[3:len(scores_list)-4:8], 
'Mean Neg Log Loss score': scores_list[4:len(scores_list)-3:8], 'Mean Neg Log Loss std dev': scores_list[5:len(scores_list)-2:8],
'Mean ROC AUC score': scores_list[6:len(scores_list)-1:8], 'Mean ROC AUC std dev': scores_list[7:len(scores_list):8]}

results_dataset_summary=pd.DataFrame(data=results_summary)#, index=['Pure Linear', 'L1 LR', 'L2 LR', 'Decision Trees', 'Gaussian NB', 'Bernoulli NB'])

print(results_dataset_summary.head(6))

fig, ax =plt.subplots(figsize=(12,4))
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=results_dataset_summary.values,colLabels=results_dataset_summary.columns,loc='center')

pp=PdfPages("../results/model_performance/summary_kfold_methods_results.pdf")
pp.savefig(fig, bbox_inches='tight')
pp.close()

""" Evaluating model performance without cross validation

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score

# Training the model with the training dataset

classifier.fit(x_train,y_train)

# Using trained model to test the predictions on the test dataset

y_pred=classifier.predict(x_test)

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