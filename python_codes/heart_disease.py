import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
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

# Parameters Sex and Exercise Angina are binary data but are not given in numerical values, as 
# it is the case of Fasting BS. We need to use label encoding to transform them

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

#scaler=StandardScaler()
#x_train[:,[13,14,15,16,17]]=scaler.fit_transform(x_train[:,[13,14,15,16,17]])
#x_test[:,[13,14,15,16,17]]=scaler.fit_transform(x_test[:,[13,14,15,16,17]])

# Creating classifier objects

classifier_lr_pure=LogisticRegression(penalty='none',random_state=0,solver='newton-cg')
classifier_lr_l2=LogisticRegression(penalty='l2',random_state=0,solver='newton-cg')
classifier_lr_l1=LogisticRegression(penalty='l1',random_state=0,solver='liblinear')
classifier_decision_trees=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier_naive_bayes=GaussianNB()
classifier_naive_bayes_bernoulli=BernoulliNB()

labels=['Logistic Regression', 'Logistic Regression L2', 'Logistic Regression L1', 'Decision Tree', 'Gaussian NB', 'Bernoulli NB']

iflag=0
roc_auc_store=[]
accuracy_store=[]
f1_store=[]
precision_store=[]
recall_store=[]

for classifier in [classifier_lr_pure, classifier_lr_l2, classifier_lr_l1, classifier_decision_trees, classifier_naive_bayes, classifier_naive_bayes_bernoulli]:

    # Training the model with the training dataset

    classifier.fit(x_train,y_train)

    # Using trained model to test the predictions on the test dataset

    y_pred=classifier.predict(x_test)

    classifier_probs=classifier.fit(x_train,y_train).predict_proba(x_test)
    classifier_roc_auc=roc_auc_score(y_test,classifier_probs[:,1]) # We use column one to choose only the probability values of the positive outcome
    roc_auc_store.append(classifier_roc_auc)
    model_roc_curve_fpr , model_roc_curve_tpr , _ = roc_curve(y_test,classifier_probs[:,1])
    accuracy_store.append(accuracy_score(y_test, y_pred))
    f1_store.append(f1_score(y_test, y_pred))
    precision_store.append(precision_score(y_test, y_pred)) # average = ???micro???, ???macro???, ???samples???, ???weighted???, ???binary???} or None, default=???binary???
    recall_store.append(recall_score(y_test, y_pred)) # average = ???micro???, ???macro???, ???samples???, ???weighted???, ???binary???} or None, default=???binary???
    pyplot.plot(model_roc_curve_fpr, model_roc_curve_tpr, '--o',label=labels[iflag])
    iflag+=1

roc_summary={'Model': labels, 'ROC AUC': roc_auc_store, 'Accuracy': accuracy_store, 'F1 score': f1_store, 'Precision': precision_store, 'Recall': recall_store}

roc_summary_df=pd.DataFrame(data=roc_summary)

print(roc_summary_df.head(6))

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.tight_layout()
pyplot.savefig("../results/model_performance/comparison_nonscaled_methods.pdf")

fig, ax =plt.subplots(figsize=(12,4))
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=roc_summary_df.values,colLabels=roc_summary_df.columns,loc='center')

pp=PdfPages("../results/model_performance/summary_nonscaled_classifiers_roc.pdf")
pp.savefig(fig, bbox_inches='tight')
pp.close()