import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn import neighbors
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

# Creating classifier objects

classifier_knn_uniform=neighbors.KNeighborsClassifier(n_neighbors=20, weights='uniform')
classifier_knn_distance=neighbors.KNeighborsClassifier(n_neighbors=20, weights='distance')
classifier_svc_rbf=svm.SVC(probability=True, kernel='rbf')
classifier_svc_linear=svm.SVC(probability=True, kernel='linear')
classifier_svc_sigmoid=svm.SVC(probability=True, kernel='sigmoid')

labels=['k-NN uniform', 'k-NN distance', 'SVC Linear', 'SVC RBF', 'SVC Sigmoid']

iflag=0
roc_auc_store=[]

for classifier in [classifier_knn_uniform, classifier_knn_distance, classifier_svc_linear, classifier_svc_rbf, classifier_svc_sigmoid]:

    # Training the model with the training dataset

    classifier.fit(x_train,y_train)

    # Using trained model to test the predictions on the test dataset

    y_pred=classifier.predict(x_test)

    classifier_probs=classifier.fit(x_train,y_train).predict_proba(x_test)
    classifier_roc_auc=roc_auc_score(y_test,classifier_probs[:,1]) # We use column one to choose only the probability values of the positive outcome
    roc_auc_store.append(classifier_roc_auc)
    model_roc_curve_fpr , model_roc_curve_tpr , _ = roc_curve(y_test,classifier_probs[:,1])
    pyplot.plot(model_roc_curve_fpr, model_roc_curve_tpr, '--o',label=labels[iflag])
    iflag+=1

roc_summary={'ROC AUC': roc_auc_store}

roc_summary_df=pd.DataFrame(data=roc_summary, index=labels)

print(roc_summary_df.head(5))

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()