import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt

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

number_of_bins=50

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

ax0.hist(x[:,[6]], number_of_bins, histtype='bar', label='Age')
ax0.set_ylabel("Number of people")
ax0.set_xlabel("Age")

ax1.hist(x[:,[7]], number_of_bins, histtype='bar', label='Resting Blood Pressure')
ax1.set_ylabel("Number of people")
ax1.set_xlabel("Resting BP")

ax2.hist(x[:,[8]], number_of_bins, histtype='bar', label='Cholesterol')
ax2.set_ylabel("Number of people")
ax2.set_xlabel("Cholesterol")

ax3.hist(x[:,[9]], number_of_bins, histtype='bar', label='Maximum Heart Rate')
ax3.set_ylabel("Number of people")
ax3.set_xlabel("Maximum Heart Rate")

fig.tight_layout()
plt.show()
