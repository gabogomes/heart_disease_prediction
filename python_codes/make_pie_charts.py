import matplotlib.pyplot as plt
import pandas as pd

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

# Taking out samples for which cholesterol and resting BP value were zero

cholesterol_threshold=0.0        # 85.0
bp_threshold=0.0                 # 80.0

dataset=dataset[dataset["Cholesterol"] > cholesterol_threshold]
dataset=dataset[dataset["RestingBP"] > bp_threshold]

labels_ST_slope=['Up', 'Flat', 'Down']
counter_ST_slope_up=len(dataset[dataset["ST_Slope"] == 'Up'].index)
counter_ST_slope_flat=len(dataset[dataset["ST_Slope"] == 'Flat'].index)
counter_ST_slope_down=len(dataset[dataset["ST_Slope"] == 'Down'].index)
sizes_ST_slope=[counter_ST_slope_up,counter_ST_slope_flat,counter_ST_slope_down]

labels_sex=['Male', 'Female']
counter_sex_male=len(dataset[dataset["Sex"] == 'M'].index)
counter_sex_female=len(dataset[dataset["Sex"] == 'F'].index)
sizes_sex=[counter_sex_male,counter_sex_female]

labels_chest_pain_type=['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymp.']
counter_chest_pain_type_ta=len(dataset[dataset["ChestPainType"] == 'TA'].index)
counter_chest_pain_type_ata=len(dataset[dataset["ChestPainType"] == 'ATA'].index)
counter_chest_pain_type_nap=len(dataset[dataset["ChestPainType"] == 'NAP'].index)
counter_chest_pain_type_asy=len(dataset[dataset["ChestPainType"] == 'ASY'].index)
sizes_chest_pain=[counter_chest_pain_type_ta,counter_chest_pain_type_ata,counter_chest_pain_type_nap,counter_chest_pain_type_asy]

labels_resting_ECG=['Normal', 'ST-T abn.', 'Left Vent. Hyp.']
counter_resting_ECG_normal=len(dataset[dataset["RestingECG"] == 'Normal'].index)
counter_resting_ECG_ST=len(dataset[dataset["RestingECG"] == 'ST'].index)
counter_resting_ECG_LVH=len(dataset[dataset["RestingECG"] == 'LVH'].index)
sizes_ECG=[counter_resting_ECG_normal,counter_resting_ECG_ST,counter_resting_ECG_LVH]


fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

ax0.pie(sizes_ST_slope, labels=labels_ST_slope, shadow=True, startangle=90)
ax0.axis('equal')

ax1.pie(sizes_sex, labels=labels_sex, shadow=True, startangle=90)
ax1.axis('equal')

ax2.pie(sizes_chest_pain, labels=labels_chest_pain_type, shadow=True, startangle=90)
ax2.axis('equal')

ax3.pie(sizes_ECG, labels=labels_resting_ECG, shadow=True, startangle=90)
ax3.axis('equal')

plt.savefig("../results/charts/pie_charts_categorial_variables.pdf")

fig, ax = plt.subplots(1,1)

ax.pie(sizes_ST_slope, labels=labels_ST_slope, autopct='%1.2f%%', shadow=True, startangle=90)
ax.axis('equal')
plt.tight_layout()
plt.savefig("../results/charts/ST_slope_pie_chart.pdf")

fig, ax = plt.subplots(1,1)

ax.pie(sizes_sex, labels=labels_sex, autopct='%1.2f%%', shadow=True, startangle=90)
ax.axis('equal')
plt.tight_layout()
plt.savefig("../results/charts/sex_pie_chart.pdf")

fig, ax = plt.subplots(1,1)

ax.pie(sizes_chest_pain, labels=labels_chest_pain_type, autopct='%1.2f%%', shadow=True, startangle=90)
ax.axis('equal')
plt.tight_layout()
plt.savefig("../results/charts/chest_pain_type_pie_chart.pdf")

fig, ax = plt.subplots(1,1)

ax.pie(sizes_ECG, labels=labels_resting_ECG, autopct='%1.2f%%', shadow=True, startangle=90)
ax.axis('equal')
plt.tight_layout()
plt.savefig("../results/charts/Resting_ECG_pie_chart.pdf")