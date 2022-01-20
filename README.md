# heart_disease_prediction
This project is related to a classification problem. The dataset was extracted from https://www.kaggle.com/fedesoriano/heart-failure-prediction as well as all the description of the data.

The objective of the algorithm presented in heart_disease.py and heart_disease_svc_improved is to predict if a person has a heart disease based on several factors:

1. Age
2. Sex
3. Type of chest pain
4. Resting blood pressure
5. Cholesterol
6. Fasting blood sugar
7. Resting ECG
8. Maximum heart rate achieved
9. Exercise angina
10. Oldpeak value
11. Slope of ST segment

The algorithm presented contains the steps of data preprocessing, including encoding of categorical data into numerical values. Binary parameters where encoded with a label encoder, whereas parameters with more than two (but discrete) possible values were encoded with the one-hot-encoding technique. Afterwords, the dataset was used to split the dataset into train/test groups. Five models of classification were used: Decision Trees, Logistic Regression, Naive Bayes, K-Nearest Neighbors (KNN) and Support Vector Classification (SVC). We evaluate the model performance by analysing the Accuracy, Confusion matrix and ROC AUC criteria. Finally, we considered K-fold cross validation to perform multiple tests with each model and average the results of the F1 score and the accuracy. More details regarding the results and discussions will be presented in a separate pdf file.
