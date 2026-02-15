# ML

Breast Cancer Classification Project

a. 
Problem Statement 
The objective of this project is to build and deploy a machine learning application that classifies tumors as Malignant or Benign based on clinical diagnostic features. This project demonstrates an end-to-end workflow from model training to cloud deployment.

b. 
Dataset Description 
•	Source: UCI Machine Learning Repository / Scikit-learn Breast Cancer Wisconsin (Diagnostic) Dataset. 
•	Instance Size: 569 instances. 
•	Feature Size: 30 numeric features (e.g., radius, texture, perimeter). 
•	Target Variable: Binary (0: Malignant, 1: Benign). 

c. 
Models Used and Comparison Table 
ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.9737	0.9974	0.9722	0.9859	0.9790	0.9439
Decision Tree	0.9386	0.9369	0.9571	0.9437	0.9504	0.8701
kNN	0.9474	0.9820	0.9577	0.9577	0.9577	0.8880
Naive Bayes	0.9649	0.9974	0.9589	0.9859	0.9722	0.9253
Random Forest	0.9649	0.9941	0.9589	0.9859	0.9722	0.9253
XGBoost	0.9561	0.9908	0.9583	0.9718	0.9650	0.9064
d. 
ML Model Name,Observation about model performance
Logistic Regression,"Achieved the highest accuracy and MCC, indicating it handles the linear relationships in this dataset best."
Decision Tree,Lowest performance likely due to splitting on features that don't generalize as well as ensemble methods.
kNN,"Strong performance, but sensitive to the scaling of the features; effectively captured local patterns."
Naive Bayes,Tied for high recall; it is highly efficient and performed exceptionally well despite feature correlations.
Random Forest,"Excellent balance of precision and recall, proving the effectiveness of bagging in reducing variance."
XGBoost,Very high AUC score; provided a stable and reliable classification using gradient boosting.
