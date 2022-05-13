# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:59:23 2022

Several constraints were placed on the selection of these instances from a larger database. 
In particular, all patients here are females at least 21 years old of Pima Indian heritage.

-Pregnancies: Number of times pregnant
-Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
-BloodPressure: Diastolic blood pressure (mm Hg)
-SkinThickness: Triceps skin fold thickness (mm)
-Insulin: 2-Hour serum insulin (mu U/ml)
-BMI: Body mass index (weight in kg/(height in m)^2)
-DiabetesPedigreeFunction: Diabetes pedigree function
-Age: Age (years)
-Outcome: Class variable (0 or 1)

Number of Instances: 768
Number of Attributes: 8 plus class
For Each Attribute: (all numeric-valued)

1)Number of times pregnant
2)Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3)Diastolic blood pressure (mm Hg)
4)Triceps skin fold thickness (mm)
5)2-Hour serum insulin (mu U/ml)
6)Body mass index (weight in kg/(height in m)^2)
7)Diabetes pedigree function
8)Age (years)
9)Class variable (0 or 1)

# need to save minmax/standard scaler and import it in deployment(using pickle, .pkl)
# also similar to hot encoding/label encoding

#NOTE: Model saving - machine learning (.pkl)
                    - deep learning (.h5)
@author: aceso
"""
#%% Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import missingno as msno
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from sklearn.exceptions import DataConversionWarning
import warnings

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Constant
DATA_PATH = os.path.join(os.getcwd(), "diabetes.csv")
SCALER_SAVE_PATH = os.path.join(os.getcwd(), "static", "mm_scaler.pkl")
MODEL_PATH = os.path.join(os.getcwd(), "static", "model.pkl")

# random forest params
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

#%% EDA

# Data Loading
df = pd.read_csv(DATA_PATH, header = [0],)
print(df.head())

# Data Inspection
df.info()
df.describe().T

plt.boxplot(df, widths = 0.6, patch_artist = True)
df.isnull().sum() # 3 glucose, 1 bloodpressure, 2 skinthickness, 1 bmi
msno.matrix(df) # missing value visualization

# Data Cleaning
imputer = SimpleImputer(strategy="mean")
np_cleaned = df.copy()
np_cleaned = imputer.fit_transform(np_cleaned)

# Feature Selection
pd_cleaned = pd.DataFrame(np_cleaned)
cor = pd_cleaned.corr() 
plt.figure(figsize=(20,5)) # Plot a figure
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show();
# Since all features are not strongly correlated to the labels so we not going
# to di feature selection

#Feature scalling

X = np_cleaned[:,:-1] # features
y = np.expand_dims(np_cleaned[:,-1], -1) #labels

plt.figure()
sns.displot(X)
plt.title("Before Scalling")

mm_scaler = MinMaxScaler()
X = mm_scaler.fit_transform(X)
plt.figure()
sns.displot(X)
plt.title("MinMax Scalling") # MinMax look more promising
# Save scaler
pickle.dump(mm_scaler, open(SCALER_SAVE_PATH, "wb"))

np.any(np.isnan(X)) # no more isnan
np.all(np.isfinite(X)) # no infinite data

# Data Processing 

# Data splitting into test and train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=420)

#%% Model Building

step_tr = [("TR", DecisionTreeClassifier())]
step_fr = [("FR", RandomForestClassifier())]
step_lg = [("LG", LogisticRegression())]
step_svm = [("SVM", SVC())]
step_knn = [("KNN", KNeighborsClassifier())]

pipeline_tr = Pipeline(step_tr)
pipeline_fr = Pipeline(step_fr)
pipeline_lg = Pipeline(step_lg)
pipeline_svm = Pipeline(step_svm)
pipeline_knn = Pipeline(step_knn)

pipelines = [pipeline_tr, pipeline_fr, pipeline_lg, pipeline_svm, pipeline_knn]


for pipe in pipelines:
    pipe.fit(X_train, y_train)
    
pipe_dict = {0: "Decision Tree", 
             1: "Random Forest", 
             2: "Logistic Regression", 
             3: "SVM", 
             4: "KNN"}

for i, model in enumerate(pipelines):
    print(f"{pipe_dict[i]}: {model.score(X_test, y_test)}")
    
# The best model so far is RandomForestClassifier

#%% Random Search Cross Validation

step_fr = [("FR", RandomForestClassifier())]
pipeline_fr.get_params().keys() # see the keys/params

random_params = {'FR__n_estimators': n_estimators,
               'FR__max_features': max_features,
               'FR__max_depth': max_depth,
               'FR__min_samples_split': min_samples_split,
               'FR__min_samples_leaf': min_samples_leaf,
               'FR__bootstrap': bootstrap} 

random_search = RandomizedSearchCV(estimator=Pipeline(step_fr), 
                                   param_distributions=random_params,
                                   cv=3,
                                   n_iter=100,
                                   random_state=420,
                                   verbose=2,
                                   n_jobs=-1) 

best_combination = random_search.fit(X_train, y_train)
best_combination.best_params_


#%% Evaluation
y_pred = best_combination.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%% Model Saving

pickle.dump(best_combination, open(MODEL_PATH, "wb"))



