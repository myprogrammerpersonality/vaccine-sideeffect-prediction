import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, train_test_split

import time
import sys
import warnings
warnings.simplefilter("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import logging

logging.basicConfig(filename='T7.log', format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logging.info('Started')

random_state = 1234


print('*** Import is Done!')
vaccine_type=sys.argv[1]
input_data_path = sys.argv[2]
output_path = sys.argv[3]

# Load data
data = pd.read_csv(input_data_path)

# Preprocess data
data = data[data.Age <= 120]
data = data[data.Age >= 18]

data = data[data.Height <= 200]
data = data[data.Height >= 100]

data = data[data.Weight <= 200]
data = data[data.Weight >= 30]

data['Dose1_Local'] = data[['Dose1_Needle_Redness', 'Dose1_Needle_Swelling', 'Dose1_Needle_Pain']].apply(lambda x: max(x), axis=1)

dose_1_columns_full = ['Dose1_Fever', 'Dose1_Needle_Pain', 'Dose1_Fatigue', 'Dose1_Headache',
                  'Dose1_Nausea', 'Dose1_Chills', 'Dose1_Needle_Redness', 'Dose1_Joint_Pain',
                  'Dose1_Muscle_Pain', 'Dose1_Other', 'Dose1_Needle_Swelling', 'Dose1_Local']

dose_1_columns = ['Dose1_Fever', 'Dose1_Fatigue', 'Dose1_Headache', 'Dose1_Nausea',
                  'Dose1_Chills', 'Dose1_Joint_Pain', 'Dose1_Muscle_Pain',
                  'Dose1_Local']


data = data.drop(['Height', 'Weight'], axis=1)

logging.info('Data loaded and processed')

# HPT-CV
## LR
logging.info('LR started')

results_df = pd.DataFrame(columns=['Symptom', 'Baseline_AUC_Mean', 'Baseline_AUC_Std', 'Baseline_Accuracy_Mean', 'Baseline_Accuracy_Std', 'Best_Param',
                                   'Best_Train_AUC_Mean', 'Best_Train_AUC_Std', 'Best_Val_AUC_Mean', 'Best_Val_AUC_Std', 'Best_Test_AUC',
                                   'Best_Train_Accuracy_Mean', 'Best_Train_Accuracy_Std', 'Best_Val_Accuracy_Mean', 'Best_Val_Accuracy_Std', 'Best_Test_Accuracy',
                                   'Best_Train_Precision_Mean', 'Best_Train_Precision_Std', 'Best_Val_Precision_Mean', 'Best_Val_Precision_Std', 'Best_Test_Precision',
                                   'Best_Train_Recall_Mean', 'Best_Train_Recall_Std', 'Best_Val_Recall_Mean', 'Best_Val_Recall_Std', 'Best_Test_Recall'])

for index, target in enumerate(dose_1_columns):
    print('***',target,'***')
    
    X = data.drop(dose_1_columns_full, axis=1)
    y = data[target]
    
    X, X_test, y, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=random_state)
    cross_val = StratifiedKFold(n_splits=5)
    
    model = LogisticRegression()
    param_grid = {'solver': ['lbfgs', 'liblinear'],
                  'C': [0.01, 0.1, 0.5, 1.0, 1.5, 10.0]}

    grid = GridSearchCV(model, param_grid, scoring=['roc_auc', 'accuracy', 'precision', 'recall'], n_jobs=-1,
                        refit='roc_auc', cv=cross_val, verbose=1, return_train_score=True)

    grid.fit(X, y)
    
    results = pd.DataFrame(grid.cv_results_)
    best_result = results[results.rank_test_roc_auc == 1]
    
    
    # Test Set
    y_pred_prob = grid.best_estimator_.predict_proba(X_test)[:,1]
    y_pred = grid.best_estimator_.predict(X_test)
    
    auc_test = roc_auc_score(y_test, y_pred_prob)
    accuracy_test = accuracy_score(y_test, y_pred)
    print(auc_test, accuracy_test)
    # Save model
    joblib.dump(grid.best_estimator_, f'Outputs/Models/{vaccine_type}/D1/{vaccine_type}_LR_{target}_Aug8.model')
