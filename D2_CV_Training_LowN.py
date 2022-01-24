import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.font_manager as fm

mpl.rcParams['font.family']  = 'Avenir LT Std'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, train_test_split

import time
import sys
import warnings
warnings.simplefilter("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import logging

logging.basicConfig(filename='logs/T3.log', format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logging.info('Started')

random_state = 1234


print('*** Import is Done!')
vaccine_type=sys.argv[1]
input_data_path = sys.argv[2]
output_path = sys.argv[3]
name_dict = {'SputnikV': 'Sputnik V',
             'AstraZeneca': 'AZD1222',
             'Sinopharm': 'BBIBP-CorV',
             'Covaxin': 'COVAXIN',
             'Barekat': 'COVIran Barekat',
             'Pfizer': 'BNT162b2',
             'Moderna': 'mRNA-1273',
             'MRNA': 'MRNA'}


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
data['Dose_Last_Local'] = data[['Dose_Last_Needle_Redness', 'Dose_Last_Needle_Swelling', 'Dose_Last_Needle_Pain']].apply(lambda x: max(x), axis=1)

dose_1_columns_full = ['Dose1_Fever', 'Dose1_Needle_Pain', 'Dose1_Fatigue', 'Dose1_Headache',
                  'Dose1_Nausea', 'Dose1_Chills', 'Dose1_Needle_Redness', 'Dose1_Joint_Pain',
                  'Dose1_Muscle_Pain', 'Dose1_Other', 'Dose1_Needle_Swelling', 'Dose1_Local']

dose_1_columns = ['Dose1_Fever', 'Dose1_Fatigue', 'Dose1_Headache', 'Dose1_Nausea',
                  'Dose1_Chills', 'Dose1_Joint_Pain', 'Dose1_Muscle_Pain',
                  'Dose1_Local']

dose_last_columns_full = ['Dose_Last_Fever', 'Dose_Last_Needle_Pain', 'Dose_Last_Fatigue', 'Dose_Last_Headache',
                     'Dose_Last_Nausea', 'Dose_Last_Chills', 'Dose_Last_Needle_Redness', 'Dose_Last_Joint_Pain',
                     'Dose_Last_Muscle_Pain', 'Dose_Last_Other', 'Dose_Last_Needle_Swelling', 'Dose_Last_Local']

dose_last_columns = ['Dose_Last_Fever', 'Dose_Last_Fatigue', 'Dose_Last_Headache', 'Dose_Last_Nausea',
                  'Dose_Last_Chills', 'Dose_Last_Joint_Pain', 'Dose_Last_Muscle_Pain',
                  'Dose_Last_Local']

data[dose_last_columns_full].describe().to_csv(f'{output_path}/reports/D2_{vaccine_type}_Report_{time.strftime("%Y-%m-%d")}.csv')
data.drop(dose_last_columns_full, axis=1).describe().to_csv(f'{output_path}/reports/D2_{vaccine_type}_Report_Features_{time.strftime("%Y-%m-%d")}.csv')

# Normalization
sc = MinMaxScaler()

data['Age'] = sc.fit_transform(data[['Age']].values)
data['Height'] = sc.fit_transform(data[['Height']].values)
data['Weight'] = sc.fit_transform(data[['Weight']].values)
data['BMI'] = sc.fit_transform(data[['BMI']].values)


data = data.drop(['Height', 'Weight'], axis=1)

data = data.drop(['Dose1_Needle_Pain', 'Dose1_Needle_Redness', 'Dose1_Needle_Swelling', 'Dose1_Other'], axis=1)

logging.info('Data loaded and processed')

# HPT-CV
## LR
logging.info('LR started')

results_df = pd.DataFrame(columns=['Symptom', 'Baseline_AUC_Mean', 'Baseline_AUC_Std', 'Baseline_Accuracy_Mean', 'Baseline_Accuracy_Std', 'Best_Param',
                                   'Best_Train_AUC_Mean', 'Best_Train_AUC_Std', 'Best_Val_AUC_Mean', 'Best_Val_AUC_Std',
                                   'Best_Train_Accuracy_Mean', 'Best_Train_Accuracy_Std', 'Best_Val_Accuracy_Mean', 'Best_Val_Accuracy_Std', 
                                   'Best_Train_Precision_Mean', 'Best_Train_Precision_Std', 'Best_Val_Precision_Mean', 'Best_Val_Precision_Std',
                                   'Best_Train_Recall_Mean', 'Best_Train_Recall_Std', 'Best_Val_Recall_Mean', 'Best_Val_Recall_Std'])

for index, target in enumerate(dose_last_columns):
    print('***',target,'***')
    
    X = data.drop(dose_last_columns_full, axis=1)
    y = data[target]
    
    cross_val = StratifiedKFold(n_splits=3)
    
    baseline_model = LogisticRegression()
    auc = cross_val_score(baseline_model, X, y, scoring='roc_auc', cv=cross_val, n_jobs=-1)
    auc_mean = auc.mean()
    auc_std = auc.std()

    baseline_model = LogisticRegression()
    acc = cross_val_score(baseline_model, X, y, scoring='accuracy', cv=cross_val, n_jobs=-1)
    acc_mean = acc.mean()
    acc_std = acc.std()
    
    model = LogisticRegression()
    param_grid = {'solver': ['lbfgs', 'liblinear'],
                  'C': [0.01, 0.1, 0.5, 1.0, 1.5, 10.0]}

    grid = GridSearchCV(model, param_grid, scoring=['roc_auc', 'accuracy', 'precision', 'recall'], n_jobs=-1,
                        refit='roc_auc', cv=cross_val, verbose=0, return_train_score=True)

    grid.fit(X, y)
    
    results = pd.DataFrame(grid.cv_results_)
    best_result = results[results.rank_test_roc_auc == 1].head(1)
    
    
    results_df.loc[index] = [target, auc_mean, auc_std, acc_mean, acc_std, grid.best_params_,
                            best_result.mean_train_roc_auc.item(), best_result.std_train_roc_auc.item(), best_result.mean_test_roc_auc.item(),  best_result.std_test_roc_auc.item(),
                            best_result.mean_train_accuracy.item(), best_result.std_train_accuracy.item(), best_result.mean_test_accuracy.item(),  best_result.std_test_accuracy.item(),
                            best_result.mean_train_precision.item(), best_result.std_train_precision.item(), best_result.mean_test_precision.item(),  best_result.std_test_precision.item(),
                            best_result.mean_train_recall.item(), best_result.std_train_recall.item(), best_result.mean_test_recall.item(),  best_result.std_test_recall.item(),
                            ]

results_df.to_csv(f'{output_path}/results/D2_Logistic_Regression_{vaccine_type}_CVTest_{time.strftime("%Y-%m-%d")}.csv', index=False)
plt.style.use('fivethirtyeight')

fig, ax1 = plt.subplots(figsize=(15, 10))


temp = results_df[['Symptom', 'Best_Val_AUC_Mean', 'Best_Val_Accuracy_Mean', 'Best_Val_Precision_Mean', 'Best_Val_Recall_Mean']].melt(id_vars='Symptom').rename(columns=str.title)

sns.barplot(data=temp, x='Symptom', y='Value', hue='Variable', ax=ax1)

plt.xticks(rotation=45)
lg = plt.legend()
lg.get_texts()[0].set_text('AUC-ROC')
lg.get_texts()[1].set_text('Accuracy')
lg.get_texts()[2].set_text('Precision')
lg.get_texts()[3].set_text('Recall')

plt.ylabel('Performance on 5-Fold Cross Validation')
plt.title(f'Dose 2 Logistic Regression {name_dict[vaccine_type]}')

fig.savefig(f'{output_path}/image/D2_Logistic_Regression_{vaccine_type}_CVTest_{time.strftime("%Y-%m-%d")}.png', bbox_inches='tight')

logging.info('LR finished')

## SVM
logging.info('SVM started')

from sklearn.svm import SVC

results_df = pd.DataFrame(columns=['Symptom', 'Baseline_AUC_Mean', 'Baseline_AUC_Std', 'Baseline_Accuracy_Mean', 'Baseline_Accuracy_Std', 'Best_Param',
                                   'Best_Train_AUC_Mean', 'Best_Train_AUC_Std', 'Best_Val_AUC_Mean', 'Best_Val_AUC_Std',
                                   'Best_Train_Accuracy_Mean', 'Best_Train_Accuracy_Std', 'Best_Val_Accuracy_Mean', 'Best_Val_Accuracy_Std', 
                                   'Best_Train_Precision_Mean', 'Best_Train_Precision_Std', 'Best_Val_Precision_Mean', 'Best_Val_Precision_Std',
                                   'Best_Train_Recall_Mean', 'Best_Train_Recall_Std', 'Best_Val_Recall_Mean', 'Best_Val_Recall_Std'])

for index, target in enumerate(dose_last_columns):
    print('***',target,'***')
    
    X = data.drop(dose_last_columns_full, axis=1)
    y = data[target]
    
    cross_val = StratifiedKFold(n_splits=3)
    
    baseline_model = SVC()
    auc = cross_val_score(baseline_model, X, y, scoring='roc_auc', cv=cross_val, n_jobs=-1)
    auc_mean = auc.mean()
    auc_std = auc.std()

    baseline_model = SVC()
    acc = cross_val_score(baseline_model, X, y, scoring='accuracy', cv=cross_val, n_jobs=-1)
    acc_mean = acc.mean()
    acc_std = acc.std()
    
    model = SVC(probability=True)
    param_grid = {'C': [0.1, 1, 10], 
                  'gamma': ['scale'],
                  'kernel': ['linear']}

    grid = GridSearchCV(model, param_grid, scoring=['roc_auc', 'accuracy', 'precision', 'recall'], n_jobs=1,
                        refit='roc_auc', cv=cross_val, verbose=2, return_train_score=True)

    grid.fit(X, y)
    
    results = pd.DataFrame(grid.cv_results_)
    best_result = results[results.rank_test_roc_auc == 1].head(1)
    
    
    results_df.loc[index] = [target, auc_mean, auc_std, acc_mean, acc_std, grid.best_params_,
                            best_result.mean_train_roc_auc.item(), best_result.std_train_roc_auc.item(), best_result.mean_test_roc_auc.item(),  best_result.std_test_roc_auc.item(),
                            best_result.mean_train_accuracy.item(), best_result.std_train_accuracy.item(), best_result.mean_test_accuracy.item(),  best_result.std_test_accuracy.item(),
                            best_result.mean_train_precision.item(), best_result.std_train_precision.item(), best_result.mean_test_precision.item(),  best_result.std_test_precision.item(),
                            best_result.mean_train_recall.item(), best_result.std_train_recall.item(), best_result.mean_test_recall.item(),  best_result.std_test_recall.item(),
                            ]

results_df.to_csv(f'{output_path}/results/D2_SVM_{vaccine_type}_CVTest_{time.strftime("%Y-%m-%d")}.csv', index=False)
plt.style.use('fivethirtyeight')

fig, ax1 = plt.subplots(figsize=(15, 10))


temp = results_df[['Symptom', 'Best_Val_AUC_Mean', 'Best_Val_Accuracy_Mean', 'Best_Val_Precision_Mean', 'Best_Val_Recall_Mean']].melt(id_vars='Symptom').rename(columns=str.title)

sns.barplot(data=temp, x='Symptom', y='Value', hue='Variable', ax=ax1)

plt.xticks(rotation=45)
lg = plt.legend()
lg.get_texts()[0].set_text('AUC-ROC')
lg.get_texts()[1].set_text('Accuracy')
lg.get_texts()[2].set_text('Precision')
lg.get_texts()[3].set_text('Recall')

plt.ylabel('Performance on 5-Fold Cross Validation')
plt.title(f'Dose 2 SVM {name_dict[vaccine_type]}')

fig.savefig(f'{output_path}/image/D2_SVM_{vaccine_type}_CVTest_{time.strftime("%Y-%m-%d")}.png', bbox_inches='tight')

logging.info('SVM finished')

## XGBoost
logging.info('XGBoost Started')

from xgboost import XGBClassifier

results_df = pd.DataFrame(columns=['Symptom', 'Baseline_AUC_Mean', 'Baseline_AUC_Std', 'Baseline_Accuracy_Mean', 'Baseline_Accuracy_Std', 'Best_Param',
                                   'Best_Train_AUC_Mean', 'Best_Train_AUC_Std', 'Best_Val_AUC_Mean', 'Best_Val_AUC_Std',
                                   'Best_Train_Accuracy_Mean', 'Best_Train_Accuracy_Std', 'Best_Val_Accuracy_Mean', 'Best_Val_Accuracy_Std', 
                                   'Best_Train_Precision_Mean', 'Best_Train_Precision_Std', 'Best_Val_Precision_Mean', 'Best_Val_Precision_Std',
                                   'Best_Train_Recall_Mean', 'Best_Train_Recall_Std', 'Best_Val_Recall_Mean', 'Best_Val_Recall_Std'])

for index, target in enumerate(dose_last_columns):
    print('***',target,'***')
    
    X = data.drop(dose_last_columns_full, axis=1)
    y = data[target]
    
    cross_val = StratifiedKFold(n_splits=3)
    
    baseline_model = XGBClassifier()
    auc = cross_val_score(baseline_model, X, y, scoring='roc_auc', cv=cross_val, n_jobs=-1)
    auc_mean = auc.mean()
    auc_std = auc.std()

    baseline_model = XGBClassifier()
    acc = cross_val_score(baseline_model, X, y, scoring='accuracy', cv=cross_val, n_jobs=-1)
    acc_mean = acc.mean()
    acc_std = acc.std()
    
    model = XGBClassifier()
    param_grid = {'n_estimators': [50, 100, 150],
                  'learning_rate': [0.01, 0.1, 1.0, 10.0],
                  'max_depth':[6,8,9,10],
                  'min_child_weight':[1,2,3,4]}

    grid = RandomizedSearchCV(model, param_grid, scoring=['roc_auc', 'accuracy', 'precision', 'recall'],
                              n_jobs=-1, refit='roc_auc', cv=cross_val, verbose=0, return_train_score=True, n_iter=50)

    grid.fit(X, y)
    
    results = pd.DataFrame(grid.cv_results_)
    best_result = results[results.rank_test_roc_auc == 1].head(1)
    
    results_df.loc[index] = [target, auc_mean, auc_std, acc_mean, acc_std, grid.best_params_,
                            best_result.mean_train_roc_auc.item(), best_result.std_train_roc_auc.item(), best_result.mean_test_roc_auc.item(),  best_result.std_test_roc_auc.item(),
                            best_result.mean_train_accuracy.item(), best_result.std_train_accuracy.item(), best_result.mean_test_accuracy.item(),  best_result.std_test_accuracy.item(),
                            best_result.mean_train_precision.item(), best_result.std_train_precision.item(), best_result.mean_test_precision.item(),  best_result.std_test_precision.item(),
                            best_result.mean_train_recall.item(), best_result.std_train_recall.item(), best_result.mean_test_recall.item(),  best_result.std_test_recall.item(),
                            ]

results_df.to_csv(f'{output_path}/results/D2_XGBClassifier_{vaccine_type}_CVTest_{time.strftime("%Y-%m-%d")}.csv', index=False)

plt.style.use('fivethirtyeight')

fig, ax1 = plt.subplots(figsize=(15, 10))


temp = results_df[['Symptom', 'Best_Val_AUC_Mean', 'Best_Val_Accuracy_Mean', 'Best_Val_Precision_Mean', 'Best_Val_Recall_Mean']].melt(id_vars='Symptom').rename(columns=str.title)

sns.barplot(data=temp, x='Symptom', y='Value', hue='Variable', ax=ax1)

plt.xticks(rotation=45)
lg = plt.legend()
lg.get_texts()[0].set_text('AUC-ROC')
lg.get_texts()[1].set_text('Accuracy')
lg.get_texts()[2].set_text('Precision')
lg.get_texts()[3].set_text('Recall')

plt.ylabel('Performance on 5-Fold Cross Validation')
plt.title(f'Dose 2 XGBClassifier {name_dict[vaccine_type]}')

fig.savefig(f'{output_path}/image/D2_XGBClassifier_{vaccine_type}_CVTest_{time.strftime("%Y-%m-%d")}.png', bbox_inches='tight')

logging.info('XGBoost finished')

## RF
logging.info('RF started')

from sklearn.ensemble import RandomForestClassifier

results_df = pd.DataFrame(columns=['Symptom', 'Baseline_AUC_Mean', 'Baseline_AUC_Std', 'Baseline_Accuracy_Mean', 'Baseline_Accuracy_Std', 'Best_Param',
                                   'Best_Train_AUC_Mean', 'Best_Train_AUC_Std', 'Best_Val_AUC_Mean', 'Best_Val_AUC_Std',
                                   'Best_Train_Accuracy_Mean', 'Best_Train_Accuracy_Std', 'Best_Val_Accuracy_Mean', 'Best_Val_Accuracy_Std', 
                                   'Best_Train_Precision_Mean', 'Best_Train_Precision_Std', 'Best_Val_Precision_Mean', 'Best_Val_Precision_Std',
                                   'Best_Train_Recall_Mean', 'Best_Train_Recall_Std', 'Best_Val_Recall_Mean', 'Best_Val_Recall_Std'])

for index, target in enumerate(dose_last_columns):
    print('***',target,'***')
    
    X = data.drop(dose_last_columns_full, axis=1)
    y = data[target]
    
    cross_val = StratifiedKFold(n_splits=3)
    
    baseline_model = RandomForestClassifier()
    auc = cross_val_score(baseline_model, X, y, scoring='roc_auc', cv=cross_val, n_jobs=-1)
    auc_mean = auc.mean()
    auc_std = auc.std()

    baseline_model = RandomForestClassifier()
    acc = cross_val_score(baseline_model, X, y, scoring='accuracy', cv=cross_val, n_jobs=-1)
    acc_mean = acc.mean()
    acc_std = acc.std()
    
    model = RandomForestClassifier()
    param_grid = {'n_estimators': [50, 100, 150],
                  'max_depth':[6,8,9,10],
                  'min_samples_split':[1,2,3,4]}

    grid = GridSearchCV(model, param_grid, scoring=['roc_auc', 'accuracy', 'precision', 'recall'], n_jobs=-1,
                        refit='roc_auc', cv=cross_val, verbose=0, return_train_score=True)

    grid.fit(X, y)
    
    results = pd.DataFrame(grid.cv_results_)
    best_result = results[results.rank_test_roc_auc == 1].head(1)
    
    
    results_df.loc[index] = [target, auc_mean, auc_std, acc_mean, acc_std, grid.best_params_,
                            best_result.mean_train_roc_auc.item(), best_result.std_train_roc_auc.item(), best_result.mean_test_roc_auc.item(),  best_result.std_test_roc_auc.item(),
                            best_result.mean_train_accuracy.item(), best_result.std_train_accuracy.item(), best_result.mean_test_accuracy.item(),  best_result.std_test_accuracy.item(),
                            best_result.mean_train_precision.item(), best_result.std_train_precision.item(), best_result.mean_test_precision.item(),  best_result.std_test_precision.item(),
                            best_result.mean_train_recall.item(), best_result.std_train_recall.item(), best_result.mean_test_recall.item(),  best_result.std_test_recall.item(),
                            ]

results_df.to_csv(f'{output_path}/results/D2_RF_{vaccine_type}_CVTest_{time.strftime("%Y-%m-%d")}.csv', index=False)

plt.style.use('fivethirtyeight')

fig, ax1 = plt.subplots(figsize=(15, 10))


temp = results_df[['Symptom', 'Best_Val_AUC_Mean', 'Best_Val_Accuracy_Mean', 'Best_Val_Precision_Mean', 'Best_Val_Recall_Mean']].melt(id_vars='Symptom').rename(columns=str.title)

sns.barplot(data=temp, x='Symptom', y='Value', hue='Variable', ax=ax1)

plt.xticks(rotation=45)
lg = plt.legend()
lg.get_texts()[0].set_text('AUC-ROC')
lg.get_texts()[1].set_text('Accuracy')
lg.get_texts()[2].set_text('Precision')
lg.get_texts()[3].set_text('Recall')

plt.ylabel('Performance on 5-Fold Cross Validation')
plt.title(f'Dose 2 RF {name_dict[vaccine_type]}')

fig.savefig(f'{output_path}/image/D2_RF_{vaccine_type}_CVTest_{time.strftime("%Y-%m-%d")}.png', bbox_inches='tight')

logging.info('RF finished')

## MLP
logging.info('MLP started')

from sklearn.neural_network import MLPClassifier


results_df = pd.DataFrame(columns=['Symptom', 'Baseline_AUC_Mean', 'Baseline_AUC_Std', 'Baseline_Accuracy_Mean', 'Baseline_Accuracy_Std', 'Best_Param',
                                   'Best_Train_AUC_Mean', 'Best_Train_AUC_Std', 'Best_Val_AUC_Mean', 'Best_Val_AUC_Std',
                                   'Best_Train_Accuracy_Mean', 'Best_Train_Accuracy_Std', 'Best_Val_Accuracy_Mean', 'Best_Val_Accuracy_Std', 
                                   'Best_Train_Precision_Mean', 'Best_Train_Precision_Std', 'Best_Val_Precision_Mean', 'Best_Val_Precision_Std',
                                   'Best_Train_Recall_Mean', 'Best_Train_Recall_Std', 'Best_Val_Recall_Mean', 'Best_Val_Recall_Std'])

for index, target in enumerate(dose_last_columns):
    print('***',target,'***')
    
    X = data.drop(dose_last_columns_full, axis=1)
    y = data[target]
    
    cross_val = StratifiedKFold(n_splits=3)
    
    baseline_model = MLPClassifier()
    auc = cross_val_score(baseline_model, X, y, scoring='roc_auc', cv=cross_val, n_jobs=-1)
    auc_mean = auc.mean()
    auc_std = auc.std()

    baseline_model = MLPClassifier()
    acc = cross_val_score(baseline_model, X, y, scoring='accuracy', cv=cross_val, n_jobs=-1)
    acc_mean = acc.mean()
    acc_std = acc.std()
    
    model = MLPClassifier()
    param_grid = {'solver': ['sgd'], 
                  'hidden_layer_sizes': [(100,), (200,), (100, 100)],
                  'learning_rate': ['constant', 'adaptive'],
                  'activation': ['tanh', 'relu']}

    grid = GridSearchCV(model, param_grid, scoring=['roc_auc', 'accuracy', 'precision', 'recall'],
                              n_jobs=-1, refit='roc_auc', cv=cross_val, verbose=0, return_train_score=True)

    grid.fit(X, y)
    
    results = pd.DataFrame(grid.cv_results_)
    best_result = results[results.rank_test_roc_auc == 1].head(1)
    
    
    results_df.loc[index] = [target, auc_mean, auc_std, acc_mean, acc_std, grid.best_params_,
                            best_result.mean_train_roc_auc.item(), best_result.std_train_roc_auc.item(), best_result.mean_test_roc_auc.item(),  best_result.std_test_roc_auc.item(),
                            best_result.mean_train_accuracy.item(), best_result.std_train_accuracy.item(), best_result.mean_test_accuracy.item(),  best_result.std_test_accuracy.item(),
                            best_result.mean_train_precision.item(), best_result.std_train_precision.item(), best_result.mean_test_precision.item(),  best_result.std_test_precision.item(),
                            best_result.mean_train_recall.item(), best_result.std_train_recall.item(), best_result.mean_test_recall.item(),  best_result.std_test_recall.item(),
                            ]

results_df.to_csv(f'{output_path}/results/D2_MLP_{vaccine_type}_CVTest_{time.strftime("%Y-%m-%d")}.csv', index=False)

plt.style.use('fivethirtyeight')

fig, ax1 = plt.subplots(figsize=(15, 10))

temp = results_df[['Symptom', 'Best_Val_AUC_Mean', 'Best_Val_Accuracy_Mean', 'Best_Val_Precision_Mean', 'Best_Val_Recall_Mean']].melt(id_vars='Symptom').rename(columns=str.title)

sns.barplot(data=temp, x='Symptom', y='Value', hue='Variable', ax=ax1)

plt.xticks(rotation=45)
lg = plt.legend()
lg.get_texts()[0].set_text('AUC-ROC')
lg.get_texts()[1].set_text('Accuracy')
lg.get_texts()[2].set_text('Precision')
lg.get_texts()[3].set_text('Recall')

plt.ylabel('Performance on 5-Fold Cross Validation')
plt.title(f'Dose 2 MLP {name_dict[vaccine_type]}')

fig.savefig(f'{output_path}/image/D2_MLP_{vaccine_type}_CVTest_{time.strftime("%Y-%m-%d")}.png', bbox_inches='tight')

logging.info('MLP finished')

## KNN

logging.info('KNN started')

from sklearn.neighbors import KNeighborsClassifier

results_df = pd.DataFrame(columns=['Symptom', 'Baseline_AUC_Mean', 'Baseline_AUC_Std', 'Baseline_Accuracy_Mean', 'Baseline_Accuracy_Std', 'Best_Param',
                                   'Best_Train_AUC_Mean', 'Best_Train_AUC_Std', 'Best_Val_AUC_Mean', 'Best_Val_AUC_Std',
                                   'Best_Train_Accuracy_Mean', 'Best_Train_Accuracy_Std', 'Best_Val_Accuracy_Mean', 'Best_Val_Accuracy_Std', 
                                   'Best_Train_Precision_Mean', 'Best_Train_Precision_Std', 'Best_Val_Precision_Mean', 'Best_Val_Precision_Std',
                                   'Best_Train_Recall_Mean', 'Best_Train_Recall_Std', 'Best_Val_Recall_Mean', 'Best_Val_Recall_Std'])

for index, target in enumerate(dose_last_columns):
    print('***',target,'***')
    
    X = data.drop(dose_last_columns_full, axis=1)
    y = data[target]
    
    cross_val = StratifiedKFold(n_splits=3)
    
    baseline_model = KNeighborsClassifier()
    auc = cross_val_score(baseline_model, X, y, scoring='roc_auc', cv=cross_val, n_jobs=-1)
    auc_mean = auc.mean()
    auc_std = auc.std()

    baseline_model = KNeighborsClassifier()
    acc = cross_val_score(baseline_model, X, y, scoring='accuracy', cv=cross_val, n_jobs=-1)
    acc_mean = acc.mean()
    acc_std = acc.std()
    
    model = KNeighborsClassifier()
    param_grid = {'n_neighbors': [1,3,5,7,9,11,13],
                  'weights': ['uniform', 'distance']}

    grid = GridSearchCV(model, param_grid, scoring=['roc_auc', 'accuracy', 'precision', 'recall'], n_jobs=-1,
                        refit='roc_auc', cv=cross_val, verbose=0, return_train_score=True)

    grid.fit(X, y)
    
    results = pd.DataFrame(grid.cv_results_)
    best_result = results[results.rank_test_roc_auc == 1].head(1)
    
    
    results_df.loc[index] = [target, auc_mean, auc_std, acc_mean, acc_std, grid.best_params_,
                            best_result.mean_train_roc_auc.item(), best_result.std_train_roc_auc.item(), best_result.mean_test_roc_auc.item(),  best_result.std_test_roc_auc.item(),
                            best_result.mean_train_accuracy.item(), best_result.std_train_accuracy.item(), best_result.mean_test_accuracy.item(),  best_result.std_test_accuracy.item(),
                            best_result.mean_train_precision.item(), best_result.std_train_precision.item(), best_result.mean_test_precision.item(),  best_result.std_test_precision.item(),
                            best_result.mean_train_recall.item(), best_result.std_train_recall.item(), best_result.mean_test_recall.item(),  best_result.std_test_recall.item(),
                            ]

results_df.to_csv(f'{output_path}/results/D2_KNN_{vaccine_type}_CVTest_{time.strftime("%Y-%m-%d")}.csv', index=False)

plt.style.use('fivethirtyeight')

fig, ax1 = plt.subplots(figsize=(15, 10))


temp = results_df[['Symptom', 'Best_Val_AUC_Mean', 'Best_Val_Accuracy_Mean', 'Best_Val_Precision_Mean', 'Best_Val_Recall_Mean']].melt(id_vars='Symptom').rename(columns=str.title)

sns.barplot(data=temp, x='Symptom', y='Value', hue='Variable', ax=ax1)

plt.xticks(rotation=45)
lg = plt.legend()
lg.get_texts()[0].set_text('AUC-ROC')
lg.get_texts()[1].set_text('Accuracy')
lg.get_texts()[2].set_text('Precision')
lg.get_texts()[3].set_text('Recall')

plt.ylabel('Performance on 5-Fold Cross Validation')
plt.title(f'Dose 2 KNN {name_dict[vaccine_type]}')

fig.savefig(f'{output_path}/image/D2_KNN_{vaccine_type}_CVTest_{time.strftime("%Y-%m-%d")}.png', bbox_inches='tight')

logging.info('KNN finished')