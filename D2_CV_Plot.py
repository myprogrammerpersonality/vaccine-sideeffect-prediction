import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
import matplotlib.font_manager as fm

mpl.rcParams['font.family']  = 'Avenir LT Std'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sys
import warnings
warnings.simplefilter("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import logging

logging.basicConfig(filename='T4.log', format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logging.info('Started')

random_state = 1234

print('*** Import is Done!')
vaccine_type = sys.argv[1]
input_data_path = sys.argv[2]
output_path = sys.argv[3]
time_to_put = '2021-08-16'
name_dict = {'SputnikV': 'Sputnik V',
             'AstraZeneca': 'AZD1222',
             'Sinopharm': 'BBIBP-CorV',
             'Covaxin': 'COVAXIN',
             'Barekat': 'COVIran Barekat'}

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


# Plot All
logging.info('Plotting started')

all_results = {'SideEffect':[], 'ModelType':[], 'ROC-AUC':[]}

for model in ['Logistic_Regression', 'SVM', 'XGBClassifier', 'RF', 'KNN', 'MLP']:
    temp = pd.read_csv(f'{output_path}/results/D2_{model}_{vaccine_type}_CVTest_{time_to_put}.csv')
    for sideeffect in dose_last_columns:
        all_results['SideEffect'].append(sideeffect)
        all_results['ModelType'].append(model)
        all_results['ROC-AUC'].append(temp[temp['Symptom'] == sideeffect].Best_Val_AUC_Mean.item())
    
all_results_df = pd.DataFrame(all_results)

plt.style.use('fivethirtyeight')

fig, ax1 = plt.subplots(figsize=(15, 10))

sns.barplot(data=all_results_df, x='SideEffect', y='ROC-AUC', hue='ModelType', ax=ax1)

plt.xticks(rotation=45)
plt.ylabel('AUC-ROC Performance on 5-Fold Cross Validation')
plt.title(f'Second Dose of {name_dict[vaccine_type]} Vaccine')
plt.legend(loc=4, facecolor="white")
plt.ylim(0.6,0.9)

fig.savefig(f'{output_path}/image/All_Models_{vaccine_type}_CVTest_{time_to_put}.png', bbox_inches='tight', dpi=200)

# Circle Plots
## Train
all_results = {'SideEffect':[], 'ModelType':[], 'ROC-AUC':[]}

for model in ['Logistic_Regression', 'SVM', 'XGBClassifier', 'RF', 'KNN', 'MLP']:
    temp = pd.read_csv(f'{output_path}/results/D2_{model}_{vaccine_type}_CVTest_{time_to_put}.csv')
    for sideeffect in dose_last_columns:
        all_results['SideEffect'].append(sideeffect)
        all_results['ModelType'].append(model)
        all_results['ROC-AUC'].append(temp[temp['Symptom'] == sideeffect].Best_Train_AUC_Mean.item())
    
all_results_df = pd.DataFrame(all_results)

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-darkgrid')

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis([-0.5,7.5,-0.5,5.5])

sns.scatterplot(x="SideEffect", y="ModelType", size = "ROC-AUC", data=all_results_df, sizes=(1000, 2000), legend=False, color='crimson')

for i, model in enumerate(['Logistic_Regression', 'SVM', 'XGBClassifier', 'RF', 'KNN', 'MLP']):
    for j, side in enumerate(dose_last_columns):
        value = all_results_df[all_results_df.SideEffect == side][all_results_df.ModelType == model]['ROC-AUC'].values[0]
        plt.annotate(f"{round(value,2)}", xy=(j, i), ha="center", va="center")

plt.xticks(rotation=45)
plt.title(f"Dose 2 of {name_dict[vaccine_type]}\nTraining Set")

fig.savefig(f'{output_path}/image/Train.png', bbox_inches='tight', dpi=200)
## Val
all_results = {'SideEffect':[], 'ModelType':[], 'ROC-AUC':[]}

for model in ['Logistic_Regression', 'SVM', 'XGBClassifier', 'RF', 'KNN', 'MLP']:
    temp = pd.read_csv(f'{output_path}/results/D2_{model}_{vaccine_type}_CVTest_{time_to_put}.csv')
    for sideeffect in dose_last_columns:
        all_results['SideEffect'].append(sideeffect)
        all_results['ModelType'].append(model)
        all_results['ROC-AUC'].append(temp[temp['Symptom'] == sideeffect].Best_Val_AUC_Mean.item())
    
all_results_df = pd.DataFrame(all_results)

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-darkgrid')

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis([-0.5,7.5,-0.5,5.5])

sns.scatterplot(x="SideEffect", y="ModelType", size = "ROC-AUC", data=all_results_df, sizes=(1000, 2000), legend=False, color='lightseagreen')

for i, model in enumerate(['Logistic_Regression', 'SVM', 'XGBClassifier', 'RF', 'KNN', 'MLP']):
    for j, side in enumerate(dose_last_columns):
        value = all_results_df[all_results_df.SideEffect == side][all_results_df.ModelType == model]['ROC-AUC'].values[0]
        plt.annotate(f"{round(value,2)}", xy=(j, i), ha="center", va="center")

plt.xticks(rotation=45)
plt.title("Validation Set")
plt.ylabel("Model Type")
plt.xlabel("Side Effect")

fig.savefig(f'{output_path}/image/Val.png', bbox_inches='tight', dpi=200)
## Test
all_results = {'SideEffect':[], 'ModelType':[], 'ROC-AUC':[]}

for model in ['Logistic_Regression', 'SVM', 'XGBClassifier', 'RF', 'KNN', 'MLP']:
    temp = pd.read_csv(f'{output_path}/results/D2_{model}_{vaccine_type}_CVTest_{time_to_put}.csv')
    for sideeffect in dose_last_columns:
        all_results['SideEffect'].append(sideeffect)
        all_results['ModelType'].append(model)
        all_results['ROC-AUC'].append(temp[temp['Symptom'] == sideeffect].Best_Test_AUC.item())
    
all_results_df = pd.DataFrame(all_results)

plt.style.use('fivethirtyeight')
plt.style.use('seaborn-darkgrid')

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis([-0.5,7.5,-0.5,5.5])

sns.scatterplot(x="SideEffect", y="ModelType", size = "ROC-AUC", data=all_results_df, sizes=(1000, 2000), legend=False, color='steelblue')

for i, model in enumerate(['Logistic_Regression', 'SVM', 'XGBClassifier', 'RF', 'KNN', 'MLP']):
    for j, side in enumerate(dose_last_columns):
        value = all_results_df[all_results_df.SideEffect == side][all_results_df.ModelType == model]['ROC-AUC'].values[0]
        plt.annotate(f"{round(value,2)}", xy=(j, i), ha="center", va="center")

plt.xticks(rotation=45)
plt.title("Test Set")
plt.xlabel("Side Effect")

fig.savefig(f'{output_path}/image/Test.png', bbox_inches='tight', dpi=200)

logging.info('Plotting finished')

## All Param
logging.info('Param started')

all_results = {'SideEffect':[], 'ModelType':[], 'Param':[]}

for model in ['Logistic_Regression', 'SVM', 'XGBClassifier', 'RF', 'KNN', 'MLP']:
    temp = pd.read_csv(f'{output_path}/results/D2_{model}_{vaccine_type}_CVTest_{time_to_put}.csv')
    for sideeffect in dose_last_columns:
        all_results['SideEffect'].append(sideeffect)
        all_results['ModelType'].append(model)
        param = temp[temp['Symptom'] == sideeffect].Best_Param.item()
        param = param.replace("'", "").replace("{", "").replace("}", "")
        all_results['Param'].append(param)
    
all_results_df = pd.DataFrame(all_results)

all_results_df.to_csv(f'D2_{vaccine_type}_Best_Param_{time_to_put}.tsv', index=False, sep='\t')

logging.info('Param finished')

## Feature importance
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

from sklearn.preprocessing import StandardScaler, MinMaxScaler

sc = MinMaxScaler()

data['Age'] = sc.fit_transform(data[['Age']].values)
data['Height'] = sc.fit_transform(data[['Height']].values)
data['Weight'] = sc.fit_transform(data[['Weight']].values)
data['BMI'] = sc.fit_transform(data[['BMI']].values)


data = data.drop(['Height', 'Weight'], axis=1)
data = data.drop(['Dose1_Needle_Pain', 'Dose1_Needle_Redness', 'Dose1_Needle_Swelling', 'Dose1_Other'], axis=1)

result_df = pd.read_csv(f'{output_path}/results/D2_Logistic_Regression_{vaccine_type}_CVTest_{time_to_put}.csv') 

coeff_df = pd.DataFrame(columns=dose_last_columns, index=data.drop(dose_last_columns_full, axis=1).columns, data=np.zeros((len(data.drop(dose_last_columns_full, axis=1).columns), len(dose_last_columns))))


for index, target in enumerate(dose_last_columns):
    print('***',target,'***')
    
    X = data.drop(dose_last_columns_full, axis=1)
    y = data[target]
    
    exec("best_param = " + result_df[result_df.Symptom == target].Best_Param.item())
    best_model = LogisticRegression(**best_param)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)
    best_model.fit(X_train, y_train)
    
    coeff_df[target] = best_model.coef_[0]
    
coeff_df.index = ['Sex', 'Age', 'A', 'AB', 'B', 'O', 'Smoking', 'Substance use',
       'Alcohol dependancy', 'Background Diabetes',
       'Background Cardiovascular disease', 'Background Hypertension',
       'Background CancerPassive', 'Background CancerActive',
       'Background Neurological', 'Background pulmonary',
       'Background ImmuneSystem', 'Background hematologic',
       'Background Gastrointestinal', 'Background Renal',
       'Background Hepatic', 'Background Skeletal', 'Background Mental',
       'Background Allergy', 'Background None', 'Hormonal med',
       'Pregnancy', 'Respiratory Inhaler', 'Corticosteroid Med',
       'Chemotherapy Med', 'Immunosuppressive Med', 'Covid Infection',
       'Covid Fever', 'Covid Fatigue', 'Covid Cough',
       'Covid Gastrointestinal', 'Covid Anosmia',
       'Covid RespiratoryProblems', 'Covid ConsciousnessDisorder',
       'Covid Paresis', 'Covid ChestPain', 'Covid Headache',
       'Covid SoreThroat', 'Covid Vertigo', 'Covid hospitalization',

       'Dose1_Fever', 'Dose1_Fatigue',
       'Dose1_Headache', 'Dose1_Nausea', 'Dose1_Chills','Dose1_Joint_Pain', 'Dose1_Muscle_Pain',
       'BMI', 'Dose1_Local']

order =['Sex', 'Age', 'A', 'AB', 'B', 'O', 'Smoking', 'Substance use',
       'Alcohol dependancy', 'Background Diabetes',
       'Background Cardiovascular disease', 'Background Hypertension',
       'Background CancerPassive', 'Background CancerActive',
       'Background Neurological', 'Background pulmonary',
       'Background ImmuneSystem', 'Background hematologic',
       'Background Gastrointestinal', 'Background Renal',
       'Background Hepatic', 'Background Skeletal', 'Background Mental',
       'Background Allergy', 'Background None', 'Hormonal med',
       'Pregnancy', 'Respiratory Inhaler', 'Corticosteroid Med',
       'Chemotherapy Med', 'Immunosuppressive Med', 'Covid Infection',
       'Covid Fever', 'Covid Fatigue', 'Covid Cough',
       'Covid Gastrointestinal', 'Covid Anosmia',
       'Covid RespiratoryProblems', 'Covid ConsciousnessDisorder',
       'Covid Paresis', 'Covid ChestPain', 'Covid Headache',
       'Covid SoreThroat', 'Covid Vertigo', 'Covid hospitalization',

       'Dose1_Fever', 'Dose1_Fatigue',
       'Dose1_Headache', 'Dose1_Nausea', 'Dose1_Chills','Dose1_Joint_Pain', 'Dose1_Muscle_Pain',
       'Dose1_Local', 'BMI']

coeff_df = coeff_df.loc[order, :]

coeff_df.to_csv(f'{output_path}/reports/D2_{vaccine_type}_feature_importance_{time_to_put}.csv')

dfu = coeff_df.unstack().reset_index()
dfu.columns = list('XYS')
dfu["Sign"] = dfu["S"].apply(lambda x: 'red' if x>=0 else 'blue')
dfu["S"] = dfu["S"].apply(lambda x: abs(x))


plt.style.use('fivethirtyeight')
plt.style.use('seaborn-darkgrid')

import matplotlib.patches as mpatches

red_patch = mpatches.Patch(color=(252/255,79/255,48/255,1), label='Increase Probability')
blue_patch = mpatches.Patch(color=(0/255,143/255,213/255,1), label='Decrease Probability')

fig, ax = plt.subplots(figsize=(10, 30))

sns.scatterplot(x="X", y="Y", size = "S", hue="Sign", hue_order=['blue', 'red'], data=dfu, sizes=(20, 2000), legend=False)
plt.xticks(rotation=45)
plt.title(f"Feature Effect {name_dict[vaccine_type]}")
plt.ylabel("Background")
plt.xlabel("Side Effect")

legend = plt.legend(frameon = 1, handles=[red_patch, blue_patch])
frame = legend.get_frame()
frame.set_color('white')

fig.savefig(f'{output_path}/image/LogisticRegression_{vaccine_type}_Feature_Effect_{time_to_put}.png', bbox_inches='tight', dpi=400)

logging.info('All finished')

