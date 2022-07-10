#/itmo_contest/preprocessing.py created by: Nikolay Pavlychev pavlychev.n.se@gmail.com

import os
import sys
import random
import numpy as np
np.random.seed(1984)
import time
import joblib
import json

import scipy
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

import itertools
from itertools import product

import xgboost as xgb

def undummify(df, prefix_sep="__"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

ROOT_DIR = os.path.abspath(os.curdir)

train_dataset_ohe_form = joblib.load(ROOT_DIR + '/samples/' + 'data_train_ohe.pickle')
val_dataset_ohe_form = joblib.load(ROOT_DIR + '/samples/' + 'data_val_ohe.pickle')

train = train_dataset_ohe_form[train_dataset_ohe_form['ST_YEAR'].isin([2018,2019])].drop(['ISU', 'ST_YEAR', 'SEMESTER','DISC_ID'],axis=1)
test = train_dataset_ohe_form[train_dataset_ohe_form['ST_YEAR'].isin([2020])].drop(['ISU', 'ST_YEAR', 'SEMESTER','DISC_ID'],axis=1)
y_train = train['DEBT']
y_test = test['DEBT']

print(train.shape,test.shape,y_train.shape,y_test.shape)

params_dict_list = {'C': [0.0005, 0.001, 0.01, 0.1, 1, 10], 'penalty': ['l2','elasticnet'], 'solver':['saga'],
                    'l1_ratio':[0.3,0.4,0.5,0.6],'class_weight':['balanced'], 'n_jobs': [-1], 'random_state':[42],
                    'max_iter':[800]}
keys, values = zip(*params_dict_list.items())
permutations_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
print('count of experiments is ',len(permutations_params_dict_list))

f1_score_score_test_max = []
f1_score_score_train_max = []
index_test_max = []

for k, item in enumerate(permutations_params_dict_list):

    model = LogisticRegression(class_weight= item['class_weight'], C= item['C'], penalty= item['penalty'], solver=item['solver'], l1_ratio=item['l1_ratio'],max_iter= item['max_iter'], n_jobs= item['n_jobs'], random_state= item['random_state'])
    model.fit(train.drop(['DEBT'], axis=1), y_train)

    pred_proba_df = pd.DataFrame(model.predict_proba(test.drop(['DEBT'], axis=1)))


    threshold_list = [0.4,0.5,0.6]
    f1_score_threshold_list = []

    for i in threshold_list:

        y_test_pred = pred_proba_df.applymap(lambda x: 1 if x > i else 0)
        test_f1_score = f1_score(y_test.to_numpy().reshape(y_test.to_numpy().size, 1),
                                               y_test_pred.iloc[:, 1].to_numpy().reshape(
                                                   y_test_pred.iloc[:, 1].to_numpy().size, 1))

        f1_score_threshold_list.append(test_f1_score)

    tmp = max(f1_score_threshold_list)
    index = f1_score_threshold_list.index(tmp)
    th_max = threshold_list[index]
    i_opt = th_max
    y_test_pred = pred_proba_df.applymap(lambda x: 1 if x > i_opt else 0)
    test_f1_score = f1_score(y_test.to_numpy().reshape(y_test.to_numpy().size, 1),
                               y_test_pred.iloc[:, 1].to_numpy().reshape(
                                   y_test_pred.iloc[:, 1].to_numpy().size, 1))

    pred_proba_df = pd.DataFrame(model.predict_proba(train.drop(['DEBT'], axis=1)))
    y_train_pred = pred_proba_df.applymap(lambda x: 1 if x > i_opt else 0)
    train_f1_score = f1_score(y_train.to_numpy().reshape(y_train.to_numpy().size, 1),
                               y_train_pred.iloc[:, 1].to_numpy().reshape(
                                   y_train_pred.iloc[:, 1].to_numpy().size, 1))

    print('test_f1_score ',test_f1_score)
    print('train_f1_score ', train_f1_score)
    print('th_max ', th_max)
    f1_score_score_test_max.append(test_f1_score)
    f1_score_score_train_max.append(train_f1_score)
    index_test_max.append(th_max)

    break

print('f1_score_score_test_max = ',np.max(f1_score_score_test_max))
print('f1_score_score_train_max = ', np.max(f1_score_score_train_max))

tmp = max(f1_score_score_test_max)
index = f1_score_score_test_max.index(tmp)
th_max = index_test_max[index]

print('opt experiment is ',index)
print('opt threshold is ',th_max)


metrics ={}
metrics.update({'f1_score_score_test_max':np.max(f1_score_score_test_max)})
metrics.update({'f1_score_score_train_max':np.max(f1_score_score_train_max)})
metrics.update({'opt_experiment': index})
metrics.update({'opt_threshold': th_max})


train_probs = train_dataset_ohe_form[train_dataset_ohe_form['ST_YEAR'].isin([2018,2019])][['ISU', 'ST_YEAR', 'SEMESTER','DISC_ID']]
test_probs = train_dataset_ohe_form[train_dataset_ohe_form['ST_YEAR'].isin([2020])][['ISU', 'ST_YEAR', 'SEMESTER','DISC_ID']]
train_probs['prob'] = model.predict_proba(train.drop(['DEBT'], axis=1))[:,1]
test_probs['prob'] = model.predict_proba(test.drop(['DEBT'], axis=1))[:,1]

train_dataset_type_name = train_dataset_ohe_form[['TYPE_NAME__Дифференцированный зачет', 'TYPE_NAME__Зачет', 'TYPE_NAME__Курсовой проект', 'TYPE_NAME__Экзамен']]
train_dataset_type_name = undummify(train_dataset_type_name)
train_probs['TYPE_NAME'] = train_dataset_type_name

test_dataset_type_name = train_dataset_ohe_form[['TYPE_NAME__Дифференцированный зачет', 'TYPE_NAME__Зачет', 'TYPE_NAME__Курсовой проект', 'TYPE_NAME__Экзамен']]
test_dataset_type_name = undummify(test_dataset_type_name)
test_probs['TYPE_NAME'] = test_dataset_type_name

print(train_probs.info())
print(test_probs.info())
print('Successfully!')


#-----------------------------------------------------------------------------------------------------------------------
#submission

preds_val =  model.predict(val_dataset_ohe_form.drop(['ISU', 'ST_YEAR', 'SEMESTER','DISC_ID'],axis=1))
preds_val = pd.DataFrame(preds_val, columns=['DEBT'])

val_dataset_type_name = val_dataset_ohe_form[['TYPE_NAME__Дифференцированный зачет', 'TYPE_NAME__Зачет', 'TYPE_NAME__Курсовой проект', 'TYPE_NAME__Экзамен']]
val_dataset_type_name = undummify(val_dataset_type_name)
val_dataset_ohe_form['TYPE_NAME'] = val_dataset_type_name

submission = pd.concat((val_dataset_ohe_form[['ISU', 'ST_YEAR', 'DISC_ID', 'SEMESTER','TYPE_NAME']], preds_val),axis=1)
submission = submission.drop_duplicates(subset=['ISU', 'ST_YEAR', 'DISC_ID', 'SEMESTER','TYPE_NAME'])


cols_test = ['ISU', 'ST_YEAR', 'SEMESTER', 'DISC_ID', 'TYPE_NAME']
test = pd.read_csv(ROOT_DIR + '/samples/' + 'test.csv', dtype=object,sep=',', header=0, names=cols_test)


test['ID'] = test[['ISU', 'ST_YEAR', 'DISC_ID', 'SEMESTER', 'TYPE_NAME']].apply(
        lambda x: f'ISU:{x[0]} | ST_YEAR:{x[1]} | DISC_ID:{x[2]} | SEMESTER:{x[3]} | TYPE_NAME:{x[4]}', axis =1)

submit = test.merge(submission, on=['ISU', 'DISC_ID', 'TYPE_NAME'],how='inner')

submit = submit[['ID','DEBT']]
print(submit['DEBT'].value_counts())

submit.to_csv(ROOT_DIR + '/samples/' + 'LogReg_opt.csv',sep=',',index=False)


#-----------------------------------------------------------------------------------------------------------------------
#XGBOOST CLassifier

clf = xgb.XGBClassifier(n_estimators=200, learning_rate=0.7, max_depth=4, max_leaves =50,random_state=42,tree_method='gpu_hist', reg_alpha=0.00001)
clf.fit(train.drop(['DEBT'],axis=1), y_train)

preds_test =  clf.predict(test.drop(['DEBT'],axis=1))
print('f1 score', f1_score(y_test, preds_test))
print('accuracy score', accuracy_score(y_test, preds_test))
print('precision score', precision_score(y_test, preds_test))
print('recall score', recall_score(y_test, preds_test))

preds_train =  clf.predict(train.drop(['DEBT'],axis=1))
print('f1 score', f1_score(y_train, preds_train))
print('accuracy score', accuracy_score(y_train, preds_train))
print('precision score', precision_score(y_train, preds_train))
print('recall score', recall_score(y_train, preds_train))


model = clf
preds_val =  model.predict(val_dataset_ohe_form.drop(['ISU', 'ST_YEAR', 'SEMESTER','DISC_ID'],axis=1))
preds_val = pd.DataFrame(preds_val, columns=['DEBT'])

val_dataset_type_name = val_dataset_ohe_form[['TYPE_NAME__Дифференцированный зачет', 'TYPE_NAME__Зачет', 'TYPE_NAME__Курсовой проект', 'TYPE_NAME__Экзамен']]
val_dataset_type_name = undummify(val_dataset_type_name)
val_dataset_ohe_form['TYPE_NAME'] = val_dataset_type_name

submission = pd.concat((val_dataset_ohe_form[['ISU', 'ST_YEAR', 'DISC_ID', 'SEMESTER','TYPE_NAME']], preds_val),axis=1)
submission = submission.drop_duplicates(subset=['ISU', 'ST_YEAR', 'DISC_ID', 'SEMESTER','TYPE_NAME'])


cols_test = ['ISU', 'ST_YEAR', 'SEMESTER', 'DISC_ID', 'TYPE_NAME']
test = pd.read_csv(ROOT_DIR + '/samples/' + 'test.csv', dtype=object,sep=',', header=0, names=cols_test)


test['ID'] = test[['ISU', 'ST_YEAR', 'DISC_ID', 'SEMESTER', 'TYPE_NAME']].apply(
        lambda x: f'ISU:{x[0]} | ST_YEAR:{x[1]} | DISC_ID:{x[2]} | SEMESTER:{x[3]} | TYPE_NAME:{x[4]}', axis =1)

submit = test.merge(submission, on=['ISU', 'DISC_ID', 'TYPE_NAME'],how='inner')

submit = submit[['ID','DEBT']]
print(submit['DEBT'].value_counts())

submit.to_csv(ROOT_DIR + '/samples/' + 'xgboost_opt.csv',sep=',',index=False)
#-----------------------------------------------------------------------------------------------------------------------
