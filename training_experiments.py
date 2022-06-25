import os
import sys
import random
import time
import joblib

import numpy as np
import scipy
import pandas as pd
import sklearn
from sklearn.metrics import classification_report

import xgboost as xgb

ROOT_DIR = os.path.abspath(os.curdir)

train_dataset_ohe_form = joblib.load(ROOT_DIR + '/samples/' + 'train_ohe.pickle')
test_dataset_ohe_form = joblib.load(ROOT_DIR + '/samples/' + 'test_ohe.pickle')
val_dataset_ohe_form = joblib.load(ROOT_DIR + '/samples/' + 'val_ohe.pickle')

train_dataset_ohe_form_drop_id_target = train_dataset_ohe_form.drop(['ISU', 'DISC_ID', 'TYPE_NAME', 'DEBT'],axis=1)
cols = list(train_dataset_ohe_form_drop_id_target.columns)

for col in cols:
    if pd.api.types.is_object_dtype(train_dataset_ohe_form_drop_id_target[col]):
        train_dataset_ohe_form_drop_id_target[col] = train_dataset_ohe_form_drop_id_target[col].astype('int')

# train_samples = xgb.DMatrix(train_dataset_ohe_form_drop_id_target,enable_categorical=False)
# train_target = xgb.DMatrix(pd.DataFrame(train_dataset_ohe_form['DEBT']),enable_categorical=False)
# print(train_samples.feature_names,train_target.feature_names)

test_dataset_ohe_form_drop_id_target = test_dataset_ohe_form.drop(['ISU', 'DISC_ID', 'TYPE_NAME', 'DEBT'],axis=1)
cols = list(test_dataset_ohe_form_drop_id_target.columns)

for col in cols:
    if pd.api.types.is_object_dtype(test_dataset_ohe_form_drop_id_target[col]):
        test_dataset_ohe_form_drop_id_target[col] = test_dataset_ohe_form_drop_id_target[col].astype('int')
        
# test_samples = xgb.DMatrix(test_dataset_ohe_form_drop_id_target,enable_categorical=False)
# test_target = xgb.DMatrix(pd.DataFrame(test_dataset_ohe_form['DEBT']),enable_categorical=False)
# print(test_samples.feature_names,test_target.feature_names)

#https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=XGBClassifier#xgboost.XGBClassifier

clf = xgb.XGBClassifier(n_estimators=300, max_depth=5, max_leaves=50, n_jobs=-1,random_state=42, reg_lambda=10)
clf.fit(train_dataset_ohe_form_drop_id_target, train_dataset_ohe_form['DEBT'],eval_set=[(test_dataset_ohe_form_drop_id_target, test_dataset_ohe_form['DEBT']), (train_dataset_ohe_form_drop_id_target, train_dataset_ohe_form['DEBT'])])
# clf.save_model(os.path.join(output_dir, "one-hot.json"))
# y_score = clf.predict_proba(test_samples)[:, 1]  # proba of positive samples
# auc = roc_auc_score(test_dataset_ohe_form_drop_id_target, y_score)
# print("AUC: ", auc)

y_predict =  clf.predict(test_dataset_ohe_form_drop_id_target)

print(classification_report(y_predict, test_dataset_ohe_form['DEBT']))

y_predict =  clf.predict(train_dataset_ohe_form_drop_id_target)

print(classification_report(y_predict, train_dataset_ohe_form['DEBT']))

#submission

val_dataset_ohe_form = joblib.load(ROOT_DIR + '/samples/' + 'val_ohe.pickle')

val_dataset_ohe_form_drop_id_target = val_dataset_ohe_form.drop(['ISU', 'DISC_ID', 'TYPE_NAME'],axis=1)
cols = list(val_dataset_ohe_form_drop_id_target.columns)

for col in cols:
    if pd.api.types.is_object_dtype(val_dataset_ohe_form_drop_id_target[col]):
        val_dataset_ohe_form_drop_id_target[col] = val_dataset_ohe_form_drop_id_target[col].astype('int')

y_val_predict =  clf.predict(val_dataset_ohe_form_drop_id_target)

y_val_predict = pd.DataFrame(y_val_predict, columns=['DEBT'])
submission = pd.concat((val_dataset_ohe_form[['ISU', 'DISC_ID', 'TYPE_NAME']], y_val_predict),axis=1)

submission['ID'] = 'ISU:'+submission['ISU']+' | '+'DISC_ID:'+submission['DISC_ID']+' | '+'TYPE_NAME:'+submission['TYPE_NAME']
submission = submission[['ID','DEBT']]

submission.to_csv(ROOT_DIR + '/samples/' + 'submission_1.csv',sep=',',index=False)
