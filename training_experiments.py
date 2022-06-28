import os
import sys
import random
import time
import joblib

import numpy as np
import scipy
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report

import xgboost as xgb

ROOT_DIR = os.path.abspath(os.curdir)

train_dataset_ohe_form = joblib.load(ROOT_DIR + '/samples/' + 'train_ohe.pickle')
test_dataset_ohe_form = joblib.load(ROOT_DIR + '/samples/' + 'test_ohe.pickle')
val_dataset_ohe_form = joblib.load(ROOT_DIR + '/samples/' + 'val_ohe.pickle')

train_dataset_ohe_form_drop_id_target = train_dataset_ohe_form.drop(['ISU', 'ST_YEAR', 'SEMESTER','DISC_ID', 'TYPE_NAME', 'DEBT'],axis=1)
cols = list(train_dataset_ohe_form_drop_id_target.columns)

for col in cols:
    if pd.api.types.is_object_dtype(train_dataset_ohe_form_drop_id_target[col]):
        train_dataset_ohe_form_drop_id_target[col] = train_dataset_ohe_form_drop_id_target[col].astype('int')

# train_samples = xgb.DMatrix(train_dataset_ohe_form_drop_id_target,enable_categorical=False)
# train_target = xgb.DMatrix(pd.DataFrame(train_dataset_ohe_form['DEBT']),enable_categorical=False)
# print(train_samples.feature_names,train_target.feature_names)

test_dataset_ohe_form_drop_id_target = test_dataset_ohe_form.drop(['ISU', 'ST_YEAR', 'SEMESTER', 'DISC_ID', 'TYPE_NAME', 'DEBT'],axis=1)
cols = list(test_dataset_ohe_form_drop_id_target.columns)

for col in cols:
    if pd.api.types.is_object_dtype(test_dataset_ohe_form_drop_id_target[col]):
        test_dataset_ohe_form_drop_id_target[col] = test_dataset_ohe_form_drop_id_target[col].astype('int')
        
# test_samples = xgb.DMatrix(test_dataset_ohe_form_drop_id_target,enable_categorical=False)
# test_target = xgb.DMatrix(pd.DataFrame(test_dataset_ohe_form['DEBT']),enable_categorical=False)
# print(test_samples.feature_names,test_target.feature_names)

#https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=XGBClassifier#xgboost.XGBClassifier

# clf = xgb.XGBClassifier(n_estimators=300, max_depth=5, max_leaves=50, n_jobs=-1,random_state=42)
# clf.fit(train_dataset_ohe_form_drop_id_target, train_dataset_ohe_form['DEBT'],eval_set=[(test_dataset_ohe_form_drop_id_target, test_dataset_ohe_form['DEBT'])])
# # clf.save_model(os.path.join(output_dir, "one-hot.json"))
# y_score = clf.predict_proba(test_samples)[:, 1]  # proba of positive samples
# auc = roc_auc_score(test_dataset_ohe_form_drop_id_target, y_score)
# print("AUC: ", auc)


from sklearn.preprocessing import normalize

float_features = ['DEBT_MEAN', 'DEBT_SUM', 'DEBT_COUNT', 'DISC_DEBT_MEAN',
       'DISC_DEBT_SUM', 'DISC_DEBT_COUNT', 'CHOICE','ADMITTED_EXAM_1',
       'ADMITTED_EXAM_2', 'ADMITTED_EXAM_3']

for col in float_features:
    mean_value = train_dataset_ohe_form_drop_id_target[col].mean()
    std_value = train_dataset_ohe_form_drop_id_target[col].std()
    train_dataset_ohe_form_drop_id_target[col] = train_dataset_ohe_form_drop_id_target[col].apply(lambda x: (x-mean_value)/std_value)

for col in float_features:
    mean_value = test_dataset_ohe_form_drop_id_target[col].mean()
    std_value = test_dataset_ohe_form_drop_id_target[col].std()
    test_dataset_ohe_form_drop_id_target[col] = test_dataset_ohe_form_drop_id_target[col].apply(lambda x: (x-mean_value)/std_value)


from pyod.models.abod import ABOD

# clf_name = 'ABOD'
# clf = ABOD(contamination = 0.25)
# clf.fit(test_dataset_ohe_form_drop_id_target)
# labels = clf.predict(test_dataset_ohe_form_drop_id_target)
# test_dataset_ohe_form_drop_id_target['label']=labels
# print(test_dataset_ohe_form_drop_id_target['label'].value_counts())
#


from sklearn.linear_model import LogisticRegression

train = pd.concat((train_dataset_ohe_form_drop_id_target,test_dataset_ohe_form_drop_id_target),axis=0)

# from pyod.models.abod import ABOD
#
# clf_name = 'ABOD'
# clf = ABOD(contamination = 0.01)
# clf.fit(train)
# labels = clf.predict(train)
# train['label']=labels
# print(train['label'].value_counts())

y_train = pd.concat((train_dataset_ohe_form['DEBT'],test_dataset_ohe_form['DEBT']),axis=0)
train['DEBT'] = y_train

#drop outliers
#train = train[train['label']==0]

#train_permutation = train.sample(train.shape[0])

# train = train_dataset_ohe_form_drop_id_target
test = test_dataset_ohe_form_drop_id_target
# y_train = train_dataset_ohe_form['DEBT']
y_test = test_dataset_ohe_form['DEBT']

model = LogisticRegression(class_weight='balanced',C=0.1, penalty='l2', max_iter=300, n_jobs=-1)
model.fit(train.drop(['DEBT'],axis=1), y_train)
preds_test =  model.predict(test_dataset_ohe_form_drop_id_target)

print('f1 score', f1_score(y_test, preds_test))
print('accuracy score', accuracy_score(y_test, preds_test))
print('precision score', precision_score(y_test, preds_test))
print('recall score', recall_score(y_test, preds_test))

preds_train =  model.predict(train.drop(['DEBT'],axis=1))
print('f1 score', f1_score(y_train, preds_train))
print('accuracy score', accuracy_score(y_train, preds_train))
print('precision score', precision_score(y_train, preds_train))
print('recall score', recall_score(y_train, preds_train))

joblib.dump(model,'./samples/LogReg_best_median.pickle')

#
# from sklearn.ensemble import GradientBoostingClassifier
# gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.2, max_depth=6, random_state=0, min_samples_split=4, min_samples_leaf=1)
# gbc.fit(train, y_train)
#
# preds_train =  gbc.predict(train_dataset_ohe_form_drop_id_target)
# y_train = train_dataset_ohe_form['DEBT']
#
# print('f1 score', f1_score(y_train, preds_train))
# print('accuracy score', accuracy_score(y_train, preds_train))
# print('precision score', precision_score(y_train, preds_train))
# print('recall score', recall_score(y_train, preds_train))
#
# preds_test =  gbc.predict(test_dataset_ohe_form_drop_id_target)
#
# print('f1 score', f1_score(y_test, preds_test))
# print('accuracy score', accuracy_score(y_test, preds_test))
# print('precision score', precision_score(y_test, preds_test))
# print('recall score', recall_score(y_test, preds_test))
#
# joblib.dump(gbc,'./samples/GBC_best_5.pickle')

from sklearn.svm import SVC

svc_cls = SVC(C=0.1, kernel='sigmoid',  gamma='auto')

svc_cls.fit(train.drop(['DEBT'],axis=1), y_train)
preds_test =  svc_cls.predict(test_dataset_ohe_form_drop_id_target)

print('f1 score', f1_score(y_test, preds_test))
print('accuracy score', accuracy_score(y_test, preds_test))
print('precision score', precision_score(y_test, preds_test))
print('recall score', recall_score(y_test, preds_test))

preds_train =  svc_cls.predict(train.drop(['DEBT'],axis=1))
print('f1 score', f1_score(y_train, preds_train))
print('accuracy score', accuracy_score(y_train, preds_train))
print('precision score', precision_score(y_train, preds_train))
print('recall score', recall_score(y_train, preds_train))


joblib.dump(svc_cls,'./samples/svc_cls.pickle')

# model = LogisticRegression(class_weight='balanced',C=1)
# model.fit(train_dataset_ohe_form_drop_id_target, train_dataset_ohe_form['DEBT'])
# preds_train = model.predict(train_dataset_ohe_form_drop_id_target)
# y_train = train_dataset_ohe_form['DEBT']
#
# print('f1 score', f1_score(y_train, preds_train))
# print('accuracy score', accuracy_score(y_train, preds_train))
# print('precision score', precision_score(y_train, preds_train))
# print('recall score', recall_score(y_train, preds_train))
#
# preds_test =  model.predict(test_dataset_ohe_form_drop_id_target)
# y_test = test_dataset_ohe_form['DEBT']
#
# print('f1 score', f1_score(y_test, preds_test))
# print('accuracy score', accuracy_score(y_test, preds_test))
# print('precision score', precision_score(y_test, preds_test))
# print('recall score', recall_score(y_test, preds_test))

#submission

#val_dataset_ohe_form = joblib.load(ROOT_DIR + '/samples/' + 'val_ohe.pickle')

val_dataset_ohe_form_drop_id_target = val_dataset_ohe_form.drop(['ISU', 'ST_YEAR', 'SEMESTER', 'DISC_ID', 'TYPE_NAME'],axis=1)
cols = list(val_dataset_ohe_form_drop_id_target.columns)

for col in cols:
    if pd.api.types.is_object_dtype(val_dataset_ohe_form_drop_id_target[col]):
        val_dataset_ohe_form_drop_id_target[col] = val_dataset_ohe_form_drop_id_target[col].astype('int')
    else:
        val_dataset_ohe_form_drop_id_target[col] = val_dataset_ohe_form_drop_id_target[col].fillna(val_dataset_ohe_form_drop_id_target[col].median())

float_features = ['DEBT_MEAN', 'DEBT_SUM', 'DEBT_COUNT', 'DISC_DEBT_MEAN',
       'DISC_DEBT_SUM', 'DISC_DEBT_COUNT', 'CHOICE','ADMITTED_EXAM_1',
       'ADMITTED_EXAM_2', 'ADMITTED_EXAM_3']

for col in float_features:
    mean_value = val_dataset_ohe_form_drop_id_target[col].mean()
    std_value = val_dataset_ohe_form_drop_id_target[col].std()
    val_dataset_ohe_form_drop_id_target[col] = val_dataset_ohe_form_drop_id_target[col].apply(lambda x: (x-mean_value)/std_value)

preds_val =  svc_cls.predict(val_dataset_ohe_form_drop_id_target)

preds_val = pd.DataFrame(preds_val, columns=['DEBT'])
submission = pd.concat((val_dataset_ohe_form[['ISU', 'ST_YEAR', 'DISC_ID', 'SEMESTER', 'TYPE_NAME']], preds_val),axis=1)
submission = submission.drop_duplicates(subset=['ISU', 'ST_YEAR', 'DISC_ID', 'SEMESTER', 'TYPE_NAME'])

cols_test = ['ISU', 'ST_YEAR', 'SEMESTER', 'DISC_ID', 'TYPE_NAME']
test = pd.read_csv(ROOT_DIR + '/samples/' + 'test.csv', dtype=object,sep=',', header=1, names=cols_test)


test['ID'] = test[['ISU', 'ST_YEAR', 'DISC_ID', 'SEMESTER', 'TYPE_NAME']].apply(
        lambda x: f'ISU:{x[0]} | ST_YEAR:{x[1]} | DISC_ID:{x[2]} | SEMESTER:{x[3]} | TYPE_NAME:{x[4]}', axis =1)

submit_2 = test.merge(submission, on=['ISU', 'DISC_ID', 'TYPE_NAME'],how='inner')

# submit_2['ID'] = 'ISU:'+submit_2['ISU']+' | '+'DISC_ID:'+submit_2['DISC_ID']+' | '+'TYPE_NAME:'+submit_2['TYPE_NAME']
submit_2 = submit_2[['ID','DEBT']]

submit_2.to_csv(ROOT_DIR + '/samples/' + 'svc_cls.csv',sep=',',index=False)

# submit_4_lr_best = pd.read_csv(ROOT_DIR + '/samples/' + 'submit_4_lr_best.csv',sep=',',dtype=object)


#3. скачать заново все данные


